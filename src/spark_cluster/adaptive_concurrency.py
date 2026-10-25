"""Learn per-runtime admission limits from real traffic, without storing content."""

import asyncio
import contextlib
import hashlib
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from types import SimpleNamespace

from .inference_tracking import SCOPE_KEY, RequestRecord
from .pool_memory import command_value

MAX_CONCURRENCY = 64  # Also bounded by the existing gateway request queue.


def runtime_parallelism(payload: dict, *, prepare=False) -> int:
    """Only validated continuous-batching engines receive automatic launch flags."""
    command = payload.setdefault("Cmd", []) if prepare else payload.get("Cmd", [])
    evidence = str(payload.get("Image", "")) + " " + str(payload.get("Entrypoint", ""))
    evidence += " " + " ".join(str(item) for item in command[:3])
    evidence = evidence.casefold()
    flag = None
    if "vllm" in evidence:
        flag = "--max-num-seqs"
    elif "sglang" in evidence:
        flag = "--max-running-requests"
    if prepare and flag:
        for index, item in enumerate(command):
            if item == flag and index + 1 < len(command):
                command[index + 1] = str(MAX_CONCURRENCY)
                break
            if isinstance(item, str) and item.startswith(flag + "="):
                command[index] = flag + "=" + str(MAX_CONCURRENCY)
                break
        else:
            command.extend([flag, str(MAX_CONCURRENCY)])
        return MAX_CONCURRENCY
    flags = [flag] if flag else ["--parallel", "-np"]
    for candidate in flags:
        value = command_value(command, candidate)
        if value is not None:
            with contextlib.suppress(ValueError):
                return max(1, min(MAX_CONCURRENCY, int(value)))
    # llama.cpp's slots divide a fixed context allocation. Never manufacture slots
    # by shrinking the context promised to clients. Other services are measured.
    return 1 if "llama" in evidence or "gguf" in evidence else MAX_CONCURRENCY


def response_feedback(scope, model_id, path, send):
    record = scope.get(SCOPE_KEY)
    if record:
        return scope, record, send
    record = RequestRecord(SimpleNamespace(resources=lambda _: None), model_id, path)
    return dict(scope, **{SCOPE_KEY: record}), record, record.wrap_send(send)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class Admission:
    controller: "ConcurrencyController"
    record: RequestRecord | None
    epoch: int
    started: float | None = None
    perf_started: float | None = None

    def begin(self):
        if self.started is None:
            self.started = self.controller.clock()
            self.perf_started = time.perf_counter()
            self.controller._account(self.started)
            self.controller.serving += 1


class ConcurrencyController:
    def __init__(self, identity, ceiling=MAX_CONCURRENCY, learned=None, clock=time.monotonic):
        self.identity = identity
        self.ceiling = max(1, min(MAX_CONCURRENCY, ceiling))
        self.clock = clock
        self.limit = min(2, self.ceiling)
        self.condition = asyncio.Condition()
        self.queue = deque()
        self.running = self.serving = 0
        self.samples = 0
        self.best_limit = self.limit
        self.best_score = None
        self.metric = None
        self.baseline_latency = None
        self.state = "learning"
        self.reason = "Learning from real requests"
        self.pressure = None
        self.pressure_at = 0.0
        self.hot_samples = 0
        self.cooldown = 0
        self.probe_down = False
        self.last_backoff = -math.inf
        self.changed = False
        self.busy_seconds = 0.0
        self.accounted_at = clock()
        self.epoch = 0
        self._reset_window()
        if learned:
            try:
                if 0 <= time.time() - learned["updated_at"] < 7 * 86400:
                    self.limit = self.best_limit = max(1, min(self.ceiling, int(learned["limit"])))
                    self.reason = "Revalidating previously learned concurrency"
            except (KeyError, TypeError, ValueError, OverflowError):
                pass

    def _account(self, now):
        if self.serving:
            self.busy_seconds += max(0.0, now - self.accounted_at)
        self.accounted_at = now

    def _reset_window(self):
        self.epoch += 1
        self.window_start = self.busy_seconds
        self.window_count = 0
        self.units = 0
        self.window_metric = None
        self.mixed_metric = False
        self.latencies = deque(maxlen=128)
        self.workloads = deque(maxlen=128)
        self.demand = False
        self.peak = 0

    def reconfigure(self, identity, ceiling):
        ceiling = max(1, min(MAX_CONCURRENCY, ceiling))
        if self.identity == identity and self.ceiling == ceiling:
            return
        # Preserve the same FIFO and in-flight counters when a runtime changes.
        self.identity, self.ceiling = identity, ceiling
        self.limit = self.best_limit = min(2, ceiling)
        self.best_score = self.baseline_latency = self.metric = None
        self.state, self.reason = "learning", "Runtime or GPU allocation changed; relearning"
        self.cooldown = 0
        self._reset_window()
        self.changed = True

    @contextlib.asynccontextmanager
    async def slot(self, record=None):
        ticket = object()
        admitted = False
        async with self.condition:
            self.queue.append(ticket)
            self.demand |= self.running >= self.limit
            try:
                await self.condition.wait_for(
                    lambda: self.queue[0] is ticket and self.running < self.limit
                )
                self.queue.popleft()
                self.running += 1
                self.peak = max(self.peak, self.running)
                self.demand |= bool(self.queue)
                admitted = True
                self.condition.notify_all()
            finally:
                if not admitted:
                    self.queue.remove(ticket)
                    self.condition.notify_all()
        admission = Admission(self, record, self.epoch)
        error = cancelled = False
        try:
            yield admission
        except asyncio.CancelledError:
            cancelled = True
            raise
        except BaseException:
            error = True
            raise
        finally:
            async with self.condition:
                now = self.clock()
                self._account(now)
                if admission.started is not None:
                    self.serving -= 1
                    if not cancelled:
                        self._complete(admission, now, error)
                self.running -= 1
                self.condition.notify_all()

    def _complete(self, admission, now, error):
        record = admission.record
        if record is None:
            return
        status = record.status or 0
        if 400 <= status < 500 and status != 429:
            return
        if error or status in {429, 503, 507} or status >= 500 or record.outcome == "failed":
            self.backoff("Runtime overload, timeout or failed response")
            return
        if not record.complete or not 200 <= status < 400 or admission.epoch != self.epoch:
            return
        elapsed = max(0.000001, now - admission.started)
        latency = (
            record.first_output - admission.perf_started
            if record.first_output and record.first_output >= admission.perf_started
            else elapsed
        )
        if record.output_tokens is not None and record.output_tokens > 0:
            unit, value = "output_tokens/s", record.output_tokens
        elif record.input_tokens is not None and record.input_tokens > 0:
            unit, value = "input_tokens/s", record.input_tokens
        else:
            unit, value = "requests/s", 1
        workload = (
            record.path,
            int(math.log2(max(1, record.input_tokens or 1))),
            int(math.log2(max(1, record.output_tokens or 1))),
        )
        self.observe(value, unit, latency, workload)

    def observe(self, units, metric, latency, workload):
        """Service-only samples: aggregate work divided by busy wall time, not sum of durations."""
        self.samples += 1
        self.window_count += 1
        self.units += units
        self.mixed_metric |= self.window_metric is not None and self.window_metric != metric
        self.window_metric = metric
        self.latencies.append(latency)
        self.workloads.append(workload)
        busy = self.busy_seconds - self.window_start
        if self.window_count < max(4, self.limit * 2) or (busy < 0.25 and self.window_count < 128):
            return
        if self.mixed_metric or busy <= 0:
            self.reason = "Waiting for comparable runtime measurements"
            self._reset_window()
            return
        latency = sorted(self.latencies)[math.ceil(len(self.latencies) * 0.95) - 1]
        workload = (
            max(set(self.workloads), key=self.workloads.count)[0],
            int(median(item[1] for item in self.workloads)),
            int(median(item[2] for item in self.workloads)),
        )
        metric_key = (metric, workload)
        score = self.units / busy
        if self.metric != metric_key:
            self.metric = metric_key
            self.best_score = self.baseline_latency = None
        if self.baseline_latency is not None and latency > max(
            self.baseline_latency * 2, self.baseline_latency + 0.1
        ):
            self.backoff("First-token or service latency regressed")
            return
        self.baseline_latency = min(self.baseline_latency or latency, latency)
        improving = self.best_score is None or score > self.best_score * 1.03
        efficient_lower = (
            self.limit < self.best_limit and score >= (self.best_score or score) * 0.97
        )
        if improving or efficient_lower:
            self.best_limit, self.best_score = self.limit, score
        elif self.limit != self.best_limit:
            self.limit = self.best_limit
            self.cooldown = 32
            self.state, self.reason = "stable", "More parallelism did not improve useful throughput"
        else:
            self.best_score = score * 0.2 + self.best_score * 0.8
        self.cooldown = max(0, self.cooldown - self.window_count)
        if not self.demand:
            self.state, self.reason = (
                "awaiting-demand",
                "More simultaneous traffic is needed to test a higher limit",
            )
        elif not self.cooldown and self._pressure_allows_probe():
            if self.probe_down and self.best_limit > 1:
                candidate = self.best_limit - 1
            else:
                candidate = min(
                    self.ceiling, self.best_limit + max(1, math.ceil(self.best_limit / 4))
                )
            self.probe_down = not self.probe_down
            self.limit = candidate
            self.state = "learning" if candidate != self.best_limit else "stable"
            self.reason = (
                "Testing concurrency against observed aggregate throughput"
                if candidate != self.best_limit
                else "Runtime concurrency ceiling reached"
            )
        self.changed = True
        self._reset_window()

    def _pressure_allows_probe(self):
        return self.pressure is None or self.clock() - self.pressure_at > 15 or self.pressure < 0.9

    def backoff(self, reason):
        now = self.clock()
        if now - self.last_backoff < 2:
            return
        self.last_backoff = now
        self.limit = max(1, math.floor(self.limit / 2))
        self.best_limit = self.limit
        self.best_score = self.baseline_latency = None
        self.cooldown = 32
        self.state, self.reason = "backoff", reason
        self.changed = True
        self._reset_window()

    def telemetry(self, sample):
        ratio = sample.get("gpu_kv_cache_usage")
        if not isinstance(ratio, (int, float)) or not math.isfinite(ratio):
            return
        self.pressure, self.pressure_at = ratio, self.clock()
        self.hot_samples = self.hot_samples + 1 if ratio >= 0.92 and self.running else 0
        if self.hot_samples >= 2 and (sample.get("queued_requests") or self.running >= self.limit):
            self.backoff("Sustained KV-cache pressure")

    def public(self):
        return {
            "mode": "automatic",
            "limit": self.limit,
            "running": self.running,
            "waiting": len(self.queue),
            "runtime_ceiling": self.ceiling,
            "state": self.state,
            "reason": self.reason,
            "samples": self.samples,
            "best_limit": self.best_limit,
            "throughput": self.best_score,
            "throughput_unit": self.metric[0] if self.metric else None,
        }


class AdaptiveConcurrency:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.controllers = {}
        self.learned = {}
        self.last_flush = time.monotonic()
        self.flush_lock = asyncio.Lock()
        if path:
            try:
                self.learned = dict(list(json.loads(path.read_text()).items())[-128:])
            except (OSError, ValueError, TypeError, AttributeError):
                pass

    def controller(self, runtime_id, identity, ceiling=MAX_CONCURRENCY):
        controller = self.controllers.get(runtime_id)
        if controller is None:
            if len(self.controllers) >= 128:
                idle = next(
                    (
                        key
                        for key, value in self.controllers.items()
                        if not value.running and not value.queue
                    ),
                    None,
                )
                if idle is not None:
                    previous = self.controllers.pop(idle)
                    if previous.changed:
                        self.learned[previous.identity] = {
                            "limit": previous.best_limit,
                            "updated_at": time.time(),
                        }
                        self.learned = dict(list(self.learned.items())[-128:])
                else:
                    raise RuntimeError("Adaptive runtime capacity is full")
            controller = ConcurrencyController(identity, ceiling, self.learned.get(identity))
            self.controllers[runtime_id] = controller
        else:
            controller.reconfigure(identity, ceiling)
        return controller

    def telemetry(self, samples):
        for runtime_id, sample in samples.items():
            if controller := self.controllers.get(runtime_id):
                controller.telemetry(sample)

    async def flush(self, force=False):
        if not self.path or (not force and time.monotonic() - self.last_flush < 10):
            return
        async with self.flush_lock:
            for controller in self.controllers.values():
                if controller.changed:
                    self.learned.pop(controller.identity, None)
                    self.learned[controller.identity] = {
                        "limit": controller.best_limit,
                        "updated_at": time.time(),
                    }
                    controller.changed = False
            self.learned = dict(list(self.learned.items())[-128:])
            payload = json.dumps(self.learned)

            def write():
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                temporary = self.path.with_suffix(".tmp")
                temporary.touch(mode=0o600, exist_ok=True)
                temporary.chmod(0o600)
                temporary.write_text(payload)
                temporary.replace(self.path)

            try:
                await asyncio.to_thread(write)
                self.last_flush = time.monotonic()
            except OSError:
                for controller in self.controllers.values():
                    controller.changed = True
