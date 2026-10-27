import { createContext, useContext, useState, type ReactNode } from "react";
import { useApi } from "@/lib/api";
import type { Performance } from "@/lib/types";
import { number } from "@/lib/utils";
import { Sheet, Button, Select, Field, Facts, Message, Chart } from "./common";
const Context = createContext<(scope: string, entity: string) => void>(
  () => {},
);
export function PerformanceButton({
  scope = "overall",
  entity = "all",
  children = "Performance ↗",
}: {
  scope?: string;
  entity?: string;
  children?: ReactNode;
}) {
  const open = useContext(Context);
  return (
    <Button
      size="sm"
      variant="ghost"
      onClick={(e) => {
        e.stopPropagation();
        open(scope, entity);
      }}
    >
      {children}
    </Button>
  );
}
export function PerformanceProvider({ children }: { children: ReactNode }) {
  const [selection, setSelection] = useState<{
    scope: string;
    entity: string;
  } | null>(null);
  const [hours, setHours] = useState("24");
  const query = useApi<Performance>(
    selection
      ? `/api/v1/inference/performance?${new URLSearchParams({ ...selection, hours })}`
      : null,
    10000,
  );
  const data = query.data;
  const s = data?.summary;
  return (
    <Context.Provider
      value={(scope, entity) => setSelection({ scope, entity })}
    >
      {children}
      <Sheet
        open={!!selection}
        onOpenChange={(open) => {
          if (!open) setSelection(null);
        }}
        title="Performance history"
        description={
          selection?.scope === "overall"
            ? "All tracked workloads"
            : selection?.entity
        }
      >
        <div className="toolbar">
          <Field label="Scope">
            <Select
              value={selection?.scope}
              onChange={(e) =>
                setSelection({
                  scope: e.target.value,
                  entity:
                    data?.entities.find((x) => x.scope === e.target.value)
                      ?.entity || "all",
                })
              }
            >
              {["overall", "model", "pool", "gpu"].map((x) => (
                <option key={x}>{x}</option>
              ))}
            </Select>
          </Field>
          {selection?.scope !== "overall" && (
            <Field label="Entity">
              <Select
                value={selection?.entity}
                onChange={(e) =>
                  setSelection((v) => v && { ...v, entity: e.target.value })
                }
              >
                {[
                  ...new Set([
                    selection?.entity,
                    ...(data?.entities
                      .filter((x) => x.scope === selection?.scope)
                      .map((x) => x.entity) || []),
                  ]),
                ].map((x) => (
                  <option key={x}>{x}</option>
                ))}
              </Select>
            </Field>
          )}
          <Field label="Range">
            <Select value={hours} onChange={(e) => setHours(e.target.value)}>
              {[1, 6, 24, 168, 720, 2160, 8760].map((x) => (
                <option key={x} value={x}>
                  {x < 24 ? `${x} hours` : `${x / 24} days`}
                </option>
              ))}
            </Select>
          </Field>
          <Button variant="outline" onClick={() => query.refetch()}>
            Refresh
          </Button>
        </div>
        <Message error={query.error} />
        {query.isLoading && <p role="status">Loading recorded performance…</p>}
        {data && s && (
          <>
            <Facts
              items={[
                ["Decode speed", number(s.decode_tokens_per_second, " tok/s")],
                [
                  "Effective prefill",
                  number(s.effective_prefill_tokens_per_second, " tok/s"),
                ],
                [
                  "Cached input",
                  number(
                    s.cached_input_ratio == null
                      ? null
                      : s.cached_input_ratio * 100,
                    "%",
                  ),
                ],
                ["Time to first token", number(s.ttft_ms_mean, " ms")],
                ["Requests", number(s.requests)],
                [
                  "Input / output",
                  `${number(s.input_tokens)} / ${number(s.output_tokens)}`,
                ],
              ]}
            />
            <div className="grid-two">
              {[
                ["decode_tokens_per_second", "Decode speed · tok/s"],
                [
                  "effective_prefill_tokens_per_second",
                  "Effective prefill · tok/s",
                ],
                ["cached_input_ratio", "Cached input ratio"],
                ["runtime_gpu_kv_cache_usage", "Runtime KV occupancy"],
              ].map(([key, label]) => (
                <Chart
                  key={key}
                  label={label}
                  values={data.history.map((p, i) => {
                    const previous = data.history[i - 1];
                    return previous &&
                      p.timestamp >
                        previous.timestamp + previous.resolution_seconds * 1.5
                      ? null
                      : (p[key] ?? null);
                  })}
                />
              ))}
            </div>
            <details>
              <summary>Runtime & latency</summary>
              <Facts
                items={[
                  "runtime_decode_tokens_per_second",
                  "runtime_prefill_tokens_per_second",
                  "runtime_gpu_kv_cache_usage",
                  "runtime_gpu_kv_cache_usage_max",
                  "runtime_cache_retrieval_p95_ms",
                  "duration_ms_mean",
                  "queue_ms_mean",
                ].map((k) => [k.replaceAll("_", " "), number(s[k])])}
              />
            </details>
            <p className="note">{data.coverage}</p>
            {["model", "pool", "gpu"].map((scope) => (
              <div key={scope} className="actions">
                <b>{scope}</b>
                {data.entities
                  .filter((x) => x.scope === scope)
                  .slice(0, 12)
                  .map((x) => (
                    <PerformanceButton
                      key={x.entity}
                      scope={scope}
                      entity={x.entity}
                    >
                      {x.entity}
                    </PerformanceButton>
                  ))}
              </div>
            ))}
            <p className="note">
              Minute detail {data.retention.minute_days}d → hourly{" "}
              {data.retention.hour_days}d → daily {data.retention.day_days}d ·
              storage cap {number(data.retention.max_rows)} buckets.{" "}
              {data.persistence.healthy
                ? "Journal healthy"
                : "Journal write failing"}{" "}
              · {data.persistence.pending} pending · {data.persistence.dropped}{" "}
              dropped.
            </p>
          </>
        )}
      </Sheet>
    </Context.Provider>
  );
}
