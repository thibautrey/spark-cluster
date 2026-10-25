/* One bounded read every two seconds, only while the overview is visible. */
(() => {
  let activity = null;
  let pending = false;
  const number = value => value == null ? '—' : formatNumber(value);
  const milliseconds = value => value == null ? '—' : `${Math.round(value).toLocaleString()} ms`;
  const cell = value => `<td>${escapeHtml(String(value))}</td>`;

  function render() {
    if (!activity) return;
    const totals = activity.totals;
    const metrics = activity.runtime_metrics;
    const metricsFresh = metrics?.available && Date.now() - Date.parse(metrics.sampled_at) < 15000;
    const pools = activity.pools.filter(pool => pool.available && pool.age_seconds < 15);
    const running = activity.active_requests;
    const queued = activity.queued_requests;
    const loaded = pools.filter(pool => pool.active_model).length;
    $('overview-model').textContent = running
      ? `${running} request${running === 1 ? '' : 's'} in progress · ${queued} waiting for a pool`
      : totals.requests ? `Idle · ${number(totals.requests)} requests recorded since ${new Date(activity.since * 1000).toLocaleString()}`
        : `No requests recorded since ${new Date(activity.since * 1000).toLocaleString()}. Earlier activity was not captured.`;
    const stats = [
      ['Requests / errors', `${number(totals.requests)} / ${number(totals.failed)}`],
      ['Active / waiting', `${running - queued} / ${queued}`],
      ['Input / output tokens', `${number(totals.input_tokens)} / ${number(totals.output_tokens)}`],
      ['Latency p50 / p95', `${milliseconds(activity.latency.p50_ms)} / ${milliseconds(activity.latency.p95_ms)}`],
      ['First token p50 / p95', `${milliseconds(activity.latency.ttft_p50_ms)} / ${milliseconds(activity.latency.ttft_p95_ms)}`],
      ['Pools busy / loaded / available', `${pools.filter(pool => pool.busy).length} / ${loaded} / ${pools.length}`],
    ];
    $('overview-runtime-stats').innerHTML = stats.map(([label, value]) => `<div><dt>${label}</dt><dd>${escapeHtml(value)}</dd></div>`).join('');
    $('inference-coverage').textContent = `${totals.successful} successful · ${totals.cancelled} cancelled · ${number(totals.cached_tokens)} cached input tokens · ${totals.unknown_usage} requests without token usage. Latency covers the latest ${activity.latency.sample_size} recorded requests.`;
    $('output-metric-label').textContent = metricsFresh ? 'Runtime output' : 'Tracked requests';
    $('output-tps').textContent = metricsFresh && metrics.decode_tokens_per_second != null
      ? metrics.decode_tokens_per_second.toFixed(1) : number(totals.requests);
    $('output-unit').textContent = metricsFresh && metrics.decode_tokens_per_second != null ? ' tok/s' : '';
    // Persisted minute buckets keep earlier activity visible after a model unload or page reload.
    const buckets = new Map(activity.history.map(point => [point.timestamp, point.output_tokens / 60]));
    const minute = Math.floor(Date.now() / 60000) * 60;
    drawSparkline(Array.from({length: 40}, (_, index) => buckets.get(minute - (39 - index) * 60) || 0));
    $('throughput-spark').title = 'Reported output tokens per second, averaged over each minute';
    const poolNames = new Map(activity.pools.map(pool => [pool.id, pool.name]));
    const active = activity.active.map(request => ({...request, outcome: request.state, duration_ms: request.elapsed_ms}));
    const rows = [...active, ...activity.recent].slice(0, 12);
    $('inference-recent').innerHTML = rows.length ? rows.map(request => {
      const memory = request.memory_end || request.memory_start;
      const pool = poolNames.get(request.cluster_id) || request.cluster_id || 'Unassigned';
      const memoryText = memory ? `${memory.used_gib.toFixed(1)} / ${memory.total_gib.toFixed(1)} GiB` : 'Unavailable';
      return `<tr>${cell(new Date(request.started_at * 1000).toLocaleTimeString())}${cell(`${request.model_id} · ${pool}`)}${cell(request.outcome)}${cell(milliseconds(request.duration_ms))}${cell(`${number(request.input_tokens)} / ${number(request.output_tokens)}`)}${cell(memoryText)}</tr>`;
    }).join('') : '<tr><td colspan="6">Tracking is ready. Requests appear here as clients use the pool endpoints.</td></tr>';
    $('inference-pools').innerHTML = activity.pools.map(pool => {
      const machines = pool.machines;
      const used = machines.reduce((sum, machine) => sum + (machine.memory_used_gib || 0), 0);
      const total = machines.reduce((sum, machine) => sum + (machine.memory_total_gib || 0), 0);
      return `<p>${escapeHtml(pool.name)} · ${escapeHtml(pool.active_model || 'No model loaded')} · ${pool.age_seconds < 15 && pool.available && machines.length && machines.every(machine => machine.available !== false) ? `${used.toFixed(1)} / ${total.toFixed(1)} GiB` : 'Resource sample unavailable'} · ${pool.busy ? 'busy' : 'idle'}</p>`;
    }).join('') || '<p>Pool resource samples will appear when a compute pool reports.</p>';
    $('inference-groups').innerHTML = activity.groups.map(group => `<p>${escapeHtml(group.model_id)} · ${escapeHtml(poolNames.get(group.cluster_id) || group.cluster_id || 'Unassigned')} · ${number(group.requests)} requests · ${number(group.input_tokens)} input / ${number(group.output_tokens)} output tokens</p>`).join('');
    const persistence = activity.persistence;
    $('inference-tracking-note').textContent = `${activity.coverage} Memory is a shared pool sample, not per-request allocation. Records retain no prompts, responses or credentials. ${persistence.healthy ? 'Journal healthy' : 'Journal write failing; retry pending'} · ${persistence.pending} pending · ${persistence.dropped} dropped.`;
  }

  window.renderTrackedInference = render;
  async function refresh() {
    if (pending || document.hidden || (location.hash && location.hash !== '#overview')) return;
    pending = true;
    try {
      activity = await api('/api/v1/inference/activity?limit=12');
      render();
    } catch {
      $('overview-model').textContent = 'Inference tracking unavailable. Activity cannot be determined.';
    } finally {
      pending = false;
    }
  }
  refresh();
  setInterval(refresh, 2000);
  window.addEventListener('hashchange', refresh);
  document.addEventListener('visibilitychange', refresh);
})();
