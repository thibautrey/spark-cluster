/* Contextual, read-only performance sheets. No polling while closed or hidden. */
(() => {
  const sheet = document.getElementById('performance-sheet');
  const panel = sheet.querySelector('[role="dialog"]');
  const body = document.getElementById('performance-body');
  const scopeField = document.getElementById('performance-scope');
  const entityField = document.getElementById('performance-entity');
  const rangeField = document.getElementById('performance-range');
  const closeButton = document.getElementById('performance-close');
  let current = {scope: 'overall', entity: 'all'};
  let opener = null;
  let serial = 0;
  let pending = false;
  let latest = null;
  let abort = null;
  const number = (value, unit = '', digits = 1) => value == null ? '—'
    : `${Number(value).toLocaleString(undefined, {maximumFractionDigits: digits})}${unit}`;
  const label = (scope, entity) => {
    if (scope === 'overall') return 'All tracked workloads';
    if (scope === 'gpu') {
      try { const [pool, gpu] = JSON.parse(entity); return `${pool} · GPU ${gpu}`; } catch { return entity; }
    }
    return entity;
  };
  const attr = value => escapeHtml(String(value));
  window.performanceButton = (scope, entity, text = 'Performance ↗') =>
    `<button type="button" class="performance-trigger" data-performance-scope="${attr(scope)}" data-performance-entity="${attr(entity)}" aria-label="View performance for ${attr(label(scope, entity))}">${escapeHtml(text)}</button>`;

  function cards(summary) {
    const cache = summary.cached_input_ratio;
    const entries = [
      ['Decode speed', number(summary.decode_tokens_per_second, ' tok/s'), 'Weighted streaming requests'],
      ['Effective prefill', number(summary.effective_prefill_tokens_per_second, ' tok/s'), 'Input / backend time to first output'],
      ['Cached input', cache == null ? '—' : number(cache * 100, '%'), 'Only requests reporting cached tokens'],
      ['Time to first token', number(summary.ttft_ms_mean, ' ms'), 'Mean successful streaming request'],
      ['Requests', number(summary.requests, '', 0), `${number(summary.failed, '', 0)} failed · ${number(summary.cancelled, '', 0)} cancelled`],
      ['Input / output', `${number(summary.input_tokens, '', 0)} / ${number(summary.output_tokens, '', 0)}`, 'Provider-reported tokens'],
    ];
    return entries.map(([title, value, note]) => `<article><span>${title}</span><strong>${value}</strong><small>${escapeHtml(note)}</small></article>`).join('');
  }

  function chart(history, key, title, unit, scale = 1) {
    const points = history.map(point => ({...point, value: point[key] == null ? null : point[key] * scale}));
    const known = points.filter(point => point.value != null);
    if (!known.length) return `<figure class="performance-chart"><figcaption>${title}<small>${unit}</small></figcaption><p class="performance-empty">No reported measurements in this range</p></figure>`;
    const maximum = Math.max(...known.map(point => point.value), 1);
    const first = points[0].timestamp;
    const end = Math.max(...points.map(point => point.timestamp + point.resolution_seconds), first + 60);
    const x = point => 38 + (point.timestamp - first) / (end - first) * 622;
    const y = point => 140 - point.value / maximum * 112;
    let path = '';
    let previous = null;
    for (const point of points) {
      if (point.value == null) { previous = null; continue; }
      const gap = !previous || point.timestamp > previous.timestamp + previous.resolution_seconds * 1.5;
      path += `${gap ? 'M' : 'L'}${x(point).toFixed(2)},${y(point).toFixed(2)} `;
      previous = point;
    }
    const circles = known.map(point => `<circle cx="${x(point)}" cy="${y(point)}" r="2.5"><title>${escapeHtml(new Date(point.timestamp * 1000).toLocaleString())} · ${number(point.value, ` ${unit}`)}</title></circle>`).join('');
    return `<figure class="performance-chart"><figcaption>${title}<small>${unit}</small></figcaption><svg viewBox="0 0 680 172" role="img" aria-label="${title} over time"><text x="4" y="25">${number(maximum)}</text><text x="15" y="144">0</text><path class="performance-axis" d="M38 28V140H660"/><path class="performance-line" d="${path}"/>${circles}</svg><div class="performance-chart-times"><span>${escapeHtml(new Date(first * 1000).toLocaleString())}</span><span>${escapeHtml(new Date(end * 1000).toLocaleString())}</span></div></figure>`;
  }

  function render(data) {
    latest = data;
    document.getElementById('performance-title').textContent = label(current.scope, current.entity);
    const options = data.entities.filter(item => item.scope === current.scope);
    if (!options.some(item => item.entity === current.entity)) options.unshift({...current});
    entityField.innerHTML = options.map(item => `<option value="${attr(item.entity)}">${escapeHtml(label(item.scope, item.entity))}</option>`).join('');
    entityField.value = current.entity;
    entityField.hidden = current.scope === 'overall';
    entityField.parentElement.hidden = entityField.hidden;
    const s = data.summary;
    const runtime = [
      ['Runtime decode', number(s.runtime_decode_tokens_per_second, ' tok/s')],
      ['Runtime prefill', number(s.runtime_prefill_tokens_per_second, ' tok/s')],
      ['KV cache occupancy', number(s.runtime_gpu_kv_cache_usage == null ? null : s.runtime_gpu_kv_cache_usage * 100, '%')],
      ['Peak KV occupancy', number(s.runtime_gpu_kv_cache_usage_max == null ? null : s.runtime_gpu_kv_cache_usage_max * 100, '%')],
      ['Cache retrieval p95', number(s.runtime_cache_retrieval_p95_ms, ' ms')],
      ['Mean request / queue', `${number(s.duration_ms_mean, ' ms')} / ${number(s.queue_ms_mean, ' ms')}`],
    ];
    const browse = ['model', 'pool', 'gpu'].map(kind => {
      const entities = data.entities.filter(item => item.scope === kind).slice(0, 12);
      return entities.length ? `<div class="performance-browse"><h4>${kind === 'gpu' ? 'GPUs' : kind === 'pool' ? 'Pools' : 'Models'}</h4>${entities.map(item => window.performanceButton(kind, item.entity, label(kind, item.entity))).join('')}</div>` : '';
    }).join('');
    const retention = data.retention;
    body.innerHTML = `<div class="performance-cards">${cards(s)}</div>
      ${!s.requests && !s.runtime_samples ? '<p class="performance-empty">No recorded activity for this selection. Measurements will appear after routed requests or runtime telemetry are recorded.</p>' : ''}
      <div class="performance-charts">${chart(data.history, 'decode_tokens_per_second', 'Decode speed', 'tok/s')}${chart(data.history, 'effective_prefill_tokens_per_second', 'Effective prefill', 'tok/s')}${chart(data.history, 'cached_input_ratio', 'Cached input', '%', 100)}${chart(data.history, 'runtime_gpu_kv_cache_usage', 'Runtime KV occupancy', '%', 100)}</div>
      <section class="performance-runtime"><h4>Runtime &amp; latency</h4><p>Runtime rates and cache gauges are sample means over this range, separate from routed request accounting.</p><dl>${runtime.map(([title, value]) => `<div><dt>${title}</dt><dd>${value}</dd></div>`).join('')}</dl></section>
      <p class="performance-note">Token usage reported on ${number(s.input_tokens_known, '', 0)} / ${number(s.requests, '', 0)} requests; cached tokens on ${number(s.cached_tokens_known, '', 0)}. ${escapeHtml(data.coverage)}</p>
      ${browse}
      <footer class="performance-note">SQLite history · minute detail ${retention.minute_days}d → hourly ${retention.hour_days}d → daily ${retention.day_days}d, then expires. A ${number(retention.max_rows, '', 0)}-bucket storage cap may shorten history. ${data.persistence.healthy ? 'Journal healthy' : 'Journal write failing'} · ${data.persistence.pending} pending · ${data.persistence.dropped} dropped.</footer>`;
    document.getElementById('performance-status').textContent = `Updated ${new Date(data.end * 1000).toLocaleTimeString()} · ${data.history.length} time buckets`;
  }

  async function refresh({loading = false} = {}) {
    if (sheet.hidden || document.hidden || (pending && !loading)) return;
    const request = ++serial;
    abort?.abort();
    abort = new AbortController();
    pending = true;
    if (loading) body.innerHTML = '<p class="performance-empty" role="status">Loading recorded performance…</p>';
    try {
      const query = new URLSearchParams({...current, hours: rangeField.value});
      const response = await fetch(`/api/v1/inference/performance?${query}`, {signal: abort.signal});
      if (!response.ok) throw new Error(`Performance history unavailable (${response.status})`);
      const data = await response.json();
      if (request === serial && !sheet.hidden) render(data);
    } catch (error) {
      if (request !== serial || error.name === 'AbortError') return;
      document.getElementById('performance-status').textContent = error.message;
      if (loading) body.innerHTML = '<p class="performance-empty">History could not be loaded. Use Refresh to retry.</p>';
    } finally { if (request === serial) pending = false; }
  }

  function open(scope = 'overall', entity = 'all', trigger = null) {
    if (sheet.hidden) opener = trigger || document.activeElement;
    current = {scope, entity: scope === 'overall' ? 'all' : entity};
    scopeField.value = scope;
    entityField.hidden = scope === 'overall';
    entityField.parentElement.hidden = entityField.hidden;
    entityField.innerHTML = `<option value="${attr(current.entity)}">${escapeHtml(label(scope, current.entity))}</option>`;
    document.getElementById('performance-title').textContent = label(scope, current.entity);
    sheet.hidden = false;
    document.body.classList.add('bottom-sheet-open');
    requestAnimationFrame(() => { sheet.classList.add('open'); closeButton.focus({preventScroll: true}); });
    void refresh({loading: true});
  }
  function close() {
    ++serial;
    abort?.abort();
    pending = false;
    sheet.classList.remove('open');
    sheet.hidden = true;
    if (!document.querySelector('.bottom-sheet.open')) document.body.classList.remove('bottom-sheet-open');
    if (opener?.isConnected) opener.focus({preventScroll: true});
  }
  document.addEventListener('click', event => {
    const trigger = event.target.closest('[data-performance-scope]');
    if (!trigger) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    open(trigger.dataset.performanceScope, trigger.dataset.performanceEntity || 'all', trigger);
  }, true);
  closeButton.addEventListener('click', close);
  document.getElementById('performance-backdrop').addEventListener('click', close);
  document.getElementById('performance-refresh').addEventListener('click', () => refresh({loading: true}));
  rangeField.addEventListener('change', () => refresh({loading: true}));
  scopeField.addEventListener('change', () => {
    const scope = scopeField.value;
    const entity = latest?.entities.find(item => item.scope === scope)?.entity;
    open(scope, entity || (scope === 'overall' ? 'all' : 'unrecorded'));
  });
  entityField.addEventListener('change', () => { current.entity = entityField.value; void refresh({loading: true}); });
  // Capture Escape so a parent sheet stays open when this nested sheet closes.
  document.addEventListener('keydown', event => {
    if (sheet.hidden) return;
    if (event.key === 'Escape') { event.preventDefault(); event.stopImmediatePropagation(); close(); }
    if (event.key === 'Tab') {
      const focusable = [...panel.querySelectorAll('button, select, a[href]')].filter(item => !item.hidden && !item.disabled);
      const first = focusable[0], last = focusable.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  }, true);
  window.openPerformanceSheet = open;
  setInterval(() => refresh(), 10000);
  document.addEventListener('visibilitychange', () => refresh());
})();
