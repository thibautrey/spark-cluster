/* One bounded read every two seconds, only while the overview is visible. */
(() => {
  let activity = null;
  let pending = false;
  let retainedModels = new Set();
  let instanceCounts = new Map();
  let menuModel = null;
  let menuTrigger = null;
  const menu = $('inference-row-menu');
  const keepMemory = $('inference-keep-memory');
  const instanceStepper = $('inference-instance-stepper');
  const instanceMinus = $('inference-instance-minus');
  const instanceCount = $('inference-instance-count');
  const instancePlus = $('inference-instance-plus');
  const poolAction = $('inference-pool-action');
  const poolActionIcon = $('inference-pool-action-icon');
  const poolActionLabel = $('inference-pool-action-label');
  const poolActionDescription = $('inference-pool-action-description');
  const menuError = $('inference-menu-error');
  const number = value => value == null ? '—' : formatNumber(value);
  const milliseconds = value => value == null ? '—' : `${Math.round(value).toLocaleString()} ms`;
  const cell = value => `<td>${escapeHtml(String(value))}</td>`;
  const actionCell = modelId => `<td class="inference-action-cell"><button type="button" class="inference-menu-trigger" data-inference-model="${escapeHtml(modelId)}" aria-label="Actions for ${escapeHtml(modelId)}" aria-haspopup="menu" aria-expanded="false">⋮</button></td>`;

  function closeMenu({restoreFocus = false} = {}) {
    if (menu.hidden) return;
    menu.hidden = true;
    menuTrigger?.setAttribute('aria-expanded', 'false');
    if (restoreFocus && menuTrigger?.isConnected) menuTrigger.focus();
    menuModel = null;
    menuTrigger = null;
  }

  function openMenu(trigger) {
    closeMenu();
    menuModel = trigger.dataset.inferenceModel;
    menuTrigger = trigger;
    menuError.hidden = true;
    menuError.textContent = '';
    keepMemory.setAttribute('aria-checked', String(retainedModels.has(menuModel)));
    const instances = instanceCounts.get(menuModel) || 0;
    instanceStepper.hidden = instances < 1;
    instanceCount.textContent = `${instances} instance${instances === 1 ? '' : 's'}`;
    instanceMinus.disabled = instances <= 1;
    instanceMinus.setAttribute('aria-label', `Remove one ${menuModel} instance`);
    instancePlus.setAttribute('aria-label', `Add one ${menuModel} instance`);
    const action = instances ? 'unload' : 'load';
    const label = action === 'load' ? 'Load' : 'Unload';
    poolAction.dataset.action = action;
    poolAction.setAttribute('aria-label', `${label} ${menuModel}`);
    poolActionIcon.textContent = action === 'load' ? '↑' : '↓';
    poolActionLabel.textContent = label;
    poolActionDescription.textContent = action === 'load'
      ? 'Load this model into an available pool.'
      : 'Unload this model and release its pool memory.';
    trigger.setAttribute('aria-expanded', 'true');
    menu.hidden = false;
    const triggerBox = trigger.getBoundingClientRect();
    const menuBox = menu.getBoundingClientRect();
    const margin = 8;
    const left = Math.min(window.innerWidth - menuBox.width - margin, Math.max(margin, triggerBox.right - menuBox.width));
    const below = triggerBox.bottom + 4;
    const top = below + menuBox.height <= window.innerHeight - margin
      ? below : Math.max(margin, triggerBox.top - menuBox.height - 4);
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;
    keepMemory.focus();
  }

  function render() {
    if (!activity) return;
    const totals = activity.totals;
    const metrics = activity.runtime_metrics;
    const metricsFresh = metrics?.available && Date.now() - Date.parse(metrics.sampled_at) < 15000;
    const pools = activity.pools.filter(pool => pool.available && pool.age_seconds < 15);
    const running = activity.active_requests;
    const queued = activity.queued_requests;
    const loaded = pools.filter(pool => pool.resident_models?.length || pool.active_model).length;
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
    retainedModels = new Set(activity.retained_models || []);
    instanceCounts = new Map();
    activity.pools.forEach(pool => {
      const residents = pool.resident_models || (pool.active_model ? [{model_id: pool.active_model}] : []);
      residents.forEach(resident => instanceCounts.set(resident.model_id, (instanceCounts.get(resident.model_id) || 0) + 1));
    });
    const active = activity.active.map(request => ({...request, outcome: request.state, duration_ms: request.elapsed_ms}));
    const rows = [...active, ...activity.recent].slice(0, 12);
    const recentMarkup = rows.length ? rows.map(request => {
      const memory = request.memory_end || request.memory_start;
      const pool = poolNames.get(request.cluster_id) || request.cluster_id || 'Unassigned';
      const memoryText = memory ? `${memory.used_gib.toFixed(1)} / ${memory.total_gib.toFixed(1)} GiB` : 'Unavailable';
      return `<tr class="inference-model-row" data-model-repo="${escapeHtml(request.model_id)}" tabindex="0" aria-label="Open details for ${escapeHtml(request.model_id)}">${cell(new Date(request.started_at * 1000).toLocaleTimeString())}<td>${escapeHtml(request.model_id)} · ${escapeHtml(pool)} ${window.performanceButton?.("model", request.model_id, "Speed & cache ↗") || ""}</td>${cell(request.outcome)}${cell(milliseconds(request.duration_ms))}${cell(`${number(request.input_tokens)} / ${number(request.output_tokens)}`)}${cell(memoryText)}${actionCell(request.model_id)}</tr>`;
    }).join('') : '<tr><td colspan="7">Tracking is ready. Requests appear here as clients use the pool endpoints.</td></tr>';
    if (menu.hidden) $('inference-recent').innerHTML = recentMarkup;
    $('inference-pools').innerHTML = activity.pools.map(pool => {
      const machines = pool.machines;
      const used = machines.reduce((sum, machine) => sum + (machine.memory_used_gib || 0), 0);
      const total = machines.reduce((sum, machine) => sum + (machine.memory_total_gib || 0), 0);
      return `<p>${window.performanceButton?.("pool", pool.id, pool.name) || escapeHtml(pool.name)} · ${escapeHtml(pool.resident_models?.map(r => r.model_id).join(', ') || pool.active_model || 'No model loaded')} · ${pool.age_seconds < 15 && pool.available && machines.length && machines.every(machine => machine.available !== false) ? `${used.toFixed(1)} / ${total.toFixed(1)} GiB` : 'Resource sample unavailable'} · ${pool.busy ? 'busy' : 'idle'}</p>`;
    }).join('') || '<p>Pool resource samples will appear when a compute pool reports.</p>';
    $('inference-groups').innerHTML = activity.groups.map(group => `<p>${window.performanceButton?.("model", group.model_id, group.model_id) || escapeHtml(group.model_id)} · ${escapeHtml(poolNames.get(group.cluster_id) || group.cluster_id || 'Unassigned')} · ${number(group.requests)} requests · ${number(group.input_tokens)} input / ${number(group.output_tokens)} output tokens</p>`).join('');
    const persistence = activity.persistence;
    $('inference-tracking-note').textContent = `${activity.coverage} Memory is a shared pool sample, not per-request allocation. Records retain no prompts, responses or credentials. ${persistence.healthy ? 'Journal healthy' : 'Journal write failing; retry pending'} · ${persistence.pending} pending · ${persistence.dropped} dropped.`;
  }

  $('inference-recent').addEventListener('click', event => {
    const trigger = event.target.closest('.inference-menu-trigger');
    if (trigger) {
      if (trigger === menuTrigger && !menu.hidden) closeMenu({restoreFocus: true});
      else openMenu(trigger);
      return;
    }
    const model = event.target.closest('[data-model-repo]');
    if (model) window.inspectCatalogModel?.(model.dataset.modelRepo, 'main', model);
  });
  $('inference-recent').addEventListener('keydown', event => {
    if (!['Enter', ' '].includes(event.key)) return;
    const model = event.target.closest('[data-model-repo]');
    if (!model || event.target !== model) return;
    event.preventDefault();
    window.inspectCatalogModel?.(model.dataset.modelRepo, 'main', model);
  });
  keepMemory.addEventListener('click', async () => {
    if (!menuModel) return;
    const modelId = menuModel;
    const shouldRetain = !retainedModels.has(modelId);
    keepMemory.disabled = true;
    menuError.hidden = true;
    try {
      await api(`/api/v1/pool/models/${encodeURIComponent(modelId)}/retention`, {
        method: 'PATCH',
        headers: window.adminAuthHeaders(),
        body: JSON.stringify({keep_loaded: shouldRetain}),
      });
      if (shouldRetain) retainedModels.add(modelId);
      else retainedModels.delete(modelId);
      keepMemory.setAttribute('aria-checked', String(shouldRetain));
      $('inference-action-status').textContent = shouldRetain
        ? `${modelId} will be kept in memory.`
        : `${modelId} may now be unloaded when the pool needs memory.`;
      closeMenu({restoreFocus: true});
      void refresh();
    } catch (error) {
      menuError.textContent = error.message;
      menuError.hidden = false;
    } finally {
      keepMemory.disabled = false;
    }
  });
  poolAction.addEventListener('click', async () => {
    if (!menuModel) return;
    const modelId = menuModel;
    const action = poolAction.dataset.action;
    keepMemory.disabled = true;
    poolAction.disabled = true;
    menuError.hidden = true;
    try {
      await api(`/api/v1/pool/${action}`, {
        method: 'POST',
        headers: window.adminAuthHeaders(),
        body: JSON.stringify({
          model_id: modelId,
          cluster_id: window.selectedClusterContext?.() === 'all'
            ? null : window.selectedClusterContext?.(),
        }),
      });
      if (action === 'unload') {
        instanceCounts.delete(modelId);
        retainedModels.delete(modelId);
      }
      $('inference-action-status').textContent = action === 'load'
        ? `Load requested for ${modelId}.`
        : `${modelId} was unloaded and its pool memory was released.`;
      closeMenu({restoreFocus: true});
      void refresh();
    } catch (error) {
      menuError.textContent = error.message;
      menuError.hidden = false;
    } finally {
      keepMemory.disabled = false;
      poolAction.disabled = false;
    }
  });
  async function scaleInstances(delta) {
    if (!menuModel) return;
    const modelId = menuModel;
    [keepMemory, instanceMinus, instancePlus, poolAction].forEach(item => { item.disabled = true; });
    menuError.hidden = true;
    try {
      const result = await api('/api/v1/pool/scale', {
        method: 'POST',
        headers: window.adminAuthHeaders(),
        body: JSON.stringify({model_id: modelId, delta}),
      });
      if (delta < 0) instanceCounts.set(modelId, result.instances);
      $('inference-action-status').textContent = delta > 0
        ? `Additional instance requested for ${modelId}.`
        : `${modelId} now has ${result.instances} active instance${result.instances === 1 ? '' : 's'}.`;
      closeMenu({restoreFocus: true});
      void refresh();
    } catch (error) {
      menuError.textContent = error.message;
      menuError.hidden = false;
    } finally {
      keepMemory.disabled = false;
      instancePlus.disabled = false;
      instanceMinus.disabled = (instanceCounts.get(modelId) || 0) <= 1;
      poolAction.disabled = false;
    }
  }
  instanceMinus.addEventListener('click', () => scaleInstances(-1));
  instancePlus.addEventListener('click', () => scaleInstances(1));
  document.addEventListener('click', event => {
    if (!menu.hidden && !menu.contains(event.target) && !event.target.closest('.inference-menu-trigger')) closeMenu();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !menu.hidden) closeMenu({restoreFocus: true});
  });
  menu.addEventListener('keydown', event => {
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const items = [
      keepMemory,
      ...(!instanceStepper.hidden ? [instanceMinus, instancePlus] : []),
      poolAction,
    ].filter(item => !item.disabled);
    const current = items.indexOf(document.activeElement);
    const next = event.key === 'Home' ? 0
      : event.key === 'End' ? items.length - 1
        : event.key === 'ArrowDown' ? (current + 1) % items.length
          : (current - 1 + items.length) % items.length;
    items[next].focus();
  });
  window.addEventListener('resize', () => closeMenu());
  window.addEventListener('scroll', () => closeMenu(), true);

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
