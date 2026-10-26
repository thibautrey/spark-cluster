(() => {
  let state = {models: [], clusters: [], jobs: []};
  let editing = null;
  let polling = false;
  let lastMarkup = null;
  let activeFilter = 'all';
  const root = document.getElementById('model-pool');
  const sheet = document.getElementById('pool-model-settings');
  let sheetOpener = null;
  let sheetCloseTimer = null;
  const modelField = document.getElementById('pool-settings-model');
  const message = document.getElementById('pool-message');
  const field = id => document.getElementById(id);
  const auth = () => window.adminAuthHeaders();
  const gib = bytes => Number(bytes || 0) / 2 ** 30;
  const pending = id => state.jobs.findLast(job => job.model_id === id && ['queued', 'waiting', 'loading'].includes(job.status));

  function capacityContext(model) {
    const selected = window.selectedClusterContext?.() || 'all';
    const capacities = (model.cluster_capacity || []).filter(item => item.available);
    const fitting = capacities.filter(item => item.fits || item.can_load);
    if (selected === 'all') {
      const best = fitting.sort((a, b) => Number(b.fits) - Number(a.fits) || Number(b.available_gib || 0) - Number(a.available_gib || 0))[0];
      if (best) return {fits: true, capacity: best};
      const closest = capacities.filter(item => item.reason || item.missing_gib != null)
        .sort((a, b) => Number(a.missing_gib ?? Infinity) - Number(b.missing_gib ?? Infinity))[0];
      return {fits: false, capacity: closest || null};
    }
    const capacity = capacities.find(item => item.cluster_id === selected) || null;
    return {fits: Boolean(capacity?.fits || capacity?.can_load), capacity};
  }

  function modelTask(model) {
    const metadata = model.runtime_models?.data?.[0] || {};
    const task = String(metadata.task || '').toLowerCase();
    const id = model.id.toLowerCase();
    if (task.includes('embedding') || task.includes('feature') || id.includes('embed') || id.includes('bge-')) {
      return {id: 'embeddings', label: 'Embeddings', description: 'Fast vector embeddings for search and retrieval.', icon: '⌘'};
    }
    if (task.includes('speech') || task.includes('transcription') || id.includes('whisper')) {
      return {id: 'transcription', label: 'Transcription', description: 'Multilingual speech-to-text transcription.', icon: '≋'};
    }
    if (task.includes('image') || task.includes('vision') || id.includes('ocr')) {
      return {id: 'vision', label: 'Vision', description: 'Visual understanding and document processing.', icon: '◇'};
    }
    if (id.includes('coder') || id.includes('code')) {
      return {id: 'code', label: 'Code', description: 'Code generation and developer assistance.', icon: '〈〉'};
    }
    return {id: 'chat', label: 'Chat', description: id.includes('smollm') ? 'Compact chat for lightweight workloads.' : 'Conversational generation and reasoning.', icon: '✧'};
  }

  function modelState(model) {
    const loaded = model.placements.length > 0;
    const job = pending(model.id);
    const capacity = capacityContext(model);
    if (job) return {id: 'launching', label: job.status === 'waiting' ? 'Waiting' : 'Launching', loaded, job, capacity};
    if (loaded) return {id: 'running', label: '● Running', loaded, job: null, capacity};
    if (model.loadable && capacity.fits) return {id: 'ready', label: 'Ready to launch', loaded, job: null, capacity};
    return {id: 'attention', label: 'Needs attention', loaded, job: null, capacity};
  }

  function renderPoolOverview() {
    const selected = window.selectedClusterContext?.() || 'all';
    const cluster = state.clusters.find(item => item.id === selected) || state.clusters.find(item => item.available) || state.clusters[0];
    if (!cluster) {
      field('launch-pool-overview').innerHTML = '<div class="empty-state">Connect a compute pool to launch models.</div>';
      return;
    }
    const gpus = cluster.gpus || [];
    const healthy = cluster.available && (cluster.machines || []).every(machine => machine.available !== false);
    const gpuMarkup = gpus.length ? gpus.map(gpu => {
      const total = Number(gpu.total_gib || 0);
      const reserved = Number(gpu.reserved_gib || 0);
      const external = Number(gpu.external_gib || 0);
      const committed = Math.max(Number(gpu.used_gib || 0), reserved + external);
      const available = Number(gpu.available_gib || 0);
      const percent = total ? Math.min(100, committed / total * 100) : 0;
      const models = (cluster.resident_models || []).filter(model => model.gpu_ids?.includes(gpu.id));
      return `<div class="launch-gpu-meter">${window.performanceButton?.('gpu', JSON.stringify([cluster.id, String(gpu.id)]), `GPU ${gpu.index ?? ""} performance ↗`) || ""}
        <div><b>GPU ${escapeHtml(String(gpu.index ?? ''))}</b><strong>${committed.toFixed(1)} / ${total.toFixed(1)} GiB</strong></div>
        <span class="launch-gpu-track"><i style="width:${percent}%"></i></span>
        <div><small>${models.length ? 'Model memory' : external ? 'External use' : 'Idle'}</small><small>${available.toFixed(1)} GiB available</small></div>
      </div>`;
    }).join('') : '<div class="launch-gpu-meter unavailable"><small>Per-GPU memory telemetry unavailable</small></div>';
    field('launch-pool-overview').innerHTML = `<div class="launch-pool-identity">
      <div><span aria-hidden="true">▣</span><h2>${escapeHtml(cluster.name)}</h2><em class="${healthy ? 'healthy' : 'unavailable'}">${healthy ? 'Healthy' : 'Unavailable'}</em></div>
      ${window.performanceButton?.("pool", cluster.id, "Pool performance ↗") || ""}<p>${gpus.length || cluster.machines?.reduce((sum, machine) => sum + Number(machine.accelerators || 0), 0) || 0} GPUs connected · Automatic placement</p>
    </div><div class="launch-gpu-grid">${gpuMarkup}</div>`;
  }

  function updateFilters(models) {
    const entries = new Map(models.map(model => {
      const task = modelTask(model);
      return [task.id, task.label];
    }));
    const selected = field('pool-task').value;
    field('pool-task').innerHTML = '<option value="all">All tasks</option>' + [...entries]
      .sort((a, b) => a[1].localeCompare(b[1]))
      .map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join('');
    field('pool-task').value = entries.has(selected) ? selected : 'all';
  }

  function progressMarkup(job) {
    if (!job) return '';
    const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
    return `<div class="launch-card-progress" aria-live="polite"><span><i style="width:${progress}%"></i></span><small>${escapeHtml(job.message || 'Preparing runtime')} · ${Math.round(progress)}%</small></div>`;
  }

  function renderCard(model) {
    const view = modelState(model);
    const task = modelTask(model);
    const runtime = view.job?.runtime || model.runtime;
    const engine = runtime?.name || 'Runtime unavailable';
    const format = /awq/i.test(model.id) ? 'AWQ' : /gguf/i.test(model.id) ? 'GGUF' : null;
    const [publisher, ...nameParts] = model.id.split('/');
    const name = nameParts.join('/') || publisher;
    const retained = model.preferences.keep_loaded || model.placements.some(item => item.keep_loaded);
    const reservation = model.placements.flatMap(item => Object.values(item.budget_gib || {}));
    const required = reservation.length ? reservation.reduce((sum, value) => sum + Number(value), 0) : Number(view.capacity.capacity?.required_gib || gib(model.size_bytes));
    const settings = model.last_settings || {};
    const degraded = settings.cpu_offload_gib > 0 ? `${settings.cpu_offload_gib} GiB RAM / GPU` : settings.gpu_layers != null ? `${settings.gpu_layers} GPU layers · CPU + RAM` : '';
    const last = state.jobs.findLast(item => item.model_id === model.id);
    let condition;
    if (view.id === 'running') condition = '<i>✓</i><span>Serving requests · Ready for your app</span>';
    else if (view.id === 'ready') condition = '<i>✓</i><span>Fits in pool · Automatic GPU placement</span>';
    else if (view.id === 'launching') condition = `<i>↻</i><span>${escapeHtml(view.job.message || 'Preparing model')}</span>`;
    else condition = `<i>ⓘ</i><span>${escapeHtml(view.capacity.capacity?.reason || runtime?.blockers?.[0] || 'Runtime or memory requirements need review.')}</span>`;
    const action = view.id === 'running' ? 'unload' : view.id === 'ready' ? 'load' : view.id === 'attention' ? 'settings' : 'load';
    const actionText = view.id === 'running' ? '■ Unload model' : view.id === 'ready' ? '▷ Launch model' : view.id === 'attention' ? 'Review requirements' : 'Launching…';
    const disabled = view.id === 'launching' ? 'disabled' : '';
    const concurrency = model.placements.find(item => item.concurrency)?.concurrency;
    return `<article class="launch-model-card ${view.id}" data-pool-model="${escapeHtml(model.id)}" data-task="${task.id}">
      <header><span class="launch-model-icon ${task.id}">${task.icon}</span><em class="launch-card-state ${view.id}">${escapeHtml(view.label)}</em></header>
      <div class="launch-model-copy"><small>${escapeHtml(publisher)}</small><h3>${escapeHtml(name)}</h3><p>${escapeHtml(task.description)}</p></div>
      <div class="launch-model-tags"><span>${escapeHtml(task.label)}</span><span>${escapeHtml(engine)}</span>${format ? `<span>${format}</span>` : ''}${degraded ? `<span>${escapeHtml(degraded)}</span>` : ''}</div>
      <div class="launch-model-metrics"><span>Est. VRAM <b>${required.toFixed(1)} GiB</b></span><span>${retained ? '↳ Keep loaded' : 'On demand'}</span></div>
      <div class="launch-model-condition ${view.id}">${condition}</div>
      ${concurrency ? `<small class="launch-model-concurrency">Automatic concurrency ${concurrency.running} / ${concurrency.limit}${concurrency.waiting ? ` · ${concurrency.waiting} waiting` : ''}</small>` : ''}
      ${progressMarkup(view.job)}
      ${!view.job && last?.status === 'failed' ? `<p class="pool-error">${escapeHtml(last.message)}</p>` : ''}
      <footer>${window.performanceButton?.("model", model.id) || ""}<button class="launch-card-action ${view.id === 'ready' ? 'primary' : 'secondary'}" data-pool-action="${action}" ${disabled}>${actionText}</button><button class="launch-card-settings secondary" data-pool-action="settings" aria-label="Settings for ${escapeHtml(model.id)}" title="Model settings">⚙</button></footer>
    </article>`;
  }

  function render() {
    renderPoolOverview();
    // Installed models stay visible so blocked launches can expose their requirements.
    const eligible = state.models;
    updateFilters(eligible);
    const views = eligible.map(model => ({model, view: modelState(model), task: modelTask(model)}));
    const running = views.filter(item => item.view.id === 'running' || item.view.id === 'launching').length;
    const ready = views.filter(item => item.view.id === 'ready').length;
    field('launch-count-all').textContent = eligible.length;
    field('launch-count-running').textContent = running;
    field('launch-count-ready').textContent = ready;
    const query = field('pool-search').value.trim().toLowerCase();
    const task = field('pool-task').value;
    const sort = field('pool-sort').value;
    const ranks = {running: 0, launching: 1, ready: 2, attention: 3};
    const visible = views.filter(item => {
      const status = activeFilter === 'all' || (activeFilter === 'running' ? ['running', 'launching'].includes(item.view.id) : item.view.id === 'ready');
      const matches = !query || item.model.id.toLowerCase().includes(query);
      return status && matches && (task === 'all' || item.task.id === task);
    }).sort((a, b) => sort === 'name'
      ? a.model.id.localeCompare(b.model.id)
      : sort === 'memory'
        ? Number(a.view.capacity.capacity?.required_gib ?? Infinity) - Number(b.view.capacity.capacity?.required_gib ?? Infinity)
        : ranks[a.view.id] - ranks[b.view.id] || a.model.id.localeCompare(b.model.id));
    field('launch-model-total').textContent = `${visible.length} model${visible.length === 1 ? '' : 's'}`;
    const markup = visible.map(item => renderCard(item.model)).join('') || '<div class="launch-model-empty"><b>No models match this view</b><span>Try another status, task or search.</span></div>';
    if (markup !== lastMarkup) {
      field('pool-model-list').innerHTML = markup;
      lastMarkup = markup;
    }
  }

  async function refresh() {
    if (polling) return;
    polling = true;
    try {
      state = await api('/api/v1/pool');
      message.textContent = '';
      render();
    } catch (error) {
      message.textContent = error.message;
    } finally {
      polling = false;
    }
  }

  function openSettings(model) {
    editing = model;
    modelField.textContent = model.id;
    clearTimeout(sheetCloseTimer);
    sheetOpener = document.activeElement;
    field('pool-settings-error').textContent = '';
    field('pool-settings-error').hidden = true;
    field('pool-keep-loaded').checked = model.preferences.keep_loaded || model.placements.some(item => item.keep_loaded);
    field('pool-cluster').innerHTML = '<option value="">Automatic — pool</option>' + state.clusters.map(cluster => `<option value="${escapeHtml(cluster.id)}">${escapeHtml(cluster.name)}${cluster.available ? '' : ' (unavailable)'}</option>`).join('');
    if (model.preferences.cluster_id && !state.clusters.some(cluster => cluster.id === model.preferences.cluster_id)) field('pool-cluster').add(new Option('Unavailable saved pool', model.preferences.cluster_id));
    field('pool-cluster').value = model.preferences.cluster_id || '';
    field('pool-api-url').textContent = `${location.origin}/v1`;
    field('pool-settings-state').textContent = model.placements.length ? (model.preferences.keep_loaded ? 'Kept loaded' : 'Running') : 'On demand';
    const runtime = pending(model.id)?.runtime || model.runtime;
    field('pool-runtime-name').textContent = runtime?.name || 'No compatible runtime detected';
    field('pool-runtime-id').textContent = runtime?.engine_id ? `Engine: ${runtime.engine_id}` : '';
    field('pool-runtime-ready').textContent = runtime?.executable ? 'Ready' : 'Unavailable';
    field('pool-runtime-ready').className = runtime?.executable ? 'ready' : 'unavailable';
    field('pool-runtime-reason').textContent = runtime?.reason || 'Runtime detection runs during pool preflight.';
    const notes = [...(runtime?.warnings || []), ...(runtime?.blockers || [])];
    field('pool-runtime-notes').hidden = !notes.length;
    field('pool-runtime-notes').innerHTML = notes.map(note => `<span>${escapeHtml(note)}</span>`).join('');
    field('pool-placement-note').textContent = model.preferences.cluster_id ? 'This compute pool is required on the next load.' : 'Automatic placement checks memory and runtime compatibility.';
    const settings = model.last_settings;
    field('pool-saved-settings').textContent = settings ? JSON.stringify(settings, null, 2) : 'The first launch uses automatic defaults.';
    field('pool-advanced-settings').hidden = !model.editable_settings;
    field('pool-context').value = settings?.context_length || 8192;
    field('pool-tensor').value = settings?.tensor_parallel || 1;
    field('pool-pipeline').value = settings?.pipeline_parallel || 1;
    field('pool-memory').value = settings?.gpu_memory_utilization || 0.82;
    field('pool-cpu-offload').value = settings?.cpu_offload_gib || 0;
    field('pool-gpu-layers').value = settings?.gpu_layers ?? '';
    field('pool-offload-field').hidden = !['vllm', 'sglang'].includes(runtime?.engine_id);
    field('pool-layers-field').hidden = runtime?.engine_id !== 'llama-cpp';
    field('pool-memory-mode').value = settings?.memory_budget_mode || (settings ? 'fixed' : 'auto');
    field('pool-served-name').value = settings?.served_model_name || model.id.split('/').at(-1);
    sheet.hidden = false;
    document.body.classList.add('bottom-sheet-open');
    requestAnimationFrame(() => {
      sheet.classList.add('open');
      field('pool-settings-close').focus();
    });
  }

  function closeSettings() {
    if (sheet.hidden) return;
    sheet.classList.remove('open');
    document.body.classList.remove('bottom-sheet-open');
    clearTimeout(sheetCloseTimer);
    sheetCloseTimer = setTimeout(() => {
      sheet.hidden = true;
      sheetOpener?.focus();
    }, 220);
  }

  field('pool-settings-close').addEventListener('click', closeSettings);
  field('pool-settings-cancel').addEventListener('click', closeSettings);
  field('pool-settings-backdrop').addEventListener('click', closeSettings);
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !sheet.hidden) closeSettings();
  });
  field('pool-copy-api').addEventListener('click', async () => {
    await navigator.clipboard.writeText(field('pool-api-url').textContent);
    field('pool-copy-api').textContent = 'Copied';
    setTimeout(() => { field('pool-copy-api').textContent = 'Copy'; }, 1200);
  });
  field('pool-settings-form').addEventListener('submit', async event => {
    event.preventDefault();
    const button = field('pool-settings-save');
    button.disabled = true;
    const lastSettings = editing.editable_settings ? {
      ...(editing.last_settings || {}), model_id: editing.id,
      served_model_name: field('pool-served-name').value.trim(),
      context_length: Number(field('pool-context').value), tensor_parallel: Number(field('pool-tensor').value),
      pipeline_parallel: Number(field('pool-pipeline').value), gpu_memory_utilization: Number(field('pool-memory').value),
      memory_budget_mode: field('pool-memory-mode').value,
      cpu_offload_gib: field('pool-offload-field').hidden ? 0 : Number(field('pool-cpu-offload').value),
      gpu_layers: field('pool-layers-field').hidden || field('pool-gpu-layers').value === '' ? null : Number(field('pool-gpu-layers').value),
    } : null;
    try {
      await api(`/api/v1/pool/models/${encodeURIComponent(editing.id)}/settings`, {method: 'PUT', headers: auth(), body: JSON.stringify({keep_loaded: field('pool-keep-loaded').checked, cluster_id: field('pool-cluster').value || null, last_settings: lastSettings})});
      closeSettings();
      await refresh();
    } catch (error) {
      field('pool-settings-error').textContent = error.message;
      field('pool-settings-error').hidden = false;
    } finally {
      button.disabled = false;
    }
  });

  root.addEventListener('click', async event => {
    const filter = event.target.closest('[data-launch-filter]');
    if (filter) {
      activeFilter = filter.dataset.launchFilter;
      root.querySelectorAll('[data-launch-filter]').forEach(button => button.classList.toggle('active', button === filter));
      lastMarkup = null;
      render();
      return;
    }
    const button = event.target.closest('[data-pool-action]');
    if (!button) return;
    const model = state.models.find(item => item.id === button.closest('[data-pool-model]').dataset.poolModel);
    if (button.dataset.poolAction === 'settings') {
      openSettings(model);
      return;
    }
    button.disabled = true;
    try {
      const action = button.dataset.poolAction;
      await api(`/api/v1/pool/${action}`, {method: 'POST', headers: auth(), body: JSON.stringify({model_id: model.id, cluster_id: window.selectedClusterContext?.() === 'all' ? null : window.selectedClusterContext?.()})});
      message.textContent = action === 'load' ? `${model.id} is queued for automatic placement.` : `${model.id} was unloaded.`;
      await refresh();
    } catch (error) {
      message.textContent = error.message;
      button.disabled = false;
    }
  });

  window.loadModelInPool = async settings => {
    await api('/api/v1/pool/load', {method: 'POST', headers: launchAuthHeaders(), body: JSON.stringify({model_id: settings.model_id, settings, cluster_id: window.selectedClusterContext?.() === 'all' ? null : window.selectedClusterContext?.()})});
    await refresh();
  };
  ['pool-search', 'pool-task', 'pool-sort'].forEach(id => field(id).addEventListener(id === 'pool-search' ? 'input' : 'change', () => { lastMarkup = null; render(); }));
  document.addEventListener('cluster-context-changed', () => { lastMarkup = null; render(); });
  void refresh();
  setInterval(refresh, 3000);
})();
