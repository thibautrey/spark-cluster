(() => {
  let state = {models: [], clusters: [], jobs: []};
  let editing = null;
  let polling = false;
  let lastMarkup = null;
  const root = document.getElementById('model-pool');
  const sheet = document.getElementById('pool-model-settings');
  let sheetOpener = null;
  let sheetCloseTimer = null;
  const modelField = document.getElementById('pool-settings-model');
  const message = document.getElementById('pool-message');
  const field = id => document.getElementById(id);
  const auth = () => window.adminAuthHeaders();

  const pending = id => state.jobs.findLast(job => job.model_id === id && ['queued', 'waiting', 'loading'].includes(job.status));
  function capacityContext(model) {
    const selected = window.selectedClusterContext?.() || 'all';
    const capacities = (model.cluster_capacity || []).filter(item => item.available);
    const fitting = capacities.filter(item => item.fits);
    if (selected === 'all') {
      if (fitting.length) {
        const best = fitting.sort((a, b) => b.available_gib - a.available_gib)[0];
        return {fits: true, selected, capacity: best, alternatives: fitting};
      }
      const closest = capacities.filter(item => item.missing_gib !== null)
        .sort((a, b) => a.missing_gib - b.missing_gib)[0];
      return {fits: false, selected, capacity: closest || null, alternatives: []};
    }
    const capacity = capacities.find(item => item.cluster_id === selected) || null;
    return {
      fits: capacity ? capacity.fits : false,
      selected,
      capacity,
      alternatives: fitting.filter(item => item.cluster_id !== selected),
    };
  }
  function capacityMarkup(context) {
    const capacity = context.capacity;
    if (!capacity) {
      const alternative = context.alternatives[0];
      return `<div class="pool-model-capacity unknown"><span>Memory preflight unavailable for this cluster</span>${alternative ? `<small>Fits on ${escapeHtml(alternative.cluster_name)} · switch cluster or choose All</small>` : ''}</div>`;
    }
    if (context.fits) {
      const location = context.selected === 'all' ? `Fits on ${escapeHtml(capacity.cluster_name)}` : `Fits in ${escapeHtml(capacity.cluster_name)}`;
      return `<div class="pool-model-capacity fits"><span>✓ ${location}</span><small>${capacity.required_gib.toFixed(1)} GiB required · ${capacity.available_gib.toFixed(1)} GiB available</small></div>`;
    }
    const alternative = context.alternatives[0];
    return `<div class="pool-model-capacity missing"><span>Needs ${capacity.missing_gib.toFixed(1)} GiB more on ${escapeHtml(capacity.cluster_name)}</span>${alternative ? `<small>Fits on ${escapeHtml(alternative.cluster_name)} · switch cluster or choose All</small>` : '<small>No compatible cluster currently has enough memory</small>'}</div>`;
  }
  function recentReady(id) {
    const now = Date.now() / 1000;
    return state.jobs.findLast(job =>
      job.model_id === id && job.status === 'ready'
      && job.completed_at && now - job.completed_at < 12
    );
  }
  function loadProgressMarkup(job) {
    if (!job || !['queued', 'waiting', 'loading', 'ready'].includes(job.status)) return '';
    const ready = job.status === 'ready';
    const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
    const machineRows = (job.machines || []).map(machine => {
      const machineProgress = Math.max(0, Math.min(100, Number(machine.progress || 0)));
      const memory = ready && machine.memory_after_gib !== null
        ? `${Number(machine.memory_after_gib || 0).toFixed(1)} / ${Number(machine.memory_total_gib || 0).toFixed(1)} GiB GPU memory`
        : machine.status === 'starting' ? 'Preparing runtime' : machine.status === 'failed' ? 'Load failed' : 'Loading runtime';
      return `<div class="pool-machine-progress ${ready ? 'ready' : ''}">
        <div><span><b>${escapeHtml(machine.name || machine.id)}</b><small>${escapeHtml(memory)}</small></span><em>${Math.round(machineProgress)}%</em></div>
        <div class="pool-progress-track"><i style="width:${machineProgress}%"></i></div>
      </div>`;
    }).join('');
    return `<section class="pool-load-progress ${ready ? 'ready' : 'active'}" aria-live="polite">
      <header><span><i>${ready ? '✓' : '↻'}</i><b>${ready ? 'Model ready' : escapeHtml(job.message || 'Loading model')}</b></span><strong>${Math.round(progress)}%</strong></header>
      <div class="pool-progress-track global"><i style="width:${progress}%"></i></div>
      ${machineRows ? `<div class="pool-machine-list">${machineRows}</div>` : ''}
    </section>`;
  }

  function render() {
    const markup = state.models.filter(model => (field('pool-show-unavailable').checked || model.loadable || model.placements.length) && model.id.toLowerCase().includes(field('pool-search').value.toLowerCase())).sort((a, b) => Number(b.placements.length > 0) - Number(a.placements.length > 0) || a.id.localeCompare(b.id)).map(model => {
      const job = pending(model.id);
      const completedJob = recentReady(model.id);
      const visibleJob = job || completedJob;
      const loaded = model.placements.length > 0;
      const retained = model.preferences.keep_loaded || model.placements.some(p => p.keep_loaded);
      const last = state.jobs.findLast(j => j.model_id === model.id);
      const runtime = visibleJob?.runtime || model.runtime;
      const runtimeName = runtime?.name || 'Runtime detection unavailable';
      const runtimeState = runtime?.selection === 'configured' ? 'Configured' : 'Auto-detected';
      const capacity = capacityContext(model);
      const canLoad = model.loadable && capacity.fits;
      return `<article class="pool-model-card" data-pool-model="${escapeHtml(model.id)}">
        <div class="pool-model-heading"><h3>${escapeHtml(model.id)}</h3><span class="pill ${loaded ? 'ready' : ''}">${retained && loaded ? 'Kept loaded' : retained ? 'Retention enabled' : loaded ? 'Loaded' : 'On demand'}</span></div>
        <p>${loaded ? model.placements.map(p => escapeHtml(p.cluster_name)).join(', ') : model.preferences.cluster_id ? `Required cluster: ${escapeHtml(state.clusters.find(c => c.id === model.preferences.cluster_id)?.name || 'unavailable')}` : 'Automatic placement in the pool'}</p>
        <div class="pool-model-runtime"><span>⚡</span><div><b>${escapeHtml(runtimeName)}</b><small>${escapeHtml(runtimeState)}</small></div></div>
        ${capacityMarkup(capacity)}
        ${loadProgressMarkup(visibleJob)}
        ${!visibleJob && last?.status === 'failed' ? `<p class="pool-error">${escapeHtml(last.message)}</p>` : ''}
        <div class="pool-model-actions"><button class="primary" data-pool-action="load" ${job || !canLoad ? 'disabled' : ''} title="${!canLoad && capacity.capacity ? `${capacity.capacity.missing_gib.toFixed(1)} GiB additional memory required` : ''}">${job ? 'Waiting…' : loaded ? 'Loaded' : 'Load model'}</button><button class="secondary" data-pool-action="settings">Settings</button>${loaded ? '<button class="secondary" data-pool-action="unload">Unload</button>' : ''}</div>
        ${!model.loadable ? '<small>Runtime support is required before loading this model.</small>' : ''}
      </article>`;
    }).join('') || '<div class="empty-state">Download a model or connect a cluster to add it to the pool.</div>';
    if (markup !== lastMarkup) { field('pool-model-list').innerHTML = markup; lastMarkup = markup; }
  }
  async function refresh() {
    if (polling) return;
    polling = true;
    try { state = await api('/api/v1/pool'); render(); }
    catch (error) { message.textContent = error.message; }
    finally { polling = false; }
  }
  function openSettings(model) {
    editing = model;
    modelField.textContent = model.id;
    clearTimeout(sheetCloseTimer);
    sheetOpener = document.activeElement;
    field('pool-settings-error').textContent = '';
    field('pool-settings-error').hidden = true;
    field('pool-keep-loaded').checked = model.preferences.keep_loaded || model.placements.some(p => p.keep_loaded);
    field('pool-cluster').innerHTML = '<option value="">Automatic — pool</option>' + state.clusters.map(cluster => `<option value="${escapeHtml(cluster.id)}">${escapeHtml(cluster.name)}${cluster.available ? '' : ' (unavailable)'}</option>`).join('');
    if (model.preferences.cluster_id && !state.clusters.some(c => c.id === model.preferences.cluster_id)) {
      field('pool-cluster').add(new Option('Unavailable saved cluster', model.preferences.cluster_id));
    }
    field('pool-cluster').value = model.preferences.cluster_id || '';
    field('pool-api-url').textContent = `${location.origin}/v1`;
    field('pool-settings-state').textContent = model.placements.length ? (model.preferences.keep_loaded ? 'Kept loaded' : 'Loaded') : 'On demand';
    const runtime = pending(model.id)?.runtime || model.runtime;
    field('pool-runtime-name').textContent = runtime?.name || 'No compatible runtime detected';
    field('pool-runtime-id').textContent = runtime?.engine_id ? `Engine: ${runtime.engine_id}` : '';
    field('pool-runtime-ready').textContent = runtime?.executable ? 'Ready' : 'Unavailable';
    field('pool-runtime-ready').className = runtime?.executable ? 'ready' : 'unavailable';
    field('pool-runtime-reason').textContent = runtime?.reason || 'Runtime detection will run during cluster preflight.';
    const notes = [...(runtime?.warnings || []), ...(runtime?.blockers || [])];
    field('pool-runtime-notes').hidden = !notes.length;
    field('pool-runtime-notes').innerHTML = notes.map(note => `<span>${escapeHtml(note)}</span>`).join('');
    field('pool-placement-note').textContent = model.preferences.cluster_id ? 'This cluster is required on the next load. Changing placement does not move a running model.' : 'Automatic placement checks residency, availability, memory and runtime compatibility.';
    const settings = model.last_settings;
    field('pool-saved-settings').textContent = settings ? JSON.stringify(settings, null, 2) : model.editable_settings ? 'The last successful launch settings will be reused. A first launch uses conservative defaults and checks every candidate cluster.' : 'The saved service configuration will be reused exactly.';
    field('pool-advanced-settings').hidden = !model.editable_settings;
    field('pool-context').value = settings?.context_length || 8192;
    field('pool-tensor').value = settings?.tensor_parallel || 1;
    field('pool-pipeline').value = settings?.pipeline_parallel || 1;
    field('pool-memory').value = settings?.gpu_memory_utilization || 0.82;
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
    } : null;
    try {
      await api(`/api/v1/pool/models/${encodeURIComponent(editing.id)}/settings`, {method: 'PUT', headers: auth(), body: JSON.stringify({keep_loaded: field('pool-keep-loaded').checked, cluster_id: field('pool-cluster').value || null, last_settings: lastSettings})});
      closeSettings();
      await refresh();
    } catch (error) {
      field('pool-settings-error').textContent = error.message;
      field('pool-settings-error').hidden = false;
    }
    finally { button.disabled = false; }
  });
  root.addEventListener('click', async event => {
    const button = event.target.closest('[data-pool-action]');
    if (!button) return;
    const model = state.models.find(m => m.id === button.closest('[data-pool-model]').dataset.poolModel);
    if (button.dataset.poolAction === 'settings') { openSettings(model); return; }
    button.disabled = true;
    try {
      const action = button.dataset.poolAction;
      await api(`/api/v1/pool/${action}`, {method: 'POST', headers: auth(), body: JSON.stringify({model_id: model.id, cluster_id: window.selectedClusterContext?.() === 'all' ? null : window.selectedClusterContext?.()})});
      message.textContent = action === 'load' ? 'Load queued. The pool will reuse the saved settings.' : 'Model unloaded and retention disabled.';
      await refresh();
    } catch (error) { message.textContent = error.message; button.disabled = false; }
  });
  window.loadModelInPool = async settings => {
    await api('/api/v1/pool/load', {method: 'POST', headers: launchAuthHeaders(), body: JSON.stringify({model_id: settings.model_id, settings, cluster_id: window.selectedClusterContext?.() === 'all' ? null : window.selectedClusterContext?.()})});
    await refresh();
  };
  field('pool-search').addEventListener('input', render);
  field('pool-show-unavailable').addEventListener('change', render);
  document.addEventListener('cluster-context-changed', render);
  void refresh();
  setInterval(refresh, 3000);
})();
