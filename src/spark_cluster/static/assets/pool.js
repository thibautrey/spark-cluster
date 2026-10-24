(() => {
  let state = {models: [], clusters: [], jobs: []};
  let editing = null;
  let polling = false;
  const root = document.getElementById('model-pool');
  const dialog = document.getElementById('pool-model-settings');
  const modelField = document.getElementById('pool-settings-model');
  const message = document.getElementById('pool-message');
  const field = id => document.getElementById(id);
  const auth = () => {
    const token = field('pool-admin-token').value.trim() || document.getElementById('launch-admin-token')?.value.trim();
    return token ? {Authorization: `Bearer ${token}`} : {};
  };
  const pending = id => state.jobs.findLast(job => job.model_id === id && ['queued', 'waiting', 'loading'].includes(job.status));
  function render() {
    field('pool-summary').textContent = `${state.clusters.filter(c => c.available).length} cluster${state.clusters.filter(c => c.available).length === 1 ? '' : 's'} available · ${state.models.filter(m => m.placements.length).length} models loaded`;
    field('pool-model-list').innerHTML = state.models.filter(model => (field('pool-show-unavailable').checked || model.loadable || model.placements.length) && model.id.toLowerCase().includes(field('pool-search').value.toLowerCase())).sort((a, b) => Number(b.placements.length > 0) - Number(a.placements.length > 0) || a.id.localeCompare(b.id)).map(model => {
      const job = pending(model.id);
      const loaded = model.placements.length > 0;
      const retained = model.preferences.keep_loaded || model.placements.some(p => p.keep_loaded);
      const last = state.jobs.findLast(j => j.model_id === model.id);
      return `<article class="pool-model-card" data-pool-model="${escapeHtml(model.id)}">
        <div class="pool-model-heading"><h3>${escapeHtml(model.id)}</h3><span class="pill ${loaded ? 'ready' : ''}">${retained && loaded ? 'Kept loaded' : retained ? 'Retention enabled' : loaded ? 'Loaded' : 'On demand'}</span></div>
        <p>${loaded ? model.placements.map(p => escapeHtml(p.cluster_name)).join(', ') : model.preferences.cluster_id ? `Required cluster: ${escapeHtml(state.clusters.find(c => c.id === model.preferences.cluster_id)?.name || 'unavailable')}` : 'Automatic placement in the pool'}</p>
        ${job ? `<p class="pool-job" role="status">${escapeHtml(job.message)}</p>` : last?.status === 'failed' ? `<p class="pool-error">${escapeHtml(last.message)}</p>` : ''}
        <div class="pool-model-actions"><button class="primary" data-pool-action="load" ${job || !model.loadable ? 'disabled' : ''}>${job ? 'Waiting…' : loaded ? 'Loaded' : 'Load model'}</button><button class="secondary" data-pool-action="settings">Settings</button>${loaded ? '<button class="secondary" data-pool-action="unload">Unload</button>' : ''}</div>
        ${!model.loadable ? '<small>Runtime support is required before loading this model.</small>' : ''}
      </article>`;
    }).join('') || '<div class="empty-state">Download a model or connect a cluster to add it to the pool.</div>';
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
    field('pool-settings-error').textContent = '';
    field('pool-keep-loaded').checked = model.preferences.keep_loaded || model.placements.some(p => p.keep_loaded);
    field('pool-cluster').innerHTML = '<option value="">Automatic — pool</option>' + state.clusters.map(cluster => `<option value="${escapeHtml(cluster.id)}">${escapeHtml(cluster.name)}${cluster.available ? '' : ' (unavailable)'}</option>`).join('');
    if (model.preferences.cluster_id && !state.clusters.some(c => c.id === model.preferences.cluster_id)) {
      field('pool-cluster').add(new Option('Unavailable saved cluster', model.preferences.cluster_id));
    }
    field('pool-cluster').value = model.preferences.cluster_id || '';
    const settings = model.last_settings;
    field('pool-saved-settings').textContent = settings ? JSON.stringify(settings, null, 2) : model.editable_settings ? 'The last successful launch settings will be reused. A first launch uses conservative defaults and checks every candidate cluster.' : 'The saved service configuration will be reused exactly.';
    field('pool-advanced-settings').hidden = !model.editable_settings;
    field('pool-context').value = settings?.context_length || 8192;
    field('pool-tensor').value = settings?.tensor_parallel || 1;
    field('pool-pipeline').value = settings?.pipeline_parallel || 1;
    field('pool-memory').value = settings?.gpu_memory_utilization || 0.82;
    field('pool-served-name').value = settings?.served_model_name || model.id.split('/').at(-1);
    dialog.showModal();
  }
  field('pool-settings-close').addEventListener('click', () => dialog.close());
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
      dialog.close();
      await refresh();
    } catch (error) { field('pool-settings-error').textContent = error.message; }
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
      await api(`/api/v1/pool/${action}`, {method: 'POST', headers: auth(), body: JSON.stringify({model_id: model.id})});
      message.textContent = action === 'load' ? 'Load queued. The pool will reuse the saved settings.' : 'Model unloaded and retention disabled.';
      await refresh();
    } catch (error) { message.textContent = error.message; button.disabled = false; }
  });
  window.loadModelInPool = async settings => {
    await api('/api/v1/pool/load', {method: 'POST', headers: launchAuthHeaders(), body: JSON.stringify({model_id: settings.model_id, settings})});
    await refresh();
  };
  field('pool-search').addEventListener('input', render);
  field('pool-show-unavailable').addEventListener('change', render);
  void refresh();
  setInterval(refresh, 3000);
})();
