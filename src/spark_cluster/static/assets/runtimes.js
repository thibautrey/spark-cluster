(() => {
  const field = id => document.getElementById(`runtime-library-${id}`);
  const esc = value => escapeHtml(String(value ?? ''));
  const base = '/api/v1/runtime-library';
  let state = {runtimes: [], artifacts: [], jobs: [], hosts: []};
  let selected = null;
  let cursor = null;
  let versionSequence = 0;
  let refreshing = false;
  let timer = null;
  const visible = () => location.hash === '#runtimes' && !document.hidden;
  const message = text => { field('message').textContent = text; };
  const platformLabels = {
    'linux/arm64': 'Linux ARM64 · DGX Spark',
    'linux/amd64': 'Linux x86-64 · NVIDIA servers',
    'darwin/arm64': 'macOS · Apple Silicon',
    'darwin/amd64': 'macOS · Intel',
    'windows/amd64': 'Windows · x64',
    'windows/arm64': 'Windows · ARM64',
  };

  // Navigation's mobile selector shares the same route catalogue.
  const option = document.createElement('option');
  option.value = 'runtimes'; option.textContent = 'Runtimes';
  document.getElementById('route-select').append(option);
  if (location.hash === '#runtimes') document.getElementById('route-select').value = 'runtimes';

  function render() {
    field('list').innerHTML = state.runtimes.map(runtime => `<button type="button" class="runtime-library-item ${selected?.id === runtime.id ? 'selected' : ''}" data-runtime="${esc(runtime.id)}" aria-pressed="${selected?.id === runtime.id}"><b>${esc(runtime.name)}</b><small>${esc(runtime.description)}</small></button>`).join('');
    field('artifacts').innerHTML = state.artifacts.length ? state.artifacts.map(item => {
      const target = field('target').value;
      const host = target.startsWith('host:') ? state.hosts.find(value => value.id === target.slice(5)) : null;
      const compatible = host
        ? item.artifact_kind === 'native' && host.platform === item.platform
        : Boolean(target) && item.artifact_kind !== 'native' && item.platform.startsWith('linux/');
      const active = state.jobs.some(job => job.operation === 'install' && ['queued', 'running'].includes(job.status) && job.request.artifact_id === item.id);
      return `<article class="runtime-cache-item"><div><b>${esc(state.runtimes.find(runtime => runtime.id === item.runtime_id)?.name || item.runtime_id)} · ${esc(item.version)}</b><small>${esc(platformLabels[item.platform] || item.platform)} · ${item.artifact_kind === 'native' ? 'native package' : 'OCI image'} · ${(item.archive_bytes / 1024 ** 3).toFixed(2)} GiB · SHA256 ${esc(item.sha256.slice(0, 12))}…</small></div><button class="secondary" type="button" data-install="${esc(item.id)}" ${compatible && !active ? '' : 'disabled'}>${item.artifact_kind === 'native' ? 'Install on host' : 'Install on pool'}</button></article>`;
    }).join('') : '<p class="empty-state">No runtime packages cached yet.</p>';
    field('jobs').innerHTML = state.jobs.slice(0, 12).map(job => `<article class="runtime-job"><div><b>${esc(job.operation === 'download' ? 'Controller download' : `Install · ${job.cluster_name}`)}</b><small>${esc(job.request.runtime_id || job.runtime_id)} ${esc(job.request.version || '')} · ${esc(job.message)}</small>${job.nodes.length ? `<small>${job.nodes.map(node => `${esc(node.address)}: ${esc(node.status)}`).join(' · ')}</small>` : ''}</div><span class="pill">${esc(job.status)}${['queued', 'running'].includes(job.status) ? ` · ${job.progress}%` : ''}</span></article>`).join('') || '<p class="empty-state">No runtime activity.</p>';
    const platform = field('platform').value;
    field('download-button').disabled = platform.startsWith('linux/') && !state.download_available;
    document.getElementById('runtime-host-count').textContent = state.hosts.length;
    document.getElementById('runtime-host-list').innerHTML = state.hosts.length ? state.hosts.map(host => `<article class="runtime-host"><div><b>${esc(host.name)}</b><small>${esc(platformLabels[host.platform] || host.platform)} · ${esc(host.username)}@${esc(host.address)}</small></div><button type="button" class="secondary" data-remove-host="${esc(host.id)}">Remove</button></article>`).join('') : '<p class="empty-state">No native hosts registered.</p>';
  }

  async function versions(next = '') {
    const sequence = ++versionSequence;
    field('more').disabled = true;
    field('page').textContent = 'Loading registry versions…';
    try {
      const platform = field('platform').value;
      const result = await api(`${base}/runtimes/${encodeURIComponent(selected.id)}/versions?cursor=${encodeURIComponent(next)}&platform=${encodeURIComponent(platform)}`, {promptForAdmin: false});
      if (sequence !== versionSequence) return;
      cursor = result.next_cursor;
      field('versions').innerHTML = result.versions.map(item => `<option value="${esc(item.version)}">${esc(item.platforms.join(', ') || 'Architecture checked before download')}</option>`).join('');
      field('page').textContent = `${result.versions.length} versions · ${result.ordering}${result.stale ? ' · cached, source unavailable' : ''}`;
      field('more').hidden = !cursor;
      field('more').disabled = false;
    } catch (error) {
      if (sequence !== versionSequence) return;
      field('page').textContent = error.message;
      field('more').hidden = true;
    }
  }

  function select(runtime) {
    selected = runtime;
    field('detail').hidden = false;
    field('name').textContent = runtime.name;
    field('description').textContent = runtime.description;
    field('project').href = runtime.project_url;
    field('hardware').textContent = runtime.hardware;
    field('note').textContent = runtime.note;
    const previousPlatform = field('platform').value;
    field('platform').innerHTML = runtime.platforms.map(platform => `<option value="${esc(platform)}">${esc(platformLabels[platform] || platform)}</option>`).join('');
    field('platform').value = runtime.platforms.includes(previousPlatform) ? previousPlatform : runtime.platforms[0];
    field('version').value = '';
    field('versions').replaceChildren();
    cursor = null;
    field('more').hidden = true;
    render();
    void versions();
  }

  async function refresh(interactive = false) {
    if (refreshing || !visible()) return;
    refreshing = true;
    try {
      const [library, pools] = await Promise.all([
        api(base, {promptForAdmin: interactive}), api('/api/v1/clusters', {promptForAdmin: false}),
      ]);
      state = library;
      const current = field('target').value;
      field('target').innerHTML = '<option value="">Choose a pool or native host</option>'
        + '<optgroup label="Linux compute pools">' + pools.clusters.map(pool => `<option value="${esc(pool.id)}">${esc(pool.name)}</option>`).join('') + '</optgroup>'
        + '<optgroup label="macOS & Windows hosts">' + state.hosts.map(host => `<option value="host:${esc(host.id)}">${esc(host.name)} · ${esc(platformLabels[host.platform] || host.platform)}</option>`).join('') + '</optgroup>';
      const targetExists = pools.clusters.some(pool => pool.id === current)
        || (current.startsWith('host:') && state.hosts.some(host => host.id === current.slice(5)));
      field('target').value = targetExists ? current : '';
      if (!selected && state.runtimes.length) select(state.runtimes[0]);
      render();
      message(state.download_available ? '' : 'Install skopeo on the controller to enable downloads. It is included in the Spark container image.');
    } catch (error) { message(error.message); }
    finally {
      refreshing = false;
      clearTimeout(timer);
      // Poll only while work is active, and only on the visible runtime page.
      if (visible() && state.jobs.some(job => ['queued', 'running'].includes(job.status))) {
        timer = setTimeout(() => refresh(), 3000);
      }
    }
  }

  field('list').addEventListener('click', event => {
    const button = event.target.closest('[data-runtime]');
    if (button) select(state.runtimes.find(item => item.id === button.dataset.runtime));
  });
  field('more').addEventListener('click', () => { if (cursor) void versions(cursor); });
  field('platform').addEventListener('change', () => {
    field('version').value = '';
    field('versions').replaceChildren();
    cursor = null;
    render();
    void versions();
  });
  field('refresh').addEventListener('click', () => refresh(true));
  field('target').addEventListener('change', render);
  field('download').addEventListener('submit', async event => {
    event.preventDefault();
    field('download-button').disabled = true;
    try {
      await api(`${base}/downloads`, {method: 'POST', body: JSON.stringify({
        runtime_id: selected.id, version: field('version').value.trim(), platform: field('platform').value,
      })});
      await refresh();
    } catch (error) { message(error.message); }
    finally { render(); }
  });
  field('artifacts').addEventListener('click', async event => {
    const button = event.target.closest('[data-install]');
    if (!button || !field('target').value) return;
    const target = field('target').value;
    const request = target.startsWith('host:')
      ? {artifact_id: button.dataset.install, host_id: target.slice(5)}
      : {artifact_id: button.dataset.install, cluster_id: target};
    button.disabled = true;
    try {
      await api(`${base}/installs`, {method: 'POST', body: JSON.stringify(request)});
      await refresh();
    } catch (error) { message(error.message); button.disabled = false; }
  });
  document.getElementById('runtime-host-form').addEventListener('submit', async event => {
    event.preventDefault();
    const button = document.getElementById('runtime-host-add');
    button.disabled = true;
    try {
      await api(`${base}/hosts`, {method: 'POST', body: JSON.stringify({
        name: document.getElementById('runtime-host-name').value.trim(),
        address: document.getElementById('runtime-host-address').value.trim(),
        username: document.getElementById('runtime-host-username').value.trim(),
        fingerprint: document.getElementById('runtime-host-fingerprint').value.trim(),
      })});
      event.currentTarget.reset();
      await refresh();
    } catch (error) { message(error.message); }
    finally { button.disabled = false; }
  });
  document.getElementById('runtime-host-list').addEventListener('click', async event => {
    const button = event.target.closest('[data-remove-host]');
    if (!button) return;
    button.disabled = true;
    try {
      await api(`${base}/hosts/${encodeURIComponent(button.dataset.removeHost)}`, {method: 'DELETE'});
      await refresh();
    } catch (error) { message(error.message); button.disabled = false; }
  });
  const checkVisibility = () => { clearTimeout(timer); if (visible()) void refresh(); };
  window.addEventListener('hashchange', checkVisibility);
  document.addEventListener('visibilitychange', checkVisibility);
  checkVisibility();
})();
