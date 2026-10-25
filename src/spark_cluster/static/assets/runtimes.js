(() => {
  const field = id => document.getElementById(`runtime-library-${id}`);
  const esc = value => escapeHtml(String(value ?? ''));
  const base = '/api/v1/runtime-library';
  const management = '/api/v1/runtime-management';
  const platformLabels = {
    'linux/arm64': 'Linux ARM64 · DGX Spark',
    'linux/amd64': 'Linux x86-64 · NVIDIA servers',
    'darwin/arm64': 'macOS · Apple Silicon',
    'darwin/amd64': 'macOS · Intel',
    'windows/amd64': 'Windows · x64',
    'windows/arm64': 'Windows · ARM64',
  };
  const stateLabels = {
    ready: 'Prepared',
    needs_preparation: 'Needs preparation',
    partial: 'Partial',
    unreachable: 'Unreachable',
    blocked: 'Blocked',
  };
  let library = {runtimes: [], artifacts: [], jobs: [], hosts: [], download_available: false};
  let clusters = [];
  let selectedCluster = null;
  let engines = null;
  let selectedRuntime = null;
  let cursor = null;
  let versionSequence = 0;
  let refreshing = false;
  let timer = null;
  let guides = {};
  let guidedWorkflow = null;
  let activeRuntimeTab = 'engines';

  const visible = () => location.hash === '#runtimes' && !document.hidden;
  const packageTarget = () => field('target').value;
  const activeLibraryJobs = () => library.jobs.some(job => ['queued', 'running'].includes(job.status));
  const activeEngineJobs = () => engines?.engines.some(engine => engine.active_job_id) || false;
  const roleFor = runtime => runtime.launch_adapter
    ? 'Managed launch'
    : runtime.native_repository ? 'Package only' : 'No launch adapter';
  const bytes = value => value >= 1024 ** 3
    ? `${(value / 1024 ** 3).toFixed(2)} GiB`
    : `${(value / 1024 ** 2).toFixed(1)} MiB`;

  const option = document.createElement('option');
  option.value = 'runtimes';
  option.textContent = 'Runtimes';
  document.getElementById('route-select').append(option);
  if (location.hash === '#runtimes') document.getElementById('route-select').value = 'runtimes';

  function resolveSelectedCluster(clusterState) {
    clusters = clusterState.clusters;
    let context = window.selectedClusterContext?.() || 'all';
    if (context === 'all' && clusters.length === 1) {
      context = clusters[0].id;
      try { localStorage.setItem('spark-cluster.cluster-context', context); } catch {}
      const selector = document.getElementById('cluster-select');
      if (selector) selector.value = context;
    }
    return context === 'all' ? null : clusters.find(cluster => cluster.id === context) || null;
  }

  function showState(kind, title, detail, action = '') {
    const node = document.getElementById('runtime-readiness-state');
    node.className = `runtime-readiness-state ${kind}`;
    const icon = kind === 'error' ? '!' : kind === 'locked' ? '⌁' : kind === 'empty' ? '○' : '…';
    node.innerHTML = `<span class="runtime-state-icon" aria-hidden="true">${icon}</span><div><b>${esc(title)}</b><small>${esc(detail)}</small></div>${action}`;
    node.hidden = false;
    document.getElementById('runtime-readiness-content').hidden = true;
    document.getElementById('runtime-library-advanced').hidden = true;
  }

  function runtimeDescription(engine) {
    return ({
      vllm: 'High-throughput LLM inference on NVIDIA GPUs with continuous batching.',
      sglang: 'Language and multimodal inference with prefix caching and structured generation.',
      'llama-cpp': 'Portable GGUF inference for CUDA and native hosts.',
      ollama: 'Local model management with an OpenAI-compatible serving API.',
      'tensorrt-llm': 'NVIDIA-optimized LLM serving through the trtllm-serve API.',
      triton: 'Multi-framework serving for prepared Triton model repositories.',
      tgi: 'Hugging Face text generation with continuous batching.',
      lmdeploy: 'TurboMind and PyTorch serving with an OpenAI-compatible API.',
      xinference: 'Multi-engine serving for models loaded from managed storage.',
      localai: 'Local OpenAI-compatible serving for GGUF models.',
      tensorfold: 'Specialized model execution for validated sparse and hybrid layouts.',
      deepinfer: 'Experimental kernels for supported NVIDIA accelerator families.',
    })[engine.id] || engine.message;
  }

  function runtimeIcon(engine) {
    return ({vllm: 'ϟ', sglang: '⌘', 'llama-cpp': 'λ', ollama: '○', 'tensorrt-llm': 'T', triton: '△', tgi: 'H', lmdeploy: 'L', xinference: 'X', localai: 'A', tensorfold: '◇', deepinfer: '◆'})[engine.id] || '◇';
  }

  function renderPoolOverview() {
    const container = document.getElementById('runtime-pool-overview');
    if (!selectedCluster || !engines) {
      container.innerHTML = '<div class="empty-state">Choose a compute pool to inspect runtime readiness.</div>';
      return;
    }
    const managed = engines.engines.filter(engine => engine.capability === 'managed_launch');
    const ready = managed.filter(engine => engine.ready).length;
    const assigned = Number(selectedCluster.gpu_count || 0);
    const verified = Math.max(0, ...managed.map(engine => Number(engine.nodes_ready || 0)));
    const healthy = selectedCluster.status === 'ready' && ready > 0;
    const units = selectedCluster.kind === 'local-docker'
      ? `1 host · ${assigned} GPU${assigned === 1 ? '' : 's'}`
      : `${selectedCluster.nodes?.length || assigned} nodes · ${assigned} GPU${assigned === 1 ? '' : 's'}`;
    container.innerHTML = `<div class="runtime-pool-identity"><div><span>▣</span><h2>${esc(selectedCluster.name)}</h2><i>⌄</i><em class="${healthy ? 'ready' : 'attention'}">${healthy ? 'Operational' : 'Needs attention'}</em></div><p>${esc(units)} · Automatic placement</p></div>
      <div class="runtime-pool-stat ready"><strong>${ready}<small> / ${managed.length}</small></strong><span>runtimes ready to launch</span></div>
      <div class="runtime-pool-stat"><strong>${verified}<small> / ${assigned}</small></strong><span>GPU targets verified · now</span></div>`;
  }

  function engineCard(engine) {
    const guide = guides[engine.id];
    const displayState = guide?.compatible && engine.state === 'blocked' ? 'needs_preparation' : engine.state;
    const status = engine.ready ? '✓ Ready to launch' : guide?.compatible ? 'Setup available' : stateLabels[displayState] || displayState;
    const count = engine.nodes_total
      ? `${engine.nodes_ready}/${engine.nodes_total} target${engine.nodes_total === 1 ? '' : 's'} ready`
      : guide?.compatible ? 'Guided setup available' : 'No managed targets';
    const activeWorkflow = guidedWorkflow?.engine_id === engine.id;
    const action = activeWorkflow && ['queued', 'running'].includes(guidedWorkflow.status)
      ? '<button class="primary" type="button" disabled>Preparing…</button>'
      : activeWorkflow && guidedWorkflow.status === 'failed'
        ? `<button class="primary" type="button" data-retry-workflow="${esc(guidedWorkflow.id)}">Retry failed step</button>`
        : engine.active_job_id
          ? '<button class="primary" type="button" disabled>Preparing…</button>'
          : engine.ready && !guide?.update_available
            ? '<button class="secondary" type="button" data-runtime-revalidate>Revalidate</button><a class="primary" href="#launch">Launch a model ↗</a>'
            : guide?.compatible
              ? `<button class="primary" type="button" data-guide-engine="${esc(engine.id)}">${guide.update_available ? 'Update runtime' : `Set up ${esc(engine.name)}`}</button>`
              : engine.prepare_supported
                ? `<button class="primary" type="button" data-prepare-engine="${esc(engine.id)}">Prepare on this pool</button>`
                : '<a class="secondary" href="#launch">Choose compatible model</a>';
    const nodeDots = engine.nodes_total
      ? `<span class="runtime-node-summary">${engine.nodes.map(node => `<i class="${esc(node.status)}"></i>`).join('')}<b>${esc(count)}</b></span>`
      : `<span class="runtime-node-summary"><b>${esc(count)}</b></span>`;
    const nodes = engine.nodes.length ? `<details class="runtime-engine-nodes"><summary>${nodeDots}<i>⌄</i></summary><div class="runtime-node-status-list">${engine.nodes.map(node => `<span><b>${esc(node.address)}</b><i>${esc(node.status.replaceAll('_', ' '))}</i></span>`).join('')}</div></details>` : `<div class="runtime-engine-nodes empty">${nodeDots}</div>`;
    const progress = engine.active_job_id ? `<div class="runtime-engine-progress"><div><i style="width:${Number(engine.active_job_progress || 0)}%"></i></div><span>${Number(engine.active_job_progress || 0).toFixed(0)}%</span></div>` : '';
    const workflow = activeWorkflow
      ? `<div class="runtime-guide-progress">${guidedWorkflow.steps.map(step => `<span class="${esc(step.status)}"><i></i><b>${esc(step.title)}</b><small>${esc(step.message)}</small></span>`).join('')}</div>${guidedWorkflow.error ? `<p class="runtime-guide-error">${esc(guidedWorkflow.error)}</p>` : ''}`
      : '';
    const version = guide?.version || engine.nodes.find(node => node.installed_version)?.installed_version || 'Configuration required';
    const technical = `<details class="runtime-guide-technical"><summary>Technical details</summary><div><code>${esc(engine.image || guide?.image || 'No configured image')}</code>${guide?.digest ? `<code>${esc(guide.digest)}</code>` : ''}<p>${esc(engine.message)}</p></div></details>`;
    return `<article class="runtime-engine-card ${esc(displayState)}" data-engine="${esc(engine.id)}">
      <div class="runtime-engine-main"><span class="runtime-engine-icon ${esc(engine.id)}">${runtimeIcon(engine)}</span><div class="runtime-engine-title"><div><b>${esc(engine.name)}</b></div><code>${esc(version)} · configured version</code></div><span class="runtime-engine-status ${esc(displayState)}">${esc(status)}</span></div>
      <p class="runtime-engine-description">${esc(runtimeDescription(engine))}</p>
      ${progress}${workflow}
      <div class="runtime-engine-footer">${nodes}<div class="runtime-engine-actions">${action}</div></div>
      ${technical}
    </article>`;
  }

  function visibleEngines() {
    if (!engines) return [];
    const query = document.getElementById('runtime-engine-search').value.trim().toLowerCase();
    const filter = document.getElementById('runtime-engine-filter').value;
    return engines.engines.filter(engine => engine.capability === 'managed_launch' || engine.image).filter(engine => {
      const matches = !query || `${engine.name} ${engine.id}`.toLowerCase().includes(query);
      const stateMatches = filter === 'all' || (filter === 'ready' ? engine.ready : !engine.ready);
      return matches && stateMatches;
    });
  }

  function showRuntimeTab(tab) {
    activeRuntimeTab = tab;
    document.querySelectorAll('[data-runtime-tab]').forEach(button => button.classList.toggle('active', button.dataset.runtimeTab === tab));
    const engineContent = document.getElementById('runtime-readiness-content');
    const packageContent = document.getElementById('runtime-library-advanced');
    engineContent.hidden = tab !== 'engines' || !engines;
    packageContent.hidden = tab !== 'packages';
    if (tab === 'packages') {
      packageContent.open = true;
      renderLibrary();
      if (!selectedRuntime && library.runtimes.length) {
        const platform = targetPlatform();
        selectRuntime(library.runtimes.find(runtime => runtime.platforms.includes(platform)) || library.runtimes[0]);
      }
    }
  }

  function renderEngines() {
    const visible = visibleEngines();
    const managed = engines.engines.filter(engine => engine.capability === 'managed_launch' || engine.image);
    document.getElementById('runtime-engine-count').textContent = managed.length;
    document.getElementById('runtime-package-count').textContent = library.runtimes.length;
    document.getElementById('runtime-engine-list').innerHTML = visible.length
      ? visible.map(engineCard).join('')
      : '<div class="empty-state"><b>No runtimes match this view</b><span>Try another state or search.</span></div>';
    document.getElementById('runtime-readiness-state').hidden = true;
    document.getElementById('runtime-readiness-activity').hidden = true;
    renderPoolOverview();
    showRuntimeTab(activeRuntimeTab);
  }

  function targetPlatform() {
    const target = packageTarget();
    if (target.startsWith('host:')) {
      return library.hosts.find(host => host.id === target.slice(5))?.platform || '';
    }
    if (target.startsWith('pool:')) {
      const pool = clusters.find(item => item.id === target.slice(5));
      if (!pool) return '';
      return engines?.cluster_id === pool.id && engines.platform
        ? engines.platform
        : pool.hardware === 'dgx-spark' ? 'linux/arm64' : 'linux/amd64';
    }
    return '';
  }

  function renderPackageTargets() {
    const current = packageTarget();
    const poolOption = selectedCluster
      ? `<option value="pool:${esc(selectedCluster.id)}">Current pool · ${esc(selectedCluster.name)}</option>`
      : '';
    const hostOptions = library.hosts.map(host => `<option value="host:${esc(host.id)}">Standalone host · ${esc(host.name)}</option>`).join('');
    field('target').innerHTML = `<option value="">Select a compute pool or native host</option>${poolOption}${hostOptions}`;
    const values = [...field('target').options].map(item => item.value);
    const fallback = selectedCluster
      ? `pool:${selectedCluster.id}`
      : library.hosts.length === 1 ? `host:${library.hosts[0].id}` : '';
    field('target').value = current && values.includes(current)
      ? current
      : fallback;
  }

  function renderHosts() {
    document.getElementById('runtime-host-count').textContent = library.hosts.length;
    document.getElementById('runtime-host-list').innerHTML = library.hosts.length
      ? library.hosts.map(host => `<article class="runtime-host"><div><b>${esc(host.name)}</b><small>${esc(platformLabels[host.platform] || host.platform)} · ${esc(host.username)}@${esc(host.address)}</small></div><button type="button" class="secondary" data-remove-host="${esc(host.id)}">Remove</button></article>`).join('')
      : '<p class="empty-state">No standalone hosts registered.</p>';
  }

  function renderLibrary() {
    document.getElementById('runtime-package-count').textContent = library.runtimes.length;
    renderPackageTargets();
    renderHosts();
    const platform = targetPlatform();
    field('list').innerHTML = library.runtimes.map(runtime => `<button type="button" class="runtime-library-item ${selectedRuntime?.id === runtime.id ? 'selected' : ''}" data-runtime="${esc(runtime.id)}" aria-pressed="${selectedRuntime?.id === runtime.id}"><b>${esc(runtime.name)}</b><small>${esc(runtime.description)}</small><span>${esc(roleFor(runtime))}</span></button>`).join('');
    field('artifacts').innerHTML = library.artifacts.length ? library.artifacts.map(item => {
      const compatible = Boolean(packageTarget()) && item.platform === platform;
      const active = library.jobs.some(job => job.operation === 'install' && ['queued', 'running'].includes(job.status) && job.request.artifact_id === item.id);
      return `<article class="runtime-cache-item"><div><b>${esc(library.runtimes.find(runtime => runtime.id === item.runtime_id)?.name || item.runtime_id)} · ${esc(item.version)}</b><small>${esc(platformLabels[item.platform] || item.platform)} · ${item.artifact_kind === 'native' ? 'native package' : 'OCI image'} · ${bytes(item.archive_bytes)} · Cached only</small></div><div class="runtime-cache-actions"><button class="secondary" type="button" data-install="${esc(item.id)}" ${compatible && !active ? '' : 'disabled'}>${active ? 'Installing…' : 'Install package'}</button><button class="secondary" type="button" data-remove-artifact="${esc(item.id)}" ${active ? 'disabled' : ''}>Remove cache</button></div></article>`;
    }).join('') : '<p class="empty-state">No runtime packages cached yet.</p>';
    const total = library.artifacts.reduce((sum, artifact) => sum + Number(artifact.archive_bytes || 0), 0);
    document.getElementById('runtime-cache-total').textContent = library.artifacts.length ? `${library.artifacts.length} · ${bytes(total)}` : '';
    const activity = document.getElementById('runtime-library-activity');
    activity.hidden = !library.jobs.length;
    field('jobs').innerHTML = library.jobs.slice(0, 12).map(job => `<article class="runtime-job"><div><b>${esc(job.operation === 'download' ? 'Controller cache' : `Package install · ${job.cluster_name || 'target'}`)}</b><small>${esc(job.request.runtime_id || job.runtime_id)} ${esc(job.request.version || '')} · ${esc(job.message)}</small></div><span class="pill">${esc(job.status)}${['queued', 'running'].includes(job.status) ? ` · ${job.progress}%` : ''}</span></article>`).join('');
    updatePackageForm();
  }

  function updatePackageForm() {
    const platform = targetPlatform();
    const compatible = selectedRuntime && platform && selectedRuntime.platforms.includes(platform);
    document.getElementById('runtime-library-platform-label').textContent = platform
      ? platformLabels[platform] || platform
      : 'Select a package target';
    field('download-button').disabled = !compatible || (platform.startsWith('linux/') && !library.download_available);
    if (selectedRuntime && platform && !compatible) {
      field('page').textContent = `${selectedRuntime.name} does not publish packages for ${platformLabels[platform] || platform}.`;
    }
  }

  async function versions(next = '') {
    if (!selectedRuntime) return;
    const platform = targetPlatform();
    if (!platform || !selectedRuntime.platforms.includes(platform)) {
      updatePackageForm();
      return;
    }
    const sequence = ++versionSequence;
    field('more').disabled = true;
    field('page').textContent = 'Loading published versions…';
    try {
      const result = await api(`${base}/runtimes/${encodeURIComponent(selectedRuntime.id)}/versions?cursor=${encodeURIComponent(next)}&platform=${encodeURIComponent(platform)}`, {promptForAdmin: false});
      if (sequence !== versionSequence) return;
      cursor = result.next_cursor;
      field('versions').innerHTML = result.versions.map(item => `<option value="${esc(item.version)}">${esc(item.platforms.join(', ') || 'Architecture verified before caching')}</option>`).join('');
      field('page').textContent = `${result.versions.length} versions · ${result.ordering}${result.stale ? ' · cached, source unavailable' : ''}`;
      field('more').hidden = !cursor;
      field('more').disabled = false;
    } catch (error) {
      if (sequence !== versionSequence) return;
      field('page').textContent = error.message;
      field('more').hidden = true;
    }
    updatePackageForm();
  }

  function selectRuntime(runtime) {
    selectedRuntime = runtime;
    field('detail').hidden = false;
    field('name').textContent = runtime.name;
    field('description').textContent = runtime.description;
    field('project').href = runtime.project_url;
    field('hardware').textContent = runtime.hardware;
    field('note').textContent = runtime.note;
    field('capability').textContent = roleFor(runtime);
    field('version').value = '';
    field('versions').replaceChildren();
    cursor = null;
    field('more').hidden = true;
    renderLibrary();
    void versions();
  }

  function revealPackageLibrary(open = false) {
    const advanced = document.getElementById('runtime-library-advanced');
    advanced.open = true;
    renderLibrary();
    if (open) showRuntimeTab('packages');
    const platform = targetPlatform();
    if (platform && (!selectedRuntime || !selectedRuntime.platforms.includes(platform))) {
      const compatible = library.runtimes.find(runtime => runtime.platforms.includes(platform));
      if (compatible) selectRuntime(compatible);
    }
  }

  async function refresh(interactive = false) {
    if (refreshing || !visible()) return;
    refreshing = true;
    clearTimeout(timer);
    showState('loading', 'Checking runtime readiness', 'Inspecting the selected compute pool.');
    try {
      const [clusterState, libraryState] = await Promise.all([
        api('/api/v1/clusters', {promptForAdmin: false}),
        api(base, {promptForAdmin: false}),
      ]);
      library = libraryState;
      selectedCluster = resolveSelectedCluster(clusterState);
      if (!clusters.length) {
        const hasNativeHost = library.hosts.length > 0;
        showState(
          'empty',
          hasNativeHost ? 'Native host ready' : 'No compute pool configured',
          hasNativeHost
            ? 'Choose a verified native package below for the registered Mac.'
            : 'Add a compute pool or register a trusted macOS or Windows host below.',
          '<a class="primary runtime-state-action" href="#setup">Open Compute pools</a>',
        );
        revealPackageLibrary(true);
        return;
      }
      if (!selectedCluster) {
        showState('empty', 'Choose one compute pool', 'Managed runtime readiness needs one pool. Native host packages remain available below.');
        revealPackageLibrary(Boolean(library.hosts.length));
        return;
      }
      engines = await api(`${management}/engines?cluster_id=${encodeURIComponent(selectedCluster.id)}`, {promptForAdmin: interactive});
      const managed = engines.engines.filter(engine => engine.capability === 'managed_launch');
      const results = await Promise.allSettled(managed.map(engine => api(`${management}/guide?cluster_id=${encodeURIComponent(selectedCluster.id)}&engine_id=${encodeURIComponent(engine.id)}`, {promptForAdmin: false})));
      guides = Object.fromEntries(results.flatMap((result, index) => result.status === 'fulfilled' ? [[managed[index].id, result.value]] : []));
      renderEngines();
      renderLibrary();
    } catch (error) {
      if (error.status === 401 || error.message.includes('authentication')) {
        showState('locked', 'Administrator access required', 'Runtime images and managed node details are protected.', '<button class="primary runtime-state-action" type="button" data-runtime-unlock>Administrator access</button>');
      } else {
        showState('error', 'Controller unavailable', error.message, '<button class="secondary runtime-state-action" type="button" data-runtime-retry>Retry</button>');
      }
    } finally {
      refreshing = false;
      if (visible() && (activeEngineJobs() || activeLibraryJobs())) timer = setTimeout(() => refresh(), 1500);
    }
  }

  document.getElementById('runtime-readiness-state').addEventListener('click', event => {
    if (event.target.closest('[data-runtime-unlock]')) void refresh(true);
    if (event.target.closest('[data-runtime-retry]')) void refresh(false);
  });
  document.getElementById('runtime-engine-list').addEventListener('click', async event => {
    if (event.target.closest('[data-runtime-revalidate]')) {
      await refresh(true);
      return;
    }
    const guided = event.target.closest('[data-guide-engine]');
    const retry = event.target.closest('[data-retry-workflow]');
    if (retry) {
      retry.disabled = true;
      guidedWorkflow = await api(`${management}/workflows/${encodeURIComponent(retry.dataset.retryWorkflow)}/retry`, {method: 'POST'});
      renderEngines();
      timer = setTimeout(pollGuidedWorkflow, 500);
      return;
    }
    if (guided && selectedCluster) {
      const guide = guides[guided.dataset.guideEngine];
      const detail = `${guide.engine_name} ${guide.version} will be downloaded, verified, installed and activated on ${guide.cluster_name}.`;
      if (!window.confirm(`${detail}\n\nContinue?`)) return;
      guided.disabled = true;
      try {
        guidedWorkflow = await api(`${management}/workflows`, {method: 'POST', body: JSON.stringify({cluster_id: selectedCluster.id, engine_id: guide.engine_id})});
        renderEngines();
        timer = setTimeout(pollGuidedWorkflow, 500);
      } catch (error) {
        guide.summary = error.message;
        renderEngines();
      }
      return;
    }
    const button = event.target.closest('[data-prepare-engine]');
    if (!button || !selectedCluster) return;
    button.disabled = true;
    button.textContent = 'Starting…';
    try {
      await api(`${management}/prepare`, {method: 'POST', body: JSON.stringify({cluster_id: selectedCluster.id, engine_id: button.dataset.prepareEngine})});
      await refresh();
    } catch (error) {
      const engine = engines.engines.find(item => item.id === button.dataset.prepareEngine);
      if (engine) { engine.message = error.message; engine.state = 'partial'; }
      renderEngines();
    }
  });

  async function pollGuidedWorkflow() {
    if (!guidedWorkflow || !visible()) return;
    try {
      guidedWorkflow = await api(`${management}/workflows/${encodeURIComponent(guidedWorkflow.id)}`, {promptForAdmin: false});
      renderEngines();
      if (['queued', 'running'].includes(guidedWorkflow.status)) {
        timer = setTimeout(pollGuidedWorkflow, 1000);
      } else if (guidedWorkflow.status === 'completed') {
        await refresh();
      }
    } catch (error) {
      const guide = guides[guidedWorkflow.engine_id];
      if (guide) guide.summary = error.message;
      renderEngines();
    }
  }
  field('list').addEventListener('click', event => {
    const button = event.target.closest('[data-runtime]');
    if (button) selectRuntime(library.runtimes.find(item => item.id === button.dataset.runtime));
  });
  field('more').addEventListener('click', () => { if (cursor) void versions(cursor); });
  field('target').addEventListener('change', () => {
    field('version').value = '';
    field('versions').replaceChildren();
    cursor = null;
    field('more').hidden = true;
    renderLibrary();
    void versions();
  });
  field('download').addEventListener('submit', async event => {
    event.preventDefault();
    const platform = targetPlatform();
    if (!selectedRuntime || !platform) return;
    field('download-button').disabled = true;
    try {
      await api(`${base}/downloads`, {method: 'POST', body: JSON.stringify({runtime_id: selectedRuntime.id, version: field('version').value.trim(), platform})});
      await refresh();
    } catch (error) { field('page').textContent = error.message; updatePackageForm(); }
  });
  field('artifacts').addEventListener('click', async event => {
    const install = event.target.closest('[data-install]');
    const remove = event.target.closest('[data-remove-artifact]');
    if (install) {
      const target = packageTarget();
      if (!target) return;
      install.disabled = true;
      const request = target.startsWith('host:')
        ? {artifact_id: install.dataset.install, host_id: target.slice(5)}
        : {artifact_id: install.dataset.install, cluster_id: target.slice(5)};
      try {
        await api(`${base}/installs`, {method: 'POST', body: JSON.stringify(request)});
        await refresh();
      } catch (error) { field('page').textContent = error.message; install.disabled = false; }
    }
    if (remove && window.confirm('Remove this verified archive from the controller cache? Installed images on machines are unchanged.')) {
      remove.disabled = true;
      try {
        await api(`${base}/artifacts/${encodeURIComponent(remove.dataset.removeArtifact)}`, {method: 'DELETE'});
        await refresh();
      } catch (error) { field('page').textContent = error.message; remove.disabled = false; }
    }
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
    } catch (error) { field('page').textContent = error.message; }
    finally { button.disabled = false; }
  });
  document.getElementById('runtime-host-list').addEventListener('click', async event => {
    const button = event.target.closest('[data-remove-host]');
    if (!button) return;
    button.disabled = true;
    try {
      await api(`${base}/hosts/${encodeURIComponent(button.dataset.removeHost)}`, {method: 'DELETE'});
      await refresh();
    } catch (error) { field('page').textContent = error.message; button.disabled = false; }
  });

  document.querySelector('.runtime-workspace-tabs').addEventListener('click', event => {
    const button = event.target.closest('[data-runtime-tab]');
    if (button) showRuntimeTab(button.dataset.runtimeTab);
  });
  ['runtime-engine-search', 'runtime-engine-filter'].forEach(id => {
    document.getElementById(id).addEventListener(id.endsWith('search') ? 'input' : 'change', renderEngines);
  });

  const checkVisibility = () => { clearTimeout(timer); if (visible()) void refresh(); };
  window.addEventListener('hashchange', checkVisibility);
  document.addEventListener('visibilitychange', checkVisibility);
  document.addEventListener('cluster-context-changed', checkVisibility);
  checkVisibility();
})();
