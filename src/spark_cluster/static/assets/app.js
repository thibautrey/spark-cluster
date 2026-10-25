const $ = (id) => document.getElementById(id);
const throughputHistory = [];
let currentFit = null;
let fitRequestVersion = 0;
let localModels = [];
let localModelQuery = '';
let launchFit = null;
let launchPerformance = null;
let launchFitRequest = 0;
let logEntries = [];
const downloadSamples = new Map();
let logsPaused = false;
let pendingModelAction = null;
let actionSheetOpener = null;
let actionSheetCloseTimer = null;
let runtimeReadiness = null;
let runtimeInstallJob = null;
let runtimeInstallTimer = null;
let pendingLaunchPayload = null;
let runtimeTerminalSocket = null;
let hfTokenStatus = null;
let hfTokenSaving = false;
let hfTokenSheetOpener = null;
let hfTokenSheetCloseTimer = null;
let latestDownloads = [];
let clusterState = {clusters: [], active_cluster_id: null};
let latestSnapshot = null;
let runtimeReady = false;
let resourceHistoryOpener = null;
let resourceHistoryCloseTimer = null;
let resourceHistoryRequest = 0;

const formatNumber = (value) => value >= 1e9 ? `${(value / 1e9).toFixed(1)}B` : value >= 1e6 ? `${(value / 1e6).toFixed(1)}M` : value >= 1e3 ? `${(value / 1e3).toFixed(1)}k` : `${Math.round(value)}`;
const formatBytes = (value) => value >= 2 ** 30 ? `${(value / 2 ** 30).toFixed(1)} GiB` : value >= 2 ** 20 ? `${(value / 2 ** 20).toFixed(1)} MiB` : `${Math.round(value / 1024)} KiB`;
const pct = (used, total) => total ? Math.min(100, used / total * 100) : 0;
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[character]));

const adminTokenStorageKey = 'spark-cluster.admin-token';
let adminTokenWaiter = null;
let adminTokenSheetOpener = null;
let adminTokenCloseTimer = null;
let adminTokenVerifying = false;

function storedAdminToken() {
  try { return localStorage.getItem(adminTokenStorageKey)?.trim() || ''; }
  catch { return ''; }
}

function saveAdminToken(token) {
  try { localStorage.setItem(adminTokenStorageKey, token); } catch {}
}

function clearAdminToken() {
  try { localStorage.removeItem(adminTokenStorageKey); } catch {}
}

function adminAuthHeaders() {
  const token = storedAdminToken();
  return token ? {Authorization: `Bearer ${token}`} : {};
}
window.adminAuthHeaders = adminAuthHeaders;

function openAdminTokenSheet(message = 'Enter the administration token to continue this action.') {
  clearTimeout(adminTokenCloseTimer);
  adminTokenSheetOpener ||= document.activeElement;
  $('admin-token-description').textContent = message;
  $('admin-token-error').hidden = true;
  $('admin-token-error').textContent = '';
  $('admin-token-value').value = '';
  $('admin-token-submit').disabled = false;
  $('admin-token-submit').textContent = 'Continue';
  $('admin-token-sheet').hidden = false;
  document.body.classList.add('bottom-sheet-open');
  requestAnimationFrame(() => {
    $('admin-token-sheet').classList.add('open');
    $('admin-token-value').focus({preventScroll: true});
  });
}

function closeAdminTokenSheet() {
  if ($('admin-token-sheet').hidden) return;
  $('admin-token-sheet').classList.remove('open');
  document.body.classList.remove('bottom-sheet-open');
  const opener = adminTokenSheetOpener;
  adminTokenSheetOpener = null;
  clearTimeout(adminTokenCloseTimer);
  adminTokenCloseTimer = setTimeout(() => {
    $('admin-token-sheet').hidden = true;
    if (opener?.isConnected) opener.focus({preventScroll: true});
  }, 220);
}

function requestAdminToken(message) {
  if (!adminTokenWaiter) {
    adminTokenWaiter = {};
    adminTokenWaiter.promise = new Promise((resolve, reject) => {
      adminTokenWaiter.resolve = resolve;
      adminTokenWaiter.reject = reject;
    });
    openAdminTokenSheet(message);
  }
  return adminTokenWaiter.promise;
}

function rejectAdminCandidate() {
  clearAdminToken();
  adminTokenVerifying = false;
  $('admin-token-error').textContent = 'The controller rejected this token. Check it and try again.';
  $('admin-token-error').hidden = false;
  $('admin-token-submit').disabled = false;
  $('admin-token-submit').textContent = 'Continue';
  $('admin-token-value').select();
}

function acceptAdminCandidate(token) {
  saveAdminToken(token);
  adminTokenVerifying = false;
  closeAdminTokenSheet();
}

async function api(path, options = {}) {
  const {promptForAdmin = true, ...fetchOptions} = options;
  let candidate = fetchOptions.headers?.Authorization?.replace(/^Bearer\s+/i, '').trim()
    || storedAdminToken();
  while (true) {
    const headers = {'Content-Type': 'application/json', ...(fetchOptions.headers || {})};
    if (candidate) headers.Authorization = `Bearer ${candidate}`;
    const response = await fetch(path, {...fetchOptions, headers});
    const body = await response.json().catch(() => ({}));
    if (response.status === 401) {
      if (candidate && candidate === storedAdminToken()) clearAdminToken();
      if (!promptForAdmin) {
        const error = new Error('Administrator authentication is required.');
        error.payload = body;
        error.status = response.status;
        throw error;
      }
      if (adminTokenVerifying) rejectAdminCandidate();
      candidate = await requestAdminToken(
        candidate
          ? 'The saved token was rejected. Enter the current administration token.'
          : 'Enter the administration token to continue this action.'
      );
      continue;
    }
    if (!response.ok) {
      const detail = typeof body.detail === 'string' ? body.detail : Array.isArray(body.detail) ? body.detail.map(item => item.msg).join('; ') : body.detail?.message || `Request failed (${response.status})`;
      const error = new Error(detail);
      error.payload = body;
      error.status = response.status;
      throw error;
    }
    if (candidate && (adminTokenVerifying || candidate !== storedAdminToken())) {
      acceptAdminCandidate(candidate);
    }
    return body;
  }
}


function clusterAuthHeaders() {
  return adminAuthHeaders();
}


const clusterContextStorageKey = 'spark-cluster.cluster-context';
function storedClusterContext() {
  try { return localStorage.getItem(clusterContextStorageKey) || 'all'; }
  catch { return 'all'; }
}
function saveClusterContext(value) {
  try { localStorage.setItem(clusterContextStorageKey, value); } catch {}
}
function selectedClusterContext() {
  return $('cluster-select')?.value || 'all';
}
window.selectedClusterContext = selectedClusterContext;

function renderClusters(state) {
  clusterState = state;
  const select = $('cluster-select');
  $('cluster-add-local').hidden = state.clusters.some(cluster => cluster.kind === 'local-docker');
  const storedContext = storedClusterContext();
  const context = storedContext === 'all' || state.clusters.some(cluster => cluster.id === storedContext)
    ? storedContext : 'all';
  select.innerHTML = state.clusters.length
    ? `<option value="all">All compute pools</option>` + state.clusters.map(cluster => `<option value="${escapeHtml(cluster.id)}">${escapeHtml(cluster.name)} · ${cluster.kind === 'local-docker' ? `${cluster.gpu_count} GPU${cluster.gpu_count === 1 ? '' : 's'}` : 'remote'}</option>`).join('')
    : '<option value="all">All compute pools</option>';
  select.value = context;
  $('cluster-list').innerHTML = state.clusters.length ? state.clusters.map(cluster => {
    const capacity = cluster.gpu_count
      ? `${cluster.gpu_count} × ${escapeHtml(cluster.gpu_model || 'NVIDIA GPU')}${cluster.gpu_memory_gib ? ` · ${cluster.gpu_memory_gib} GiB each` : ''}`
      : escapeHtml(cluster.endpoint || 'Remote compute pool');
    const issue = cluster.status === 'ready' ? '' : ` · ${escapeHtml(cluster.status)}`;
    const activeActions = `<div class="cluster-card-actions"><span class="pill ready">Active</span>${cluster.kind === 'local-docker' ? '<button class="secondary cluster-refresh" type="button" data-refresh-local aria-label="Refresh local GPUs" title="Refresh local GPUs">↻</button>' : ''}</div>`;
    return `<article class="cluster-card ${cluster.active ? 'active' : ''}" data-cluster-id="${escapeHtml(cluster.id)}"><div class="cluster-card-head"><span class="cluster-card-icon">${cluster.kind === 'local-docker' ? 'GPU' : 'LAN'}</span><div><b>${escapeHtml(cluster.name)}</b><small>${capacity}${issue}</small></div></div>${cluster.active ? activeActions : `<button class="secondary" type="button" data-select-cluster="${escapeHtml(cluster.id)}">Use this pool</button>`}</article>`;
  }).join('') : '<div class="empty-state"><b>No compute pool configured</b><span>Add a compute pool to begin.</span></div>';
  const active = state.clusters.find(cluster => cluster.id === state.active_cluster_id);
  const contextual = context === 'all' ? null : state.clusters.find(cluster => cluster.id === context);
  updateMemoryLanguage(contextual || active);
  if (context === 'all') {
    $('cluster-fleet-status').textContent = 'All compute pools are available to the model scheduler.';
  } else if (contextual) {
    $('cluster-fleet-status').textContent = `${contextual.name} is required for model capacity and manual loads.`;
  }
  if (active) {
    if (latestSnapshot) {
      renderTopology(latestSnapshot.nodes);
      renderMemory(latestSnapshot.nodes);
    }
  }
}

function updateMemoryLanguage(active) {
  const label = active?.hardware === 'dgx-spark' ? 'Unified memory' : active?.hardware === 'nvidia-gpu' ? 'GPU memory' : 'Memory';
  dashboardRoutes.memory.title = label;
  $('memory-metric-label').textContent = label;
  $('memory-bar').parentElement.setAttribute('aria-label', `${label} allocated`);
  if (selectedDashboardRoute() === 'memory') showDashboardRoute({resetScroll: false});
}

async function loadClusters() {
  const state = await api('/api/v1/clusters');
  renderClusters(state);
  return state;
}

async function selectCluster(clusterId) {
  if (!clusterId) return;
  const previous = selectedClusterContext();
  saveClusterContext(clusterId);
  if (clusterId === 'all') {
    $('cluster-fleet-status').textContent = 'All compute pools are available to the model scheduler.';
    document.dispatchEvent(new CustomEvent('cluster-context-changed', {detail: {clusterId}}));
    return;
  }
  if (clusterId === clusterState.active_cluster_id) {
    document.dispatchEvent(new CustomEvent('cluster-context-changed', {detail: {clusterId}}));
    return;
  }
  try {
    await api(`/api/v1/clusters/${encodeURIComponent(clusterId)}/select`, {method:'POST', headers:clusterAuthHeaders()});
    location.reload();
  } catch (error) {
    saveClusterContext(previous);
    $('cluster-select').value = previous;
    $('cluster-fleet-status').textContent = error.message;
  }
}

function render(snapshot) {
  latestSnapshot = snapshot;
  renderOverviewMetrics(snapshot);
  renderTopology(snapshot.nodes);
  renderCache(snapshot.cache_tiers);
  renderMemory(snapshot.nodes);
  window.renderTrackedInference?.();
}

function renderOverviewMetrics(snapshot) {
  renderOverviewDetails(snapshot);
  const stale = Date.now() - Date.parse(snapshot.timestamp) > 15000;
  const memoryUsed = snapshot.nodes.reduce((sum, node) => sum + node.memory.used_gib, 0);
  const memoryTotal = snapshot.nodes.reduce((sum, node) => sum + node.memory.total_gib, 0);
  $('stream-status').title = `Updated ${new Date(snapshot.timestamp).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'})}`;
  $('output-tps').textContent = runtimeReady ? Math.round(snapshot.total_output_tokens_per_second) : 'Idle';
  $('output-unit').textContent = runtimeReady ? ' tok/s' : '';
  $('hit-rate').textContent = snapshot.cache_hit_rate.toFixed(1);
  $('hit-ring').style.background = `conic-gradient(var(--green) ${snapshot.cache_hit_rate}%,#202a38 0)`;
  $('memory-used').textContent = stale ? '—' : memoryUsed.toFixed(1);
  $('memory-total').textContent = ` / ${Math.round(memoryTotal)} GiB`;
  const memoryPercent = pct(memoryUsed, memoryTotal);
  $('memory-total').title = `${memoryPercent.toFixed(0)}% allocated`;
  $('memory-bar').style.width = `${memoryPercent}%`;
  $('memory-bar').parentElement.setAttribute('aria-valuenow', memoryPercent.toFixed(0));
  $('prefix-reuse').textContent = formatNumber(snapshot.prefix_reuse_tokens);
  if (runtimeReady) {
    throughputHistory.push(snapshot.total_output_tokens_per_second);
    if (throughputHistory.length > 40) throughputHistory.shift();
    drawSparkline(throughputHistory);
  } else {
    drawSparkline([]);
  }
}

// Pure client-side summaries: no additional polling, storage, or server aggregation.
function renderOverviewDetails(snapshot) {
  const nodes = snapshot.nodes;
  const stale = Date.now() - Date.parse(snapshot.timestamp) > 15000;
  const reporting = stale ? [] : nodes.filter(node => node.status !== 'offline');
  const healthy = reporting.filter(node => node.status === 'healthy').length;
  $('overview-health').textContent = `${reporting.length} / ${nodes.length}`;
  $('overview-health-note').textContent = `${healthy} healthy · ${reporting.length - healthy} degraded · ${nodes.length - reporting.length} offline`;
  $('overview-gpu').textContent = reporting.length ? `${(reporting.reduce((sum, node) => sum + node.gpu_utilization, 0) / reporting.length).toFixed(0)}%` : '—';
  $('overview-free').textContent = reporting.length ? `${reporting.reduce((sum, node) => sum + Math.max(0, node.memory.total_gib - node.memory.used_gib), 0).toFixed(1)} GiB` : '—';
  const hottest = reporting.filter(node => node.temperature_c > 0).reduce((best, node) => !best || node.temperature_c > best.temperature_c ? node : best, null);
  $('overview-temperature').textContent = hottest ? `${hottest.temperature_c.toFixed(0)}°C` : '—';
  $('overview-temperature-node').textContent = hottest ? hottest.hostname : 'Temperature unavailable';
  $('overview-nodes').innerHTML = nodes.length ? nodes.map(node => {
    const online = !stale && node.status !== 'offline';
    const models = online ? (node.loaded_model_ids || []) : [];
    const modelLabel = models.length ? `<small class="overview-node-model" title="${escapeHtml(models.join(', '))}">Model · ${escapeHtml(models.join(', '))}</small>` : '';
    const historyButton = (metric, label, value) => `<button class="resource-metric-trigger" type="button" data-resource-node="${escapeHtml(node.id)}" data-resource-metric="${metric}" aria-label="View ${label} history for ${escapeHtml(node.hostname)}">${value}<span aria-hidden="true">↗</span></button>`;
    return `<tr><th scope="row"><span class="overview-node-name">${escapeHtml(node.hostname)}${modelLabel}</span></th><td><span class="overview-node-status ${escapeHtml(node.status)}">${stale ? 'stale' : escapeHtml(node.status)}</span></td><td>${online ? historyButton('gpu', 'GPU load', `${node.gpu_utilization.toFixed(0)}%`) : '—'}</td><td>${online ? historyButton('memory', 'memory usage', `${node.memory.used_gib.toFixed(1)} / ${node.memory.total_gib.toFixed(0)} GiB`) : '—'}</td><td>${online && node.temperature_c > 0 ? historyButton('temperature', 'temperature', `${node.temperature_c.toFixed(0)}°C`) : '—'}</td></tr>`;
  }).join('') : '<tr><td colspan="5">No nodes in this pool yet. Add a compute pool to get started.</td></tr>';
  $('overview-updated').textContent = `Snapshot ${new Date(snapshot.timestamp).toLocaleString()}${stale ? " · stale; current resources unknown" : " · measured resource sample"}`;
}

const resourceHistoryMetrics = {
  gpu: {
    title: 'GPU load',
    value: node => node.gpu_utilization,
    max: () => 100,
    format: value => `${value.toFixed(0)}%`,
  },
  memory: {
    title: 'Memory usage',
    value: node => node.memory.used_gib,
    max: points => Math.max(...points.map(point => point.node.memory.total_gib), 1),
    format: value => `${value.toFixed(1)} GiB`,
  },
  temperature: {
    title: 'Temperature',
    value: node => node.temperature_c,
    max: points => Math.max(100, ...points.map(point => point.node.temperature_c)),
    format: value => `${value.toFixed(0)}°C`,
  },
};

function resourceHistoryPoints(history, nodeId, metric) {
  const configuration = resourceHistoryMetrics[metric];
  return history.flatMap(snapshot => {
    const node = snapshot.nodes?.find(candidate => candidate.id === nodeId);
    if (!node) return [];
    const value = Number(configuration.value(node));
    const timestamp = Date.parse(snapshot.timestamp);
    return Number.isFinite(value) && Number.isFinite(timestamp) ? [{timestamp, value, node}] : [];
  });
}

function drawResourceHistory(points, metric) {
  const configuration = resourceHistoryMetrics[metric];
  const svg = $('resource-history-chart');
  const width = 920;
  const height = 320;
  const padding = {top: 24, right: 24, bottom: 30, left: 64};
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const start = points[0].timestamp;
  const duration = Math.max(points.at(-1).timestamp - start, 1);
  const ceiling = configuration.max(points);
  const x = point => padding.left + ((point.timestamp - start) / duration) * plotWidth;
  const y = point => padding.top + plotHeight - Math.min(1, Math.max(0, point.value / ceiling)) * plotHeight;
  const path = points.map((point, index) => `${index ? 'L' : 'M'} ${x(point).toFixed(2)} ${y(point).toFixed(2)}`).join(' ');
  const area = `${path} L ${x(points.at(-1)).toFixed(2)} ${padding.top + plotHeight} L ${x(points[0]).toFixed(2)} ${padding.top + plotHeight} Z`;
  const grid = [0, .25, .5, .75, 1].map(ratio => {
    const lineY = padding.top + plotHeight * (1 - ratio);
    return `<line x1="${padding.left}" y1="${lineY}" x2="${width - padding.right}" y2="${lineY}"/><text x="${padding.left - 12}" y="${lineY + 4}">${escapeHtml(configuration.format(ceiling * ratio))}</text>`;
  }).join('');
  const capacity = metric === 'memory' ? `<line class="resource-chart-capacity" x1="${padding.left}" y1="${padding.top}" x2="${width - padding.right}" y2="${padding.top}"/>` : '';
  const last = points.at(-1);
  svg.innerHTML = `<title id="resource-history-chart-title">${escapeHtml(configuration.title)} history</title><desc id="resource-history-chart-description">${points.length} recorded values from ${new Date(start).toLocaleString()} to ${new Date(last.timestamp).toLocaleString()}.</desc><g class="resource-chart-grid">${grid}</g>${capacity}<path class="resource-chart-area" d="${area}"/><path class="resource-chart-line" d="${path}"/><circle class="resource-chart-point" cx="${x(last)}" cy="${y(last)}" r="5"/>`;
}

function renderResourceHistory(history, nodeId, metric) {
  const configuration = resourceHistoryMetrics[metric];
  const points = resourceHistoryPoints(history, nodeId, metric);
  $('resource-history-loading').hidden = true;
  if (!points.length) {
    $('resource-history-empty').hidden = false;
    return;
  }
  const values = points.map(point => point.value);
  const latest = points.at(-1);
  const durationMinutes = Math.max(1, Math.round((latest.timestamp - points[0].timestamp) / 60000));
  $('resource-history-window').textContent = `${durationMinutes} min recorded`;
  $('resource-history-current').textContent = configuration.format(latest.value);
  $('resource-history-updated').textContent = `Updated ${new Date(latest.timestamp).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'})}`;
  $('resource-history-min').textContent = configuration.format(Math.min(...values));
  $('resource-history-average').textContent = configuration.format(values.reduce((sum, value) => sum + value, 0) / values.length);
  $('resource-history-max').textContent = configuration.format(Math.max(...values));
  $('resource-history-start').textContent = new Date(points[0].timestamp).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
  $('resource-history-end').textContent = new Date(latest.timestamp).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
  $('resource-history-samples').textContent = `${points.length} sample${points.length === 1 ? '' : 's'}`;
  $('resource-history-data').hidden = false;
  drawResourceHistory(points, metric);
}

async function openResourceHistory(nodeId, metric, opener) {
  const node = latestSnapshot?.nodes.find(candidate => candidate.id === nodeId);
  const configuration = resourceHistoryMetrics[metric];
  if (!node || !configuration) return;
  const request = ++resourceHistoryRequest;
  resourceHistoryOpener = opener;
  clearTimeout(resourceHistoryCloseTimer);
  $('resource-history-node').textContent = node.hostname;
  $('resource-history-title').textContent = configuration.title;
  $('resource-history-window').textContent = 'Last 30 minutes';
  $('resource-history-loading').hidden = false;
  $('resource-history-data').hidden = true;
  $('resource-history-empty').hidden = true;
  $('resource-history-error').hidden = true;
  $('resource-history-sheet').hidden = false;
  document.body.classList.add('bottom-sheet-open');
  requestAnimationFrame(() => {
    $('resource-history-sheet').classList.add('open');
    $('resource-history-close').focus({preventScroll: true});
  });
  try {
    const history = await api('/api/v1/history?minutes=30', {promptForAdmin: false});
    if (request === resourceHistoryRequest) renderResourceHistory(history, nodeId, metric);
  } catch (error) {
    if (request !== resourceHistoryRequest) return;
    $('resource-history-loading').hidden = true;
    $('resource-history-error').textContent = `Unable to load resource history: ${error.message}`;
    $('resource-history-error').hidden = false;
  }
}

function closeResourceHistory() {
  if ($('resource-history-sheet').hidden) return;
  resourceHistoryRequest += 1;
  $('resource-history-sheet').classList.remove('open');
  document.body.classList.remove('bottom-sheet-open');
  const opener = resourceHistoryOpener;
  resourceHistoryOpener = null;
  clearTimeout(resourceHistoryCloseTimer);
  resourceHistoryCloseTimer = setTimeout(() => {
    $('resource-history-sheet').hidden = true;
    if (opener?.isConnected) opener.focus({preventScroll: true});
  }, 220);
}

function renderTopology(nodes) {
  const active = clusterState.clusters.find(cluster => cluster.id === clusterState.active_cluster_id);
  const topology = $('topology-map');
  if (active?.kind === 'local-docker') {
    topology.classList.add('local-host-topology');
    const gpus = active.gpus?.length ? active.gpus : nodes.map((node, index) => ({name: active.gpu_model || node.hostname, memory_gib: node.memory.total_gib, index}));
    const gpuCount = active.gpu_count || gpus.length;
    const gpuCards = gpus.map((gpu, index) => {
      const node = nodes[index];
      return `<article class="topology-gpu ${escapeHtml(node?.status || 'offline')}">
        <div class="topology-gpu-head"><span class="node-icon">${index}</span><div><b>GPU ${index}</b><small>${escapeHtml(gpu.name || active.gpu_model || 'NVIDIA GPU')}</small></div><i class="node-state ${escapeHtml(node?.status || 'offline')}" title="${escapeHtml(node?.status || 'offline')}"></i></div>
        <div class="node-stats"><span>LOAD<b>${(node?.gpu_utilization || 0).toFixed(0)}%</b></span><span>MEM<b>${(node?.memory.used_gib || 0).toFixed(0)} / ${(node?.memory.total_gib || gpu.memory_gib || 0).toFixed(0)}</b></span><span>OUTPUT<b>${(node?.output_tokens_per_second || 0).toFixed(0)} t/s</b></span><span>TEMP<b>${(node?.temperature_c || 0).toFixed(0)}°C</b></span></div>
      </article>`;
    }).join('');
    topology.innerHTML = `<article class="topology-host"><div class="topology-host-head"><span class="cluster-card-icon">HOST</span><div><b>${escapeHtml(active.name)}</b><small>${escapeHtml(active.host_address || 'local Docker host')} · ${gpuCount} GPU${gpuCount === 1 ? '' : 's'}</small></div></div><div class="topology-gpus">${gpuCards}</div></article>`;
    $('node-count').textContent = `1 host · ${gpuCount} GPU${gpuCount === 1 ? '' : 's'}`;
  } else {
    topology.classList.remove('local-host-topology');
    topology.innerHTML = '<div class="fabric-line"></div>' + nodes.map(node => `
    <div class="node ${escapeHtml(node.status)}">
      <div class="node-top"><span class="node-icon">${node.role === 'head' ? 'H' : 'W'}</span><i class="node-state ${escapeHtml(node.status)}" title="${escapeHtml(node.status)}"></i></div>
      <h3>${escapeHtml(node.hostname)}</h3><p>${escapeHtml(node.role.toUpperCase())} · ${escapeHtml(node.address)}</p>
      <div class="node-stats"><span>GPU<b>${node.gpu_utilization.toFixed(0)}%</b></span><span>MEM<b>${node.memory.used_gib.toFixed(0)} / ${node.memory.total_gib.toFixed(0)}</b></span><span>OUTPUT<b>${node.output_tokens_per_second.toFixed(0)} t/s</b></span><span>TEMP<b>${node.temperature_c.toFixed(0)}°C</b></span></div>
    </div>`).join('');
    $('node-count').textContent = `${nodes.length} node${nodes.length === 1 ? '' : 's'}`;
  }
  const states = [['healthy', 'green', 'Healthy'], ['degraded', 'amber', 'Degraded'], ['offline', 'muted', 'Offline']];
  $('topology-statuses').innerHTML = states.filter(([status]) => nodes.some(node => node.status === status)).map(([, className, label]) => `<span><i class="${className}"></i>${label}</span>`).join('');
}

function renderCache(tiers) {
  const cacheTiers = tiers.filter(tier => !tier.name.toLowerCase().includes('model storage'));
  if (!cacheTiers.length) {
    $('cache-tiers').innerHTML = '<div class="cache-empty"><div><b>KV cache metrics unavailable</b><span>Metrics appear after a runtime starts.</span></div><a href="#launch">Launch a model</a></div>';
    return;
  }
  $('cache-tiers').innerHTML = cacheTiers.map(tier => {
    const total = tier.hits + tier.misses;
    const hitRate = total ? tier.hits / total * 100 : 0;
    return `<div class="cache-row">
      <div class="cache-row-head"><b>${escapeHtml(tier.name)}</b><span>${tier.used_gib.toFixed(0)} / ${tier.capacity_gib.toFixed(0)} GiB</span></div>
      <div class="cache-track"><i style="width:${pct(tier.used_gib, tier.capacity_gib)}%"></i></div>
      <div class="cache-meta"><span>${hitRate.toFixed(1)}% hit</span><span>${tier.read_latency_ms.toFixed(1)} ms</span></div>
    </div>`;
  }).join('');
}

function renderMemory(nodes) {
  const active = clusterState.clusters.find(cluster => cluster.id === clusterState.active_cluster_id);
  const unified = active?.hardware === 'dgx-spark';
  const segment = (className, label, value) => value > 0
    ? `<i class="${className}" style="--share:${value}" title="${escapeHtml(label)} ${value.toFixed(1)} GiB"></i>`
    : '';
  $('memory-grid').innerHTML = nodes.map((node, index) => {
    const kvKnown = node.memory.kv_cache_source !== 'unavailable';
    const total = Math.max(0, node.memory.total_gib);
    const model = Math.min(total, Math.max(0, node.memory.model_gib));
    const kv = kvKnown ? Math.min(Math.max(0, total - model), Math.max(0, node.memory.kv_cache_gib)) : 0;
    const other = Math.min(Math.max(0, total - model - kv), Math.max(0, node.memory.used_gib - model - kv));
    const free = Math.max(0, total - model - kv - other);
    const nodeLabel = active?.kind === 'local-docker' ? `GPU ${active.gpus?.[index]?.index ?? index}` : node.hostname;
    return `<div><div class="memory-node-head"><b>${escapeHtml(nodeLabel)}</b><span>${node.memory.used_gib.toFixed(1)} / ${total.toFixed(1)} GiB</span></div><div class="memory-block" role="img" aria-label="${escapeHtml(nodeLabel)}: ${node.memory.used_gib.toFixed(1)} of ${total.toFixed(1)} GiB used">${segment('model', 'Model', model)}${segment('kv', 'KV cache', kv)}${segment('runtime', unified ? 'Runtime' : 'Used', other)}${segment('free', 'Free', free)}</div></div>`;
  }).join('');
  const hasModels = nodes.some(node => node.memory.model_gib > 0);
  const hasKv = nodes.some(node => node.memory.kv_cache_source !== 'unavailable' && node.memory.kv_cache_gib > 0);
  const hasOther = nodes.some(node => node.memory.used_gib - node.memory.model_gib - node.memory.kv_cache_gib > 0);
  const hasFree = nodes.some(node => node.memory.total_gib - node.memory.used_gib > 0);
  const legend = $('memory-legend');
  legend.hidden = nodes.length === 0;
  legend.innerHTML = `${hasModels ? '<span><i class="m"></i>model</span>' : ''}${hasKv ? '<span><i class="k"></i>KV</span>' : ''}${hasOther ? `<span><i class="r"></i>${unified ? 'runtime' : 'used'}</span>` : ''}${hasFree ? '<span><i class="f"></i>free</span>' : ''}`;
}

function drawSparkline(values) {
  const canvas = $('throughput-spark');
  const context = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  if (values.length < 2) return;
  const minimum = Math.min(...values) - 4;
  const maximum = Math.max(...values) + 4;
  context.beginPath();
  values.forEach((value, index) => {
    const x = index / (values.length - 1) * width;
    const y = height - (value - minimum) / (maximum - minimum) * height;
    index ? context.lineTo(x, y) : context.moveTo(x, y);
  });
  context.strokeStyle = '#63f2b3';
  context.lineWidth = 2;
  context.stroke();
  context.lineTo(width, height);
  context.lineTo(0, height);
  context.closePath();
  const gradient = context.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, 'rgba(99,242,179,.24)');
  gradient.addColorStop(1, 'rgba(99,242,179,0)');
  context.fillStyle = gradient;
  context.fill();
}

function setStreamStatus(state, label) {
  $('stream-status').className = `live ${state}`;
  $('stream-status').lastElementChild.textContent = label;
}

function connect() {
  setStreamStatus('connecting', 'Connecting');
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/api/v1/stream`);
  socket.onopen = () => setStreamStatus('connecting', 'Waiting for sample');
  socket.onmessage = event => {
    const snapshot = JSON.parse(event.data);
    const stale = Date.now() - Date.parse(snapshot.timestamp) > 15000;
    setStreamStatus(stale ? 'reconnecting' : 'ready', stale ? 'Stale telemetry' : 'Live');
    render(snapshot);
  };
  socket.onclose = () => {
    setStreamStatus('reconnecting', 'Reconnecting');
    setTimeout(connect, 1500);
  };
}

function renderEndpoint(endpoint) {
  runtimeReady = endpoint.status === 'ready';
  $('overview').classList.toggle('runtime-idle', !runtimeReady);
  $('output-metric-label').textContent = runtimeReady ? 'Cluster output' : 'Inference';
  $('cache-hit-metric').hidden = !runtimeReady;
  $('prefix-reuse-metric').hidden = !runtimeReady;
  if (latestSnapshot) renderOverviewMetrics(latestSnapshot);
  window.renderTrackedInference?.();
}

function renderPoolEndpoint(pool) {
  const availableClusters = pool.clusters.filter(cluster => cluster.available);
  const availableModels = pool.models.filter(model => model.loadable);
  const endpoint = `${window.location.origin}/v1`;
  $('api-endpoint').textContent = endpoint;
  $('api-models-route').textContent = `${endpoint}/models`;
  $('api-chat-route').textContent = `${endpoint}/chat/completions`;
  $('api-embeddings-route').textContent = `${endpoint}/embeddings`;
  $('api-status').textContent = availableClusters.length ? '● pool ready' : '○ pool unavailable';
  $('api-status').className = `pill ${availableClusters.length ? 'ready' : 'unreachable'}`;
  $('api-routing-mode').textContent = availableClusters.length
    ? `${availableClusters.length} cluster${availableClusters.length === 1 ? '' : 's'} · ${availableModels.length} models · automatic loading`
    : 'No reachable cluster can accept inference';
}


function localModelAvatar(model) {
  if (window.catalogAvatarForRepo) return window.catalogAvatarForRepo(model.id);
  const name = model.id.split('/').pop() || model.id;
  const monogram = name.replace(/[^A-Za-z0-9]+/g, ' ').trim().split(/\s+/)
    .slice(0, 2).map(part => part[0]).join('').toUpperCase() || 'HF';
  return `<span class="catalog-avatar"><span aria-hidden="true">${escapeHtml(monogram)}</span></span>`;
}

function renderLocalModelCards() {
  const query = localModelQuery.trim().toLocaleLowerCase();
  const models = query
    ? localModels.filter(model => model.id.toLocaleLowerCase().includes(query))
    : localModels;
  if (!localModels.length) {
    $('local-models').innerHTML = '<div class="empty-state"><b>No local model found</b><span>Add one from Hugging Face.</span></div>';
    return;
  }
  if (!models.length) {
    $('local-models').innerHTML = `<div class="empty-state"><b>No downloaded model found</b><span>Nothing matches “${escapeHtml(localModelQuery.trim())}”.</span></div>`;
    return;
  }
  $('local-models').innerHTML = models.map(model => `
    <article class="model-card local-model-card">
      <button class="local-model-open" type="button" data-model-repo="${escapeHtml(model.id)}" data-model-revision="${escapeHtml(model.revision || 'main')}" aria-label="Open details for ${escapeHtml(model.id)}">
        ${localModelAvatar(model)}
        <span><h3>${escapeHtml(model.id)}</h3><p>${formatBytes(model.size_bytes)}</p></span>
      </button>
      <div class="model-card-actions">${model.loaded ? '<span class="model-state loaded">● In use</span>' : ''}${model.source === 'managed' ? `<button class="model-delete danger" type="button" data-model-action="delete" data-repo-id="${escapeHtml(model.id)}" aria-label="Delete ${escapeHtml(model.id)}" title="${model.loaded ? 'Stop the runtime before deleting this model' : 'Delete model'}" ${model.loaded ? 'disabled' : ''}><svg aria-hidden="true" viewBox="0 0 24 24"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v5M14 11v5" /></svg></button>` : ''}</div>
    </article>`).join('');
  window.activateCatalogAvatars?.($('local-models'));
}

function renderModels(models) {
  localModels = models;
  $('local-model-count').textContent = models.length;
  updateLaunchModelOptions(models);
  renderLocalModelCards();
}

function updateLaunchModelOptions(models) {
  const select = $('launch-model');
  const selected = select.value;
  const managed = models.filter(model => model.source === 'managed');
  select.innerHTML = `<option value="">${managed.length ? 'Select a downloaded model' : 'No downloaded models'}</option>` + managed.map(model => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.id)}${model.launch_supported === false ? ' · runtime required' : ''}</option>`).join('');
  if (managed.some(model => model.id === selected)) select.value = selected;
}

const activeDownloadStatuses = new Set(['queued', 'resolving', 'downloading', 'paused', 'preparing-transfer', 'distributing']);

function downloadPhase(job) {
  if (['queued', 'resolving'].includes(job.status)) return 0;
  if (['downloading', 'paused'].includes(job.status)) return 1;
  return 2;
}

function downloadPhaseLabel(job) {
  return ({queued: 'Queued', resolving: 'Resolving repository', downloading: 'Downloading from Hugging Face', paused: 'Download paused', 'preparing-transfer': 'Preparing LAN delivery', distributing: 'Distributing to cluster'})[job.status] || job.status;
}

function downloadRate(job) {
  const bytes = job.status === 'distributing' ? job.transferred_bytes : job.downloaded_bytes;
  const now = Date.now();
  const previous = downloadSamples.get(job.id);
  let rate = previous?.rate || 0;
  if (previous && bytes >= previous.bytes && now > previous.time) {
    const instant = (bytes - previous.bytes) / ((now - previous.time) / 1000);
    if (instant > 0) rate = rate ? rate * 0.65 + instant * 0.35 : instant;
  }
  downloadSamples.set(job.id, {bytes, time: now, rate});
  return rate;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return 'Calculating';
  if (seconds < 60) return `${Math.ceil(seconds)} sec`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)} min`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.ceil((seconds % 3600) / 60)}m`;
  return `${Math.ceil(seconds / 86400)} days`;
}

function openHuggingFaceTokenSheet(opener) {
  clearTimeout(hfTokenSheetCloseTimer);
  hfTokenSheetOpener = opener || hfTokenSheetOpener;
  $('hf-token-sheet').hidden = false;
  document.body.classList.add('bottom-sheet-open');
  requestAnimationFrame(() => {
    $('hf-token-sheet').classList.add('open');
    $('hf-token-close').focus({preventScroll: true});
  });
}

function closeHuggingFaceTokenSheet() {
  if ($('hf-token-sheet').hidden) return;
  $('hf-token-sheet').classList.remove('open');
  document.body.classList.remove('bottom-sheet-open');
  const opener = hfTokenSheetOpener;
  hfTokenSheetCloseTimer = setTimeout(() => {
    $('hf-token-sheet').hidden = true;
    if (opener?.isConnected && !opener.hidden) opener.focus({preventScroll: true});
  }, 220);
}

function renderHuggingFacePrompt() {
  const downloading = latestDownloads.some(job => job.source === 'huggingface' && ['queued', 'resolving', 'downloading'].includes(job.status));
  const visible = downloading && hfTokenStatus && !hfTokenStatus.configured;
  $('hf-token-prompt').hidden = !visible;
  $('hf-token-form').hidden = !visible;
  if (!visible) closeHuggingFaceTokenSheet();
}

$('hf-token-prompt').addEventListener('click', event => openHuggingFaceTokenSheet(event.currentTarget));
$('hf-token-close').addEventListener('click', closeHuggingFaceTokenSheet);
$('hf-token-backdrop').addEventListener('click', closeHuggingFaceTokenSheet);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !$('hf-token-sheet').hidden) closeHuggingFaceTokenSheet();
});
window.addEventListener('hashchange', closeHuggingFaceTokenSheet);

$('hf-token-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (hfTokenSaving) return;
  hfTokenSaving = true;
  $('hf-token-save').disabled = true;
  $('hf-token-message').textContent = 'Validating token and restarting active downloads…';
  try {
    const state = await api('/api/v1/controller/huggingface', {
      method: 'PUT',
      headers: adminAuthHeaders(),
      body: JSON.stringify({token: $('hf-download-token').value.trim()}),
    });
    hfTokenStatus = {...hfTokenStatus, configured: state.configured};
    $('hf-download-token').value = '';
    $('hf-token-message').textContent = '';
    $('hf-token-saved').textContent = `Hugging Face token saved. ${state.restarted_downloads.length} active download(s) restarted from preserved files. Higher request limits apply; a speed increase is not guaranteed.`;
    $('hf-token-saved').hidden = false;
    renderHuggingFacePrompt();
    downloadSamples.clear();
    await refreshLibrary();
  } catch (error) {
    $('hf-token-message').textContent = error.message;
  } finally {
    hfTokenSaving = false;
    $('hf-token-save').disabled = false;
  }
});

function renderActiveDownloads(downloads) {
  latestDownloads = downloads;
  renderHuggingFacePrompt();
  const active = downloads.filter(job => activeDownloadStatuses.has(job.status));
  const section = $('active-downloads');
  section.hidden = active.length === 0;
  if (!active.length) {
    $('active-download-list').innerHTML = '';
    downloadSamples.clear();
    return;
  }
  $('active-download-count').textContent = `${active.length} in progress`;
  $('active-download-title').textContent = 'Downloads';
  const activeIds = new Set(active.map(job => job.id));
  for (const id of downloadSamples.keys()) if (!activeIds.has(id)) downloadSamples.delete(id);
  $('active-download-list').innerHTML = active.map(job => {
    const distributing = job.status === 'distributing';
    const paused = job.status === 'paused';
    const progress = distributing ? job.transfer_progress : job.progress;
    const currentBytes = distributing ? job.transferred_bytes : job.downloaded_bytes;
    const rate = paused ? 0 : downloadRate(job);
    const remaining = Math.max(0, job.total_bytes - currentBytes);
    const timing = paused ? '' : rate ? `<span>${formatBytes(rate)}/s</span><span>${formatDuration(remaining / rate)} remaining</span>` : '<span>Calculating speed…</span>';
    return `<article class="active-download-card ${paused ? 'paused' : ''}" data-model-repo="${escapeHtml(job.repo_id)}" data-model-revision="${escapeHtml(job.revision || 'main')}" tabindex="0" aria-label="Open details for ${escapeHtml(job.repo_id)}">
      <div class="active-download-identity"><span class="active-download-icon">↓</span><div><p>${escapeHtml(downloadPhaseLabel(job))}</p><h4>${escapeHtml(job.repo_id)}</h4></div></div>
      <div class="active-download-progress"><div><i style="width:${progress}%"></i></div><b>${progress.toFixed(2)}%</b></div>
      <div class="active-download-footer"><div class="active-download-summary"><span><b>${formatBytes(currentBytes)}</b> of ${formatBytes(job.total_bytes)}</span>${timing}</div><div class="active-download-actions"><button class="download-control" type="button" data-download-action="${paused ? 'resume' : 'pause'}" data-job-id="${escapeHtml(job.id)}" data-repo-id="${escapeHtml(job.repo_id)}">${paused ? 'Resume' : 'Pause'}</button><button class="download-control danger" type="button" data-download-action="cancel" data-job-id="${escapeHtml(job.id)}" data-repo-id="${escapeHtml(job.repo_id)}">Cancel</button></div></div>
    </article>`;
  }).join('');
}

function renderDownloads(downloads) {
  renderActiveDownloads(downloads);
  const history = downloads.filter(job => !activeDownloadStatuses.has(job.status));
  $('download-history-count').textContent = history.length;
  if (!history.length) {
    $('download-list').innerHTML = '<div class="empty-downloads">Completed and failed transfers will appear here.</div>';
    return;
  }
  $('download-list').innerHTML = history.map(job => `
    <div class="download-row">
      <div class="download-name"><b>${escapeHtml(job.repo_id)}</b><small>${job.destination === 'cluster' ? `Internet → controller → ${escapeHtml(job.target_cluster_name || 'cluster LAN')}` : job.source === 'controller' ? 'Controller LAN → shared cluster storage' : `Internet → ${escapeHtml(job.target_cluster_name || 'controller storage')}`} · attempt ${job.attempts}</small><small>${escapeHtml(job.message || job.status)}</small></div>
      <div class="download-progress"><div class="cache-track"><i style="width:${job.status === 'distributing' ? job.transfer_progress : job.progress}%"></i></div><span>${(job.status === 'distributing' ? job.transfer_progress : job.progress).toFixed(1)}%</span></div>
      <span class="download-status ${job.status}">${escapeHtml(job.status)}</span>
      ${job.status === 'failed' ? `<button class="download-retry" data-job-id="${escapeHtml(job.id)}" title="${escapeHtml(job.error || 'Retry download')}">Retry</button>` : '<span></span>'}
    </div>`).join('');
}

const launchStages = [
  ['validating', 'Validate model'],
  ['scaling-workers', 'Scale Ray workers'],
  ['starting-runtime', 'Start vLLM runtime'],
  ['waiting-ready', 'Wait for readiness'],
  ['ready', 'Serve OpenAI API'],
];

function renderLaunches(launches) {
  if (!launches.length) {
    $('launch').classList.add('idle');
    $('launch-layout').classList.add('idle');
    $('launch-status').hidden = true;
    $('launch-status').innerHTML = '';
    $('launch-history').hidden = true;
    $('launch-history').innerHTML = '';
    return;
  }
  $('launch').classList.remove('idle');
  $('launch-layout').classList.remove('idle');
  $('launch-status').hidden = false;
  $('launch-history').hidden = false;
  const current = launches.find(job => !['ready', 'failed'].includes(job.status)) || launches[0];
  const activeIndex = launchStages.findIndex(([status]) => status === current.status);
  $('launch-status').innerHTML = `
    <div class="launch-job-head"><div><h3>${escapeHtml(current.request.served_model_name)}</h3><p>${escapeHtml(current.request.model_id)} · ${escapeHtml(current.target_cluster_name || 'active cluster')} · ${escapeHtml(current.execution_plan?.engine_name || current.request.engine_id)} · TP ${current.request.tensor_parallel} × PP ${current.request.pipeline_parallel} · ${Math.round(current.request.context_length / 1024)}k</p></div><span class="launch-progress-value">${current.progress.toFixed(0)}%</span></div>
    <div class="launch-progress-bar"><i style="width:${current.progress}%"></i></div>
    <div class="launch-steps">${launchStages.map(([status, label], index) => `<div class="launch-step ${index < activeIndex || current.status === 'ready' ? 'done' : index === activeIndex ? 'active' : ''}"><i>${index < activeIndex || current.status === 'ready' ? '✓' : index + 1}</i><span>${label}</span><span>${status === current.status ? escapeHtml(current.message) : ''}</span></div>`).join('')}</div>
    ${current.error ? `<div class="launch-error">${escapeHtml(current.error)}</div>` : ''}`;
  $('launch-history').innerHTML = launches.slice(0, 8).map(job => `
    <div class="launch-history-row"><b>${escapeHtml(job.request.served_model_name)}</b><span>${new Date(job.created_at).toLocaleTimeString()}</span><span class="${job.status}">${escapeHtml(job.status)}</span><span>${job.status === 'failed' ? `<button class="launch-retry" data-launch-id="${escapeHtml(job.id)}">Retry</button>` : escapeHtml(job.message)}</span></div>`).join('');
}

function launchPayload() {
  return {
    model_id: $('launch-model').value,
    served_model_name: $('served-model-name').value.trim(),
    bits_per_weight: Number($('launch-bits').value),
    context_length: Number($('launch-context').value),
    concurrent_sequences: Number($('launch-concurrency').value),
    tensor_parallel: Number($('tensor-parallel').value),
    pipeline_parallel: Number($('pipeline-parallel').value),
    gpu_memory_utilization: Number($('gpu-memory-utilization').value),
    optimization_profile: $('launch-workload').value,
    kv_cache_dtype: launchPerformance?.kv_cache_dtype || 'auto',
    max_num_seqs: launchPerformance?.max_num_seqs || Number($('launch-concurrency').value),
    max_num_batched_tokens: launchPerformance?.max_num_batched_tokens || 8192,
    engine_id: $('launch-engine').value,
    speculative_method: $('launch-speculative').value,
    speculative_tokens: Number($('launch-speculative-tokens').value),
    expert_parallel: $('launch-expert-parallel').checked,
    all2all_backend: $('launch-all2all').value,
    expert_load_balancing: $('launch-eplb').checked,
    redundant_experts: Number($('launch-redundant-experts').value),
    cuda_graphs: $('launch-cuda-graphs').checked,
    chunked_prefill: $('launch-chunked-prefill').checked,
    cache_cpu_gib: Number($('launch-cache-cpu').value),
    cache_disk_gib: Number($('launch-cache-disk').value),
  };
}

async function loadPerformancePlan(apply = true) {
  const modelId = $('launch-model').value;
  if (!modelId) {
    launchPerformance = null;
    $('performance-plan').hidden = true;
    return;
  }
  $('performance-plan').hidden = false;
  $('performance-plan').innerHTML = '<span>…</span><div><b>Calculating performance plan</b><small>Matching the model, workload, node count, and fabric.</small></div>';
  try {
    const plan = await api('/api/v1/performance/plan', {method: 'POST', body: JSON.stringify({
      model_id: modelId,
      revision: localModels.find(model => model.id === modelId)?.revision || null,
      workload: $('launch-workload').value,
      context_length: Number($('launch-context').value),
      concurrent_sequences: Number($('launch-concurrency').value),
    })});
    launchPerformance = plan;
    if (apply) {
      const contextOption = [...$('launch-context').options].find(option => Number(option.value) === plan.context_length);
      if (contextOption) $('launch-context').value = contextOption.value;
      $('tensor-parallel').value = plan.tensor_parallel;
      $('pipeline-parallel').value = plan.pipeline_parallel;
      $('gpu-memory-utilization').value = plan.gpu_memory_utilization;
      $('launch-engine').value = plan.engine_id;
      $('launch-speculative').value = plan.speculative_method;
      $('launch-speculative-tokens').value = plan.speculative_tokens;
      $('launch-expert-parallel').checked = plan.expert_parallel;
      $('launch-eplb').checked = plan.expert_load_balancing;
      $('launch-all2all').value = plan.all2all_backend;
      $('launch-cache-cpu').value = plan.cache_cpu_gib;
      $('launch-cache-disk').value = plan.cache_disk_gib;
    }
    const evidence = plan.evidence.map(item => `<a href="${escapeHtml(item.source_url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.title)} ↗</a>`).join(' · ');
    const warnings = plan.warnings.length ? `<p>${plan.warnings.map(item => escapeHtml(item)).join(' ')}</p>` : '';
    const blockers = plan.blockers?.length ? `<p><b>Required before launch:</b> ${plan.blockers.map(item => escapeHtml(item)).join(' ')}</p>` : '';
    const recipe = plan.recipe;
    const recipeDetails = recipe ? `<p>Recipe ${escapeHtml(recipe.recipe_revision.slice(0, 12))} · checkpoint ${escapeHtml(recipe.checkpoint_revision.slice(0, 12))} · ${escapeHtml(recipe.license)}</p><ul>${recipe.prerequisites.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul><p>Merge these reviewed settings into the full upstream ${escapeHtml(recipe.env_file)} profile, preserving its production adapters and local fabric configuration.</p><pre>${escapeHtml(Object.entries(recipe.environment).map(([key, value]) => `${key}=${JSON.stringify(value)}`).join('\n'))}</pre><p>${recipe.measurements.map(item => escapeHtml(item)).join(' ')}</p><a href="${escapeHtml(recipe.recipe_url)}" target="_blank" rel="noopener noreferrer">Pinned recipe and preparation steps ↗</a>` : '';
    const detail = `${escapeHtml(plan.engine_guidance)} ${escapeHtml(plan.topology_guidance)}`;
    const details = detail || warnings || evidence ? `<details class="performance-plan-details"><summary>Plan details</summary><p>${detail}</p>${blockers}${warnings}${recipeDetails}${evidence ? `<small>${evidence}</small>` : ''}</details>` : '';
    $('performance-plan').innerHTML = `<span>${plan.engine_executable ? '↗' : '!'}</span><div><b>${escapeHtml(plan.profile)}</b><small>${escapeHtml(plan.engine)} · ${escapeHtml(plan.collective_backend)} · TP ${plan.tensor_parallel} · ${plan.max_num_seqs} sequences · ${plan.kv_cache_dtype === 'fp8_e4m3' ? 'FP8 KV' : 'automatic KV'}</small>${details}</div>`;
  } catch (error) {
    launchPerformance = null;
    $('performance-plan').innerHTML = `<span>!</span><div><b>Performance plan unavailable</b><small>${escapeHtml(error.message)}</small></div>`;
  }
}

function launchAuthHeaders() {
  return adminAuthHeaders();
}

function launchClusterId() {
  const context = selectedClusterContext();
  return context === 'all' ? clusterState.active_cluster_id : context;
}


async function checkLaunchFit() {
  const request = ++launchFitRequest;
  const button = $('launch-model-button');
  if (!$('launch-model').value) {
    launchFit = null;
    button.disabled = true;
    button.textContent = 'Launch on cluster';
    $('launch-fit').hidden = true;
    return;
  }
  launchFit = null;
  button.disabled = true;
  button.textContent = 'Checking compatibility…';
  $('launch-fit').hidden = true;
  const payload = launchPayload();
  try {
    const fit = await api('/api/v1/models/fit-local', {method: 'POST', body: JSON.stringify({repo_id: payload.model_id, bits_per_weight: payload.bits_per_weight, context_length: payload.context_length, concurrent_sequences: payload.concurrent_sequences, kv_cache_dtype: payload.kv_cache_dtype})});
    if (request !== launchFitRequest) return;
    launchFit = fit;
    const requestedNodes = payload.tensor_parallel * payload.pipeline_parallel;
    const parallelFits = requestedNodes >= launchFit.minimum_nodes && requestedNodes <= launchFit.cluster_nodes;
    const canLaunch = launchFit.fits && parallelFits && launchPerformance?.engine_executable !== false;
    $('launch-fit').className = `fit-result ${canLaunch ? 'fit' : 'no-fit'}`;
    const accelerators = count => `${count} accelerator${count === 1 ? '' : 's'}`;
    let decision = `Fits on ${accelerators(requestedNodes)}`;
    let detail = `${launchFit.total_required_gib.toFixed(1)} GiB required · ${launchFit.headroom_gib.toFixed(1)} GiB headroom`;
    if (!launchFit.runtime_storage_fits) {
      decision = 'Not enough runtime storage';
      detail = `${launchFit.runtime_storage_required_gib.toFixed(1)} GiB required for runtime artifacts`;
    } else if (!launchFit.launch_supported) {
      decision = 'Specialized runtime required';
      detail = launchFit.engine_recommendation;
    } else if (!launchFit.runtime_fits) {
      decision = `Needs ${accelerators(launchFit.minimum_nodes)}`;
      detail = `${accelerators(launchFit.cluster_nodes)} available at the selected context · ${launchFit.total_required_gib.toFixed(1)} GiB required`;
    } else if (!parallelFits) {
      decision = requestedNodes < launchFit.minimum_nodes ? `Use at least ${accelerators(launchFit.minimum_nodes)}` : `Only ${accelerators(launchFit.cluster_nodes)} available`;
      detail = `TP ${payload.tensor_parallel} × PP ${payload.pipeline_parallel} selects ${accelerators(requestedNodes)}`;
    } else if (launchPerformance?.engine_executable === false) {
      decision = 'Runtime preparation required';
      detail = launchPerformance.blockers?.join(' ') || launchPerformance.engine_guidance;
    }
    $('launch-fit').innerHTML = `<span class="fit-symbol">${canLaunch ? '✓' : '!'}</span><div><b>${escapeHtml(decision)}</b><small>${escapeHtml(detail)}</small></div>`;
    launchFit.parallel_fits = parallelFits;
    $('launch-fit').hidden = false;
    button.textContent = 'Launch on cluster';
    button.disabled = !canLaunch || !payload.served_model_name;
  } catch (error) {
    if (request !== launchFitRequest) return;
    launchFit = null;
    button.disabled = true;
    button.textContent = 'Launch on cluster';
    $('launch-fit').className = 'fit-result no-fit';
    $('launch-fit').innerHTML = `<span class="fit-symbol">!</span><div><b>Launch validation failed</b><small>${escapeHtml(error.message)}</small></div>`;
    $('launch-fit').hidden = false;
  }
}

function renderRuntimeReadiness(report) {
  runtimeReadiness = report;
  $('runtime-name').textContent = report.runtime_name;
  $('runtime-message').textContent = report.message;
  $('runtime-version').textContent = report.required_version;
  $('runtime-node-grid').innerHTML = report.nodes.length ? report.nodes.map((node, index) => `
    <article class="runtime-node ${escapeHtml(node.status)}">
      <i>${node.role === 'head' ? 'H' : index + 1}</i>
      <div><b>${escapeHtml(node.address)}</b><small>${escapeHtml(node.installed_version ? `Installed ${node.installed_version} · ${node.message}` : node.message)}</small></div>
      <span>${escapeHtml(node.status)}</span>
    </article>`).join('') : '<div class="catalog-empty">No SSH-managed nodes are available.<br>Complete Compute pools before installing a runtime.</div>';
  const selected = $('runtime-terminal-node').value;
  $('runtime-terminal-node').innerHTML = report.nodes.map(node => `<option value="${escapeHtml(node.address)}">${escapeHtml(node.address)} · ${escapeHtml(node.role)}</option>`).join('');
  if (report.nodes.some(node => node.address === selected)) $('runtime-terminal-node').value = selected;
  $('runtime-terminal-connect').disabled = !report.nodes.length;
  const hasOutdated = report.nodes.some(node => node.status === 'outdated');
  const button = $('runtime-install');
  button.textContent = report.ready ? 'Launch model' : hasOutdated ? 'Update all nodes' : 'Install on all nodes';
  button.disabled = !report.ready && (!report.install_supported || !report.nodes.length);
  $('runtime-sheet-error').hidden = true;
}

function openRuntimeReadinessSheet(report, opener) {
  pendingLaunchPayload = launchPayload();
  renderRuntimeReadiness(report);
  $('runtime-technical').hidden = true;
  $('runtime-technical-toggle').textContent = 'Open terminal & logs';
  $('runtime-sheet').hidden = false;
  document.body.classList.add('bottom-sheet-open');
  requestAnimationFrame(() => {
    $('runtime-sheet').classList.add('open');
    $('runtime-sheet-close').focus({preventScroll: true});
  });
  $('runtime-sheet').dataset.opener = opener?.id || '';
}

function closeRuntimeReadinessSheet() {
  if ($('runtime-sheet').hidden) return;
  clearTimeout(runtimeInstallTimer);
  if (runtimeTerminalSocket) runtimeTerminalSocket.close();
  runtimeTerminalSocket = null;
  $('runtime-sheet').classList.remove('open');
  document.body.classList.remove('bottom-sheet-open');
  setTimeout(() => {
    $('runtime-sheet').hidden = true;
    const opener = $($('runtime-sheet').dataset.opener);
    if (opener) opener.focus({preventScroll: true});
  }, 220);
}

function renderRuntimeInstallJob(job) {
  runtimeInstallJob = job;
  $('runtime-install-progress').hidden = false;
  $('runtime-install-bar').style.width = `${job.progress}%`;
  $('runtime-install-label').textContent = `${job.status.replaceAll('-', ' ')} · ${job.progress.toFixed(0)}%`;
  if (job.nodes?.length) {
    renderRuntimeReadiness({...runtimeReadiness, nodes: job.nodes, ready: job.status === 'completed', message: job.status === 'completed' ? 'Required runtime is ready on every node' : 'Installing the runtime on cluster nodes'});
  }
  if (!['completed', 'failed'].includes(job.status)) {
    $('runtime-install').disabled = true;
    $('runtime-install').textContent = 'Installing…';
  }
  $('runtime-install-logs').innerHTML = job.logs.length ? job.logs.map(log => `<div class="runtime-install-log ${escapeHtml(log.level)}"><time>${new Date(log.timestamp).toLocaleTimeString([], {hour12: false})}</time><b>${escapeHtml(log.node)}</b><span>${escapeHtml(log.message)}</span></div>`).join('') : '<span>Waiting for node installation logs…</span>';
  $('runtime-install-logs').scrollTop = $('runtime-install-logs').scrollHeight;
}

async function pollRuntimeInstall() {
  if (!runtimeInstallJob || $('runtime-sheet').hidden) return;
  try {
    const job = await api(`/api/v1/runtime-management/install/${runtimeInstallJob.id}`, {headers: launchAuthHeaders()});
    renderRuntimeInstallJob(job);
    if (job.status === 'completed') {
      const report = await api('/api/v1/runtime-management/readiness', {method: 'POST', headers: launchAuthHeaders(), body: JSON.stringify({model_id: pendingLaunchPayload.model_id, engine_id: pendingLaunchPayload.engine_id, cluster_id: launchClusterId()})});
      renderRuntimeReadiness(report);
      return;
    }
    if (job.status === 'failed') {
      $('runtime-sheet-error').textContent = job.error || 'One or more nodes need attention.';
      $('runtime-sheet-error').hidden = false;
      $('runtime-technical').hidden = false;
      $('runtime-technical-toggle').textContent = 'Hide terminal & logs';
      return;
    }
    runtimeInstallTimer = setTimeout(pollRuntimeInstall, 1000);
  } catch (error) {
    $('runtime-sheet-error').textContent = error.message;
    $('runtime-sheet-error').hidden = false;
  }
}

async function installRuntimeOnNodes() {
  if (runtimeReadiness?.ready) {
    closeRuntimeReadinessSheet();
    await queueModelLaunch(pendingLaunchPayload);
    return;
  }
  const button = $('runtime-install');
  button.disabled = true;
  button.textContent = 'Starting installation…';
  try {
    runtimeInstallJob = await api('/api/v1/runtime-management/install', {method: 'POST', headers: launchAuthHeaders(), body: JSON.stringify({model_id: pendingLaunchPayload.model_id, engine_id: pendingLaunchPayload.engine_id, cluster_id: launchClusterId()})});
    renderRuntimeInstallJob(runtimeInstallJob);
    runtimeInstallTimer = setTimeout(pollRuntimeInstall, 500);
  } catch (error) {
    $('runtime-sheet-error').textContent = error.message;
    $('runtime-sheet-error').hidden = false;
    button.disabled = false;
    button.textContent = 'Install on all nodes';
  }
}

function appendTerminalOutput(value) {
  const clean = value.replace(/\x1B\[[0-?]*[ -/]*[@-~]/g, '').replace(/\r/g, '');
  const terminal = $('runtime-terminal-output');
  terminal.textContent = (terminal.textContent + clean).slice(-100000);
  terminal.scrollTop = terminal.scrollHeight;
}

async function connectRuntimeTerminal() {
  if (runtimeTerminalSocket) runtimeTerminalSocket.close();
  const address = $('runtime-terminal-node').value;
  if (!address) return;
  $('runtime-terminal-output').textContent = `Connecting to ${address}…\n`;
  try {
    const ticket = await api('/api/v1/runtime-management/terminal-ticket', {method: 'POST', headers: launchAuthHeaders(), body: JSON.stringify({address})});
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    runtimeTerminalSocket = new WebSocket(`${protocol}//${location.host}/api/v1/runtime-management/terminal/${encodeURIComponent(ticket.ticket)}`);
    runtimeTerminalSocket.onopen = () => {
      $('runtime-terminal-command').disabled = false;
      $('runtime-terminal-interrupt').disabled = false;
      $('runtime-terminal-command').focus();
    };
    runtimeTerminalSocket.onmessage = event => {
      const payload = JSON.parse(event.data);
      if (payload.type === 'output') appendTerminalOutput(payload.data);
    };
    runtimeTerminalSocket.onclose = () => {
      appendTerminalOutput('\n[terminal disconnected]\n');
      $('runtime-terminal-command').disabled = true;
      $('runtime-terminal-interrupt').disabled = true;
      runtimeTerminalSocket = null;
    };
  } catch (error) {
    appendTerminalOutput(`[connection failed] ${error.message}\n`);
  }
}

async function queueModelLaunch(payload) {
  $('launch-model-button').disabled = true;
  $('launch-model-button').textContent = 'Launching…';
  try {
    await api('/api/v1/launches', {method: 'POST', headers: launchAuthHeaders(), body: JSON.stringify(payload)});
    $('launch-model-button').textContent = 'Launch queued';
    await refreshLibrary();
  } catch (error) {
    $('launch-fit').className = 'fit-result no-fit';
    $('launch-fit').innerHTML = `<span class="fit-symbol">!</span><div><b>Launch failed</b><small>${escapeHtml(error.message)}</small></div>`;
    $('launch-model-button').textContent = 'Launch on cluster';
    $('launch-model-button').disabled = false;
  }
}

function renderLogLines() {
  const instance = $('log-instance').value;
  const level = $('log-level').value;
  const visible = logEntries.filter(entry => (!instance || entry.pod === instance) && (!level || entry.level === level));
  const terminal = $('log-terminal');
  const hasLogs = logEntries.length > 0;
  $('logs').classList.toggle('idle', !hasLogs);
  $('logs-toolbar').hidden = !hasLogs;
  $('log-footer').hidden = !hasLogs;
  $('log-count').textContent = `${visible.length} line${visible.length === 1 ? '' : 's'}`;
  $('pause-logs').disabled = !hasLogs;
  $('clear-logs').disabled = !hasLogs;
  terminal.classList.toggle('empty', visible.length === 0);
  if (!visible.length) {
    terminal.innerHTML = `<div class="log-empty">${logEntries.length ? 'No matching logs' : 'Logs will appear when a runtime starts.'}</div>`;
    return;
  }
  terminal.innerHTML = visible.slice(-1000).map(entry => `
    <div class="log-line ${entry.level}"><time>${new Date(entry.timestamp).toLocaleTimeString([], {hour12: false})}</time><span class="log-source" title="${escapeHtml(entry.pod)} / ${escapeHtml(entry.container)}">${escapeHtml(entry.pod)}</span><span class="log-level">${escapeHtml(entry.level)}</span><span class="log-message">${escapeHtml(entry.message)}</span></div>`).join('');
  if (!logsPaused) terminal.scrollTop = terminal.scrollHeight;
}

function updateLogInstances(instances) {
  const select = $('log-instance');
  const selected = select.value;
  select.innerHTML = '<option value="">All instances</option>' + instances.map(instance => `<option value="${escapeHtml(instance)}">${escapeHtml(instance)}</option>`).join('');
  if (instances.includes(selected)) select.value = selected;
}

function addLogInstance(instance) {
  const select = $('log-instance');
  if ([...select.options].some(option => option.value === instance)) return;
  const option = document.createElement('option');
  option.value = instance;
  option.textContent = instance;
  select.appendChild(option);
}

async function loadLogHistory() {
  try {
    const [entries, instances] = await Promise.all([api('/api/v1/logs?limit=1000'), api('/api/v1/logs/instances')]);
    logEntries = entries;
    updateLogInstances(instances);
    renderLogLines();
  } catch (error) {
    $('logs-toolbar').hidden = true;
    $('log-footer').hidden = true;
    $('log-terminal').classList.add('empty');
    $('log-terminal').innerHTML = `<div class="log-empty">${escapeHtml(error.message)}</div>`;
    $('pause-logs').disabled = true;
    $('clear-logs').disabled = true;
  }
}

function connectLogs() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/api/v1/logs/stream`);
  socket.onmessage = event => {
    const entry = JSON.parse(event.data);
    logEntries.push(entry);
    addLogInstance(entry.pod);
    if (logEntries.length > 2000) logEntries.shift();
    if (!logsPaused) renderLogLines();
  };
  socket.onclose = () => setTimeout(connectLogs, 1500);
}

let libraryRefreshRunning = false;
let modelInventoryRefreshRunning = false;
let modelInventoryLoadedAt = 0;

function modelInventoryIsNeeded() {
  return document.body.dataset.route === 'launch' || $('local-models-section').open;
}

async function refreshModelInventory(force = false) {
  if (modelInventoryRefreshRunning || (!force && Date.now() - modelInventoryLoadedAt < 10_000)) return;
  modelInventoryRefreshRunning = true;
  if (!localModels.length) {
    $('local-model-count').textContent = '…';
    $('launch-model').options[0].textContent = 'Loading models…';
  }
  try {
    renderModels(await api('/api/v1/models'));
    modelInventoryLoadedAt = Date.now();
    $('local-model-count').removeAttribute('title');
  } catch {
    $('local-model-count').textContent = '!';
    $('local-model-count').title = 'Inventory is temporarily unavailable';
    if (!localModels.length) $('launch-model').options[0].textContent = 'Models unavailable';
  } finally {
    modelInventoryRefreshRunning = false;
  }
}

async function refreshLibrary() {
  if (libraryRefreshRunning) return;
  libraryRefreshRunning = true;
  if (modelInventoryIsNeeded()) void refreshModelInventory();
  const requests = [
    Promise.all([api('/api/v1/downloads'), api('/api/v1/controller/huggingface')]).then(([downloads, status]) => {
      hfTokenStatus = status;
      renderDownloads(downloads);
      document.dispatchEvent(new CustomEvent('library-updated', {detail: downloads}));
    }),
    api('/api/v1/runtime').then(renderEndpoint),
    api('/api/v1/pool').then(renderPoolEndpoint).catch(() => {
      $('api-status').textContent = '○ pool unavailable';
      $('api-status').className = 'pill unreachable';
      $('api-routing-mode').textContent = 'Pool inventory is temporarily unavailable';
    }),
    api('/api/v1/launches').then(renderLaunches),
  ];
  await Promise.allSettled(requests);
  libraryRefreshRunning = false;
}

function metricValue(value, suffix = '') {
  return value === null || value === undefined ? '—' : `${Number(value).toFixed(1)}${suffix}`;
}

function renderTelemetry(telemetry) {
  $('telemetry-state').textContent = telemetry.available ? '● measured live' : '○ runtime unavailable';
  $('telemetry-state').className = `pill ${telemetry.available ? 'ready' : 'unreachable'}`;
  const values = [
    ['TTFT p50 / p95', `${metricValue(telemetry.ttft_p50_ms, ' ms')} / ${metricValue(telemetry.ttft_p95_ms, ' ms')}`],
    ['Inter-token p50 / p95', `${metricValue(telemetry.inter_token_p50_ms, ' ms')} / ${metricValue(telemetry.inter_token_p95_ms, ' ms')}`],
    ['Prefill / decode', `${metricValue(telemetry.prefill_tokens_per_second, ' t/s')} / ${metricValue(telemetry.decode_tokens_per_second, ' t/s')}`],
    ['Speculative acceptance', telemetry.speculative_acceptance_rate == null ? '—' : `${(telemetry.speculative_acceptance_rate * 100).toFixed(1)}%`],
    ['Measured KV occupancy', telemetry.gpu_kv_cache_usage == null ? '—' : `${(telemetry.gpu_kv_cache_usage * 100).toFixed(1)}%`],
    ['LMCache hits / misses', `${telemetry.cache_hits ?? '—'} / ${telemetry.cache_misses ?? '—'}`],
    ['NCCL p95', metricValue(telemetry.nccl_collective_p95_ms, ' ms')],
    ['Expert imbalance', metricValue(telemetry.expert_imbalance)],
  ];
  $('telemetry-grid').innerHTML = values.map(([label, value]) => `<article><small>${label}</small><b>${value}</b></article>`).join('');
}

function renderBenchmarks(results) {
  $('benchmark-results').innerHTML = results.length ? results.map(result => `
    <article><div><b>${escapeHtml(result.model_id)}</b><small>${escapeHtml(result.engine_id)} · ${escapeHtml(result.topology)} · ${result.concurrency} concurrent · ${escapeHtml(result.cache_state)}</small></div><span><b>${metricValue(result.output_tokens_per_second, ' t/s')}</b><small>output</small></span><span><b>${metricValue(result.ttft_p95_ms, ' ms')}</b><small>TTFT p95</small></span><span class="${result.error ? 'failed' : 'ready'}">${result.error ? escapeHtml(result.error) : `${result.successful_requests} measured`}</span></article>`).join('') : '<div class="empty-state">No benchmark recorded yet.</div>';
}

function renderCachePrefixes(records) {
  $('cache-prefix-records').innerHTML = records.length ? records.map(record => `
    <article><div><b>${escapeHtml(record.model_id)}</b><small>${formatNumber(record.prompt_tokens)} tokens · priority ${record.priority} · ${record.prewarm_count} prewarm(s)</small></div><span class="${escapeHtml(record.state)}">${escapeHtml(record.state)}</span><small>${record.last_ttft_ms == null ? 'TTFT unavailable' : `${record.last_ttft_ms.toFixed(1)} ms`}${record.last_cache_hit_delta == null ? '' : ` · +${record.last_cache_hit_delta} cache hits`}</small></article>`).join('') : '<div class="empty-state">No tracked prefixes yet.</div>';
}

async function refreshOptimizationLab() {
  const [telemetry, benchmarks, prefixes] = await Promise.all([
    api('/api/v1/telemetry'),
    api('/api/v1/optimization/benchmarks?limit=12'),
    api('/api/v1/cache/prefixes?limit=20'),
  ]);
  const lab = $('optimization-lab');
  lab.hidden = !telemetry.available && benchmarks.length === 0;
  if (lab.hidden) lab.open = false;
  const cacheLifecycle = $('cache-lifecycle');
  cacheLifecycle.hidden = !telemetry.available && prefixes.length === 0;
  if (cacheLifecycle.hidden) cacheLifecycle.open = false;
  renderTelemetry(telemetry);
  renderBenchmarks(benchmarks);
  renderCachePrefixes(prefixes);
}

async function runBenchmark() {
  const button = $('run-benchmark');
  const modelId = $('launch-model').value;
  if (!modelId) {
    button.textContent = 'Select a launched model first';
    return;
  }
  button.disabled = true;
  button.textContent = 'Benchmarking…';
  try {
    await api('/api/v1/optimization/benchmarks', {method: 'POST', headers: launchAuthHeaders(), body: JSON.stringify({
      model_id: modelId,
      prompt: $('benchmark-prompt').value,
      max_tokens: Number($('benchmark-tokens').value),
      repetitions: Number($('benchmark-repetitions').value),
      concurrency: Number($('benchmark-concurrency').value),
      cache_state: 'cold',
      objective: $('launch-workload').value,
    })});
    await refreshOptimizationLab();
  } catch (error) {
    button.textContent = error.message;
  } finally {
    button.disabled = false;
    if (button.textContent === 'Benchmarking…') button.textContent = 'Run measured benchmark';
  }
}

async function prewarmPrefix() {
  const button = $('cache-prewarm');
  button.disabled = true;
  button.textContent = 'Prewarming…';
  try {
    const modelId = $('cache-model-id').value.trim();
    await api('/api/v1/cache/prefixes/prewarm', {method: 'POST', headers: authHeaders(), body: JSON.stringify({
      model_id: modelId,
      tokenizer_id: $('cache-tokenizer-id').value.trim() || modelId,
      prompt: $('cache-prefix-prompt').value,
      engine_id: $('launch-engine').value === 'auto' ? 'vllm' : $('launch-engine').value,
      priority: Number($('cache-prefix-priority').value),
      engine_config: {context_length: Number($('launch-context').value), kv_cache_dtype: launchPerformance?.kv_cache_dtype || 'auto'},
    })});
    $('cache-prefix-prompt').value = '';
    await refreshOptimizationLab();
  } catch (error) {
    button.textContent = error.message;
  } finally {
    button.disabled = false;
    if (button.textContent === 'Prewarming…') button.textContent = 'Prewarm prefix';
  }
}

function openModelActionSheet(action, target, opener) {
  clearTimeout(actionSheetCloseTimer);
  pendingModelAction = {action, ...target};
  actionSheetOpener = opener;
  const model = localModels.find(item => item.id === target.repoId);
  const configurations = {
    pause: {
      eyebrow: 'DOWNLOAD CONTROL',
      title: 'Pause this model download?',
      description: 'The transfer stops immediately while keeping every completed and partial file ready for resume.',
      icon: 'Ⅱ',
      confirm: 'Pause download',
      impact: ['Network transfer is suspended', 'Partial files stay on the controller and cluster node', 'Resume continues from preserved data'],
    },
    resume: {
      eyebrow: 'DOWNLOAD CONTROL',
      title: 'Resume this model download?',
      description: 'The controller and cluster continue from the files already present instead of starting over.',
      icon: '▶',
      confirm: 'Resume download',
      impact: ['Existing progress is preserved', 'Internet or LAN transfer restarts', 'Cluster distribution resumes automatically'],
    },
    cancel: {
      eyebrow: 'DESTRUCTIVE ACTION',
      title: 'Cancel and remove this download?',
      description: 'The transfer stops and its partial files are removed from the controller and attached cluster storage.',
      icon: '×',
      confirm: 'Cancel and remove files',
      danger: true,
      impact: ['The running transfer is terminated', 'Partial controller files are deleted', 'Partial cluster-node files are deleted'],
    },
    delete: {
      eyebrow: 'DESTRUCTIVE ACTION',
      title: 'Delete this model everywhere?',
      description: `This removes ${model ? formatBytes(model.size_bytes) : 'the model files'} from the controller and every attached cluster storage location.`,
      icon: '⌫',
      confirm: 'Delete model everywhere',
      danger: true,
      impact: ['Controller model files are deleted', 'The managed cluster copy is deleted', 'A running model must be stopped first'],
    },
  };
  const configuration = configurations[action];
  $('action-sheet-eyebrow').textContent = configuration.eyebrow;
  $('action-sheet-title').textContent = configuration.title;
  $('action-sheet-model').textContent = target.repoId;
  $('action-sheet-description').textContent = configuration.description;
  $('action-sheet-icon').textContent = configuration.icon;
  $('action-sheet-impact').innerHTML = configuration.impact.map(item => `<span><i>✓</i>${escapeHtml(item)}</span>`).join('');
  $('action-sheet-confirm').textContent = configuration.confirm;
  $('action-sheet-confirm').className = configuration.danger ? 'danger' : 'primary';
  $('action-sheet-error').hidden = true;
  $('action-sheet').hidden = false;
  document.body.classList.add('bottom-sheet-open');
  requestAnimationFrame(() => {
    $('action-sheet').classList.add('open');
    $('action-sheet-close').focus({preventScroll: true});
  });
}

function closeModelActionSheet() {
  if ($('action-sheet').hidden) return;
  $('action-sheet').classList.remove('open');
  document.body.classList.remove('bottom-sheet-open');
  const opener = actionSheetOpener;
  pendingModelAction = null;
  actionSheetCloseTimer = setTimeout(() => {
    $('action-sheet').hidden = true;
    if (opener?.isConnected) opener.focus({preventScroll: true});
  }, 220);
}

async function confirmModelAction() {
  if (!pendingModelAction) return;
  const action = pendingModelAction;
  const button = $('action-sheet-confirm');
  button.disabled = true;
  $('action-sheet-cancel').disabled = true;
  $('action-sheet-error').hidden = true;
  const originalLabel = button.textContent;
  button.textContent = 'Applying…';
  try {
    const path = action.action === 'delete'
      ? `/api/v1/model-management/models/${action.repoId.split('/').map(encodeURIComponent).join('/')}`
      : `/api/v1/downloads/${encodeURIComponent(action.jobId)}/${action.action}`;
    await api(path, {method: action.action === 'delete' ? 'DELETE' : 'POST', headers: authHeaders()});
    await refreshLibrary();
    closeModelActionSheet();
  } catch (error) {
    $('action-sheet-error').textContent = error.message;
    $('action-sheet-error').hidden = false;
    button.textContent = originalLabel;
  } finally {
    button.disabled = false;
    $('action-sheet-cancel').disabled = false;
  }
}

function fitPayload() {
  return {
    repo_id: $('repo-id').value.trim(),
    revision: $('revision').value.trim() || 'main',
    bits_per_weight: Number($('weight-bits').value),
    context_length: Number($('context-length').value),
    concurrent_sequences: Number($('concurrency').value),
    kv_cache_dtype: $('fit-kv-dtype').value,
  };
}

function authHeaders() {
  return adminAuthHeaders();
}


function renderFit(plan) {
  currentFit = plan;
  const result = $('fit-result');
  const states = {
    'full-memory': ['fit', '✓', 'Estimated fit in cluster memory'],
    'hybrid-ready': ['hybrid', '↯', 'Estimated hybrid placement'],
    'engine-required': ['warning', '↓', 'Download fits · specialized engine required'],
    'download-only': ['warning', '↓', 'Download fits · runtime plan needs validation'],
    'insufficient-storage': ['no-fit', '!', 'Not enough model storage'],
  };
  const [className, symbol, title] = states[plan.fit_status] || states['download-only'];
  result.className = `fit-result ${className}`;
  result.innerHTML = `<span class="fit-symbol">${symbol}</span><div><b>${title}</b><small>${escapeHtml(plan.detected_format)} · ${escapeHtml(plan.engine_recommendation)} · ${plan.minimum_nodes <= plan.cluster_nodes ? `${plan.minimum_nodes} node${plan.minimum_nodes === 1 ? '' : 's'} minimum` : `more than ${plan.cluster_nodes} available nodes`} · ${formatBytes(plan.repository_bytes)} repository</small></div>`;
  const detail = $('fit-details');
  const metrics = [
    ['Resident weights', `${plan.resident_weights_gib.toFixed(1)} GiB`],
    ['SSD/offloaded weights', `${plan.offloaded_weights_gib.toFixed(1)} GiB`],
    ['Resident KV', `${plan.resident_kv_cache_gib.toFixed(1)} GiB`],
    ['CPU/NVMe KV tier', `${plan.offloaded_kv_cache_gib.toFixed(1)} GiB`],
    ['Runtime overhead', `${plan.runtime_overhead_gib.toFixed(1)} GiB`],
    ['Runnable artifacts', `${plan.runtime_storage_required_gib.toFixed(1)} GiB`],
  ];
  const requirements = plan.requirements.length ? `<div class="fit-guidance"><b>Required</b>${plan.requirements.map(item => `<p>• ${escapeHtml(item)}</p>`).join('')}</div>` : '';
  const tradeoffs = plan.tradeoffs.length ? `<div class="fit-guidance"><b>Tradeoffs · ${escapeHtml(plan.performance_impact)} impact</b>${plan.tradeoffs.map(item => `<p>• ${escapeHtml(item)}</p>`).join('')}</div>` : '';
  const notes = plan.notes.length ? `<div class="fit-guidance planner-notes"><b>Planner notes</b>${plan.notes.map(item => `<p>• ${escapeHtml(item)}</p>`).join('')}</div>` : '';
  detail.innerHTML = `<summary>Technical details</summary><div class="fit-details-body"><div class="fit-strategy"><span>${escapeHtml(plan.placement_strategy.replaceAll('-', ' '))}</span><small>${plan.memory_fits ? 'Estimated memory fit' : plan.runtime_fits ? 'Estimated hybrid fit' : 'No validated runtime fit'} · ${plan.storage_fits ? `${plan.storage_available_gib.toFixed(0)} GiB storage free` : 'repository exceeds free storage'}${plan.ssd_required_gib_per_node ? ` · ${plan.ssd_required_gib_per_node.toFixed(1)} GiB local SSD per node` : ''}</small></div><div class="fit-metrics">${metrics.map(([label, value]) => `<span><small>${label}</small><b>${value}</b></span>`).join('')}</div>${requirements}${tradeoffs}${notes}</div>`;
  detail.open = false;
  detail.hidden = false;
  $('download-model').disabled = !plan.download_fits;
  $('download-model').textContent = plan.runtime_fits ? 'Download model' : 'Download for later';
}

function renderFitError(message) {
  currentFit = null;
  $('fit-result').className = 'fit-result no-fit';
  $('fit-result').innerHTML = `<span class="fit-symbol">!</span><div><b>Fit check failed</b><small>${escapeHtml(message)}</small></div>`;
  $('fit-details').open = false;
  $('fit-details').hidden = true;
  $('download-model').disabled = true;
}

async function checkFit() {
  if (!$('repo-id').value.trim()) {
    renderFitError('Enter a Hugging Face model id.');
    return null;
  }
  const version = ++fitRequestVersion;
  const payload = JSON.stringify(fitPayload());
  $('check-fit').disabled = true;
  $('check-fit').textContent = 'Inspecting…';
  try {
    const plan = await api('/api/v1/models/fit', {method: 'POST', body: payload});
    if (version !== fitRequestVersion || payload !== JSON.stringify(fitPayload())) return null;
    renderFit(plan);
    return plan;
  } catch (error) {
    if (version === fitRequestVersion) renderFitError(error.message);
    return null;
  } finally {
    if (version === fitRequestVersion) {
      $('check-fit').disabled = false;
      $('check-fit').textContent = 'Check fit';
    }
  }
}

$('admin-token-form').addEventListener('submit', event => {
  event.preventDefault();
  if (adminTokenVerifying || !adminTokenWaiter) return;
  const token = $('admin-token-value').value.trim();
  if (!token) return;
  adminTokenVerifying = true;
  $('admin-token-error').hidden = true;
  $('admin-token-submit').disabled = true;
  $('admin-token-submit').textContent = 'Verifying…';
  const waiter = adminTokenWaiter;
  adminTokenWaiter = null;
  waiter.resolve(token);
});
function cancelAdminTokenPrompt() {
  if (adminTokenVerifying) return;
  if (adminTokenWaiter) {
    const waiter = adminTokenWaiter;
    adminTokenWaiter = null;
    waiter.reject(new Error('Administrator authentication was cancelled.'));
  }
  closeAdminTokenSheet();
}
$('admin-token-close').addEventListener('click', cancelAdminTokenPrompt);
$('admin-token-cancel').addEventListener('click', cancelAdminTokenPrompt);
$('admin-token-backdrop').addEventListener('click', cancelAdminTokenPrompt);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !$('admin-token-sheet').hidden) cancelAdminTokenPrompt();
});

$('copy-endpoint').addEventListener('click', async () => {
  await navigator.clipboard.writeText($('api-endpoint').textContent);
  $('copy-endpoint').textContent = 'Copied';
  setTimeout(() => { $('copy-endpoint').textContent = 'Copy'; }, 1200);
});
$('mcp-endpoint').textContent = `${window.location.origin}/mcp`;
$('copy-mcp-endpoint').addEventListener('click', async () => {
  await navigator.clipboard.writeText($('mcp-endpoint').textContent);
  $('copy-mcp-endpoint').textContent = 'Copied';
  setTimeout(() => { $('copy-mcp-endpoint').textContent = 'Copy'; }, 1200);
});
$('check-fit').addEventListener('click', checkFit);
$('model-form').addEventListener('input', event => {
  ++fitRequestVersion;
  currentFit = null;
  $('check-fit').disabled = false;
  $('check-fit').textContent = 'Check fit';
  $('download-model').disabled = true;
  $('download-model').textContent = 'Download once';
  $('fit-details').open = false;
  $('fit-details').hidden = true;
});
$('model-form').addEventListener('submit', async event => {
  event.preventDefault();
  const plan = currentFit || await checkFit();
  if (!plan) return;
  $('download-model').disabled = true;
  $('download-model').textContent = 'Queueing…';
  try {
    await api('/api/v1/models/download', {method: 'POST', headers: authHeaders(), body: JSON.stringify({...fitPayload(), allow_oversized: false})});
    $('download-model').textContent = 'Queued';
    await refreshLibrary();
  } catch (error) {
    renderFitError(error.message);
    $('download-model').textContent = 'Download once';
  }
});
$('download-list').addEventListener('click', async event => {
  const button = event.target.closest('[data-job-id]');
  if (!button) return;
  button.disabled = true;
  button.textContent = 'Retrying…';
  try {
    await api(`/api/v1/downloads/${button.dataset.jobId}/retry`, {method: 'POST', headers: authHeaders()});
    await refreshLibrary();
  } catch (error) {
    button.textContent = 'Failed';
    button.title = error.message;
  }
});
$('active-download-list').addEventListener('click', event => {
  const button = event.target.closest('[data-download-action]');
  if (button) {
    openModelActionSheet(button.dataset.downloadAction, {
      jobId: button.dataset.jobId,
      repoId: button.dataset.repoId,
    }, button);
    return;
  }
  const model = event.target.closest('[data-model-repo]');
  if (model) window.inspectCatalogModel?.(
    model.dataset.modelRepo,
    model.dataset.modelRevision || 'main',
    model,
  );
});
$('active-download-list').addEventListener('keydown', event => {
  if (!['Enter', ' '].includes(event.key)) return;
  const model = event.target.closest('[data-model-repo]');
  if (!model || event.target !== model) return;
  event.preventDefault();
  window.inspectCatalogModel?.(
    model.dataset.modelRepo,
    model.dataset.modelRevision || 'main',
    model,
  );
});
$('local-models').addEventListener('click', event => {
  const button = event.target.closest('[data-model-action="delete"]');
  if (button) {
    if (!button.disabled) openModelActionSheet('delete', {repoId: button.dataset.repoId}, button);
    return;
  }
  const model = event.target.closest('[data-model-repo]');
  if (model) window.inspectCatalogModel?.(
    model.dataset.modelRepo,
    model.dataset.modelRevision || 'main',
    model,
  );
});
$('local-model-query').addEventListener('input', event => {
  localModelQuery = event.currentTarget.value;
  renderLocalModelCards();
});
$('action-sheet-confirm').addEventListener('click', confirmModelAction);
$('action-sheet-cancel').addEventListener('click', closeModelActionSheet);
$('action-sheet-close').addEventListener('click', closeModelActionSheet);
$('action-sheet-backdrop').addEventListener('click', closeModelActionSheet);
$('overview-nodes').addEventListener('click', event => {
  const button = event.target.closest('[data-resource-node][data-resource-metric]');
  if (button) void openResourceHistory(button.dataset.resourceNode, button.dataset.resourceMetric, button);
});
$('resource-history-close').addEventListener('click', closeResourceHistory);
$('resource-history-backdrop').addEventListener('click', closeResourceHistory);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !$('action-sheet').hidden) closeModelActionSheet();
  if (event.key === 'Escape' && !$('resource-history-sheet').hidden) closeResourceHistory();
});

$('launch-model').addEventListener('change', async () => {
  const model = localModels.find(item => item.id === $('launch-model').value);
  if (model) {
    $('served-model-name').value = model.id.split('/').pop().toLowerCase();
    $('cache-model-id').value = model.id;
    $('cache-tokenizer-id').value = model.id;
  }
  await loadPerformancePlan(true);
  checkLaunchFit();
});
['served-model-name', 'launch-context', 'launch-bits', 'tensor-parallel', 'pipeline-parallel', 'gpu-memory-utilization', 'launch-concurrency', 'launch-engine', 'launch-speculative', 'launch-speculative-tokens', 'launch-expert-parallel', 'launch-eplb', 'launch-all2all', 'launch-redundant-experts', 'launch-cuda-graphs', 'launch-chunked-prefill', 'launch-cache-cpu', 'launch-cache-disk'].forEach(id => {
  $(id).addEventListener('change', checkLaunchFit);
});
$('launch-workload').addEventListener('change', async () => {
  await loadPerformancePlan(true);
  checkLaunchFit();
});
$('served-model-name').addEventListener('input', () => {
  $('launch-model-button').disabled = !launchFit?.fits || !launchFit?.parallel_fits || !$('served-model-name').value.trim();
});
$('launch-form').addEventListener('submit', async event => {
  event.preventDefault();
  const button = $('launch-model-button');
  button.disabled = true;
  try {
    await window.loadModelInPool(launchPayload());
    button.textContent = 'Load queued';
  } catch (error) {
    $('launch-fit').hidden = false;
    $('launch-fit').textContent = error.message;
    button.textContent = 'Load model';
  } finally { button.disabled = false; }
});
$('runtime-install').addEventListener('click', installRuntimeOnNodes);
$('runtime-sheet-close').addEventListener('click', closeRuntimeReadinessSheet);
$('runtime-sheet-later').addEventListener('click', closeRuntimeReadinessSheet);
$('runtime-sheet-backdrop').addEventListener('click', closeRuntimeReadinessSheet);
$('runtime-technical-toggle').addEventListener('click', () => {
  $('runtime-technical').hidden = !$('runtime-technical').hidden;
  $('runtime-technical-toggle').textContent = $('runtime-technical').hidden ? 'Open terminal & logs' : 'Hide terminal & logs';
});
$('runtime-terminal-connect').addEventListener('click', connectRuntimeTerminal);
$('runtime-terminal-interrupt').addEventListener('click', () => {
  if (runtimeTerminalSocket?.readyState === WebSocket.OPEN) runtimeTerminalSocket.send(JSON.stringify({type: 'input', data: '\u0003'}));
});
$('runtime-terminal-command').addEventListener('keydown', event => {
  if (event.key !== 'Enter' || event.shiftKey) return;
  event.preventDefault();
  const input = $('runtime-terminal-command');
  if (runtimeTerminalSocket?.readyState === WebSocket.OPEN && input.value) {
    runtimeTerminalSocket.send(JSON.stringify({type: 'input', data: `${input.value}\n`}));
    input.value = '';
  }
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !$('runtime-sheet').hidden) closeRuntimeReadinessSheet();
});
$('launch-history').addEventListener('click', async event => {
  const button = event.target.closest('[data-launch-id]');
  if (!button) return;
  button.disabled = true;
  try {
    await api(`/api/v1/launches/${button.dataset.launchId}/retry`, {method: 'POST', headers: launchAuthHeaders()});
    await refreshLibrary();
  } catch (error) {
    button.textContent = error.message;
  }
});
$('log-instance').addEventListener('change', renderLogLines);
$('log-level').addEventListener('change', renderLogLines);
$('pause-logs').addEventListener('click', () => {
  logsPaused = !logsPaused;
  $('pause-logs').classList.toggle('active', logsPaused);
  $('pause-logs').textContent = logsPaused ? 'Resume' : 'Pause';
  if (!logsPaused) renderLogLines();
});
$('clear-logs').addEventListener('click', () => {
  logsPaused = false;
  $('pause-logs').classList.remove('active');
  $('pause-logs').textContent = 'Pause';
  logEntries = [];
  renderLogLines();
});
$('run-benchmark').addEventListener('click', runBenchmark);
$('cache-prewarm').addEventListener('click', prewarmPrefix);

$('cluster-select').addEventListener('change', event => selectCluster(event.target.value));
$('cluster-list').addEventListener('click', event => {
  const refresh = event.target.closest('[data-refresh-local]');
  if (refresh) {
    void detectLocalCluster(refresh);
    return;
  }
  const button = event.target.closest('[data-select-cluster]');
  if (button) selectCluster(button.dataset.selectCluster);
});
$('local-models-section').addEventListener('toggle', event => {
  if (event.currentTarget.open) void refreshModelInventory(true);
});
window.addEventListener('hashchange', () => {
  if (document.body.dataset.route === 'launch') void refreshModelInventory(true);
});

async function detectLocalCluster(button) {
  button.disabled = true;
  $('cluster-fleet-status').textContent = 'Inspecting Docker and NVIDIA GPUs…';
  try {
    await api('/api/v1/clusters/local-docker', {method:'POST', headers:clusterAuthHeaders(), body:JSON.stringify({name:'Unraid GPUs'})});
    await loadClusters();
    location.reload();
  } catch (error) {
    $('cluster-fleet-status').textContent = error.message;
    button.disabled = false;
  }
}
$('cluster-add-local').addEventListener('click', event => detectLocalCluster(event.currentTarget));
loadClusters().catch(error => { $('cluster-fleet-status').textContent = error.message; });

connect();
connectLogs();
loadLogHistory();
refreshLibrary();
refreshOptimizationLab().catch(() => {});
setInterval(() => {
  refreshLibrary();
  if (latestSnapshot && Date.now() - Date.parse(latestSnapshot.timestamp) > 15000) setStreamStatus("reconnecting", "Stale telemetry");
}, 2000);
setInterval(() => refreshOptimizationLab().catch(() => {}), 5000);
