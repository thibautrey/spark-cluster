const $ = (id) => document.getElementById(id);
const throughputHistory = [];
let currentFit = null;

const formatNumber = (value) => value >= 1e9 ? `${(value / 1e9).toFixed(1)}B` : value >= 1e6 ? `${(value / 1e6).toFixed(1)}M` : value >= 1e3 ? `${(value / 1e3).toFixed(1)}k` : `${Math.round(value)}`;
const formatBytes = (value) => value >= 2 ** 30 ? `${(value / 2 ** 30).toFixed(1)} GiB` : value >= 2 ** 20 ? `${(value / 2 ** 20).toFixed(1)} MiB` : `${Math.round(value / 1024)} KiB`;
const pct = (used, total) => total ? Math.min(100, used / total * 100) : 0;
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[character]));

async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {'Content-Type': 'application/json', ...(options.headers || {})}});
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : body.detail?.message || `Request failed (${response.status})`;
    const error = new Error(detail);
    error.payload = body;
    throw error;
  }
  return body;
}

function render(snapshot) {
  const memoryUsed = snapshot.nodes.reduce((sum, node) => sum + node.memory.used_gib, 0);
  const memoryTotal = snapshot.nodes.reduce((sum, node) => sum + node.memory.total_gib, 0);
  $('cluster-name').textContent = snapshot.cluster_name.toUpperCase();
  $('timestamp').textContent = new Date(snapshot.timestamp).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
  $('output-tps').textContent = Math.round(snapshot.total_output_tokens_per_second);
  $('hit-rate').textContent = snapshot.cache_hit_rate.toFixed(1);
  $('hit-ring').style.background = `conic-gradient(var(--green) ${snapshot.cache_hit_rate}%,#202a38 0)`;
  $('memory-used').textContent = Math.round(memoryUsed);
  $('memory-total').textContent = `${Math.round(memoryTotal)} GiB total · ${pct(memoryUsed, memoryTotal).toFixed(0)}% allocated`;
  $('memory-bar').style.width = `${pct(memoryUsed, memoryTotal)}%`;
  $('prefix-reuse').textContent = formatNumber(snapshot.prefix_reuse_tokens);
  $('model-short').textContent = snapshot.runtime.model.split('/').pop().replace('DeepSeek-', '');
  $('runtime-context').textContent = `${Math.round(snapshot.runtime.context_length / 1024)}k`;
  $('runtime-parallel').textContent = `${snapshot.runtime.tensor_parallel} × ${snapshot.runtime.pipeline_parallel}`;
  $('runtime-status').textContent = snapshot.status[0].toUpperCase() + snapshot.status.slice(1);
  $('node-count').textContent = `${snapshot.nodes.length} nodes`;
  renderTopology(snapshot.nodes);
  renderCache(snapshot.cache_tiers);
  renderMemory(snapshot.nodes);
  throughputHistory.push(snapshot.total_output_tokens_per_second);
  if (throughputHistory.length > 40) throughputHistory.shift();
  drawSparkline(throughputHistory);
}

function renderTopology(nodes) {
  $('topology-map').innerHTML = '<div class="fabric-line"></div>' + nodes.map(node => `
    <div class="node">
      <div class="node-top"><span class="node-icon">${node.role === 'head' ? 'H' : 'W'}</span><i class="node-state"></i></div>
      <h3>${escapeHtml(node.hostname)}</h3><p>${escapeHtml(node.role.toUpperCase())} · ${escapeHtml(node.address)}</p>
      <div class="node-stats"><span>GPU<b>${node.gpu_utilization.toFixed(0)}%</b></span><span>MEM<b>${node.memory.used_gib.toFixed(0)} / ${node.memory.total_gib.toFixed(0)}</b></span><span>OUTPUT<b>${node.output_tokens_per_second.toFixed(0)} t/s</b></span><span>TEMP<b>${node.temperature_c.toFixed(0)}°C</b></span></div>
    </div>`).join('');
}

function renderCache(tiers) {
  $('cache-tiers').innerHTML = tiers.map(tier => {
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
  $('memory-grid').innerHTML = nodes.map(node => {
    const model = Math.round(node.memory.model_gib / 8);
    const kv = Math.round(node.memory.kv_cache_gib / 3);
    const runtime = Math.max(1, Math.round((node.memory.used_gib - node.memory.model_gib - node.memory.kv_cache_gib) / 3));
    const free = Math.max(1, 16 - model - kv - runtime);
    const blocks = (className, count) => Array(Math.max(0, count)).fill(`<i class="${className}"></i>`).join('');
    return `<div><div class="memory-node-head"><b>${escapeHtml(node.hostname)}</b><span>${node.memory.used_gib.toFixed(0)} / ${node.memory.total_gib.toFixed(0)} GiB</span></div><div class="memory-block">${blocks('model', model)}${blocks('kv', kv)}${blocks('runtime', runtime)}${blocks('free', free)}</div><div class="memory-labels"><span><i class="m"></i>model</span><span><i class="k"></i>KV</span><span><i class="r"></i>runtime</span><span><i class="f"></i>free</span></div></div>`;
  }).join('');
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

function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/api/v1/stream`);
  socket.onopen = () => {
    $('connection-dot').style.background = 'var(--green)';
    $('connection-label').textContent = 'Live stream';
  };
  socket.onmessage = event => render(JSON.parse(event.data));
  socket.onclose = () => {
    $('connection-dot').style.background = 'var(--amber)';
    $('connection-label').textContent = 'Reconnecting';
    setTimeout(connect, 1500);
  };
}

function renderEndpoint(endpoint) {
  $('api-endpoint').textContent = endpoint.base_url;
  $('api-models-route').textContent = endpoint.models_url;
  $('api-chat-route').textContent = endpoint.chat_completions_url;
  $('api-status').textContent = endpoint.status === 'ready' ? '● ready' : '○ awaiting runtime';
  $('api-status').className = `pill ${endpoint.status}`;
}

function renderModels(models) {
  $('model-count').textContent = `${models.length} model${models.length === 1 ? '' : 's'}`;
  if (!models.length) {
    $('local-models').innerHTML = '<div class="empty-state"><b>No local model found</b><span>Add one from Hugging Face.</span></div>';
    return;
  }
  $('local-models').innerHTML = models.map(model => `
    <article class="model-card">
      <span class="model-glyph">M</span>
      <div><h3>${escapeHtml(model.id)}</h3><p>${escapeHtml(model.source)} · ${formatBytes(model.size_bytes)}${model.revision ? ` · ${escapeHtml(model.revision)}` : ''}</p></div>
      <span class="model-state ${model.loaded ? 'loaded' : ''}">${model.loaded ? '● loaded' : '○ local'}</span>
    </article>`).join('');
}

function renderDownloads(downloads) {
  if (!downloads.length) {
    $('download-list').innerHTML = '<div class="empty-downloads">No downloads yet.</div>';
    return;
  }
  $('download-list').innerHTML = downloads.map(job => `
    <div class="download-row">
      <div class="download-name"><b>${escapeHtml(job.repo_id)}</b><small>attempt ${job.attempts} · ${formatBytes(job.downloaded_bytes)} / ${formatBytes(job.total_bytes)}</small></div>
      <div class="download-progress"><div class="cache-track"><i style="width:${job.progress}%"></i></div><span>${job.progress.toFixed(1)}%</span></div>
      <span class="download-status ${job.status}">${escapeHtml(job.status)}</span>
      ${job.status === 'failed' ? `<button class="download-retry" data-job-id="${escapeHtml(job.id)}" title="${escapeHtml(job.error || 'Retry download')}">Retry</button>` : '<span></span>'}
    </div>`).join('');
}

async function refreshLibrary() {
  try {
    const [endpoint, models, downloads] = await Promise.all([
      api('/api/v1/runtime'),
      api('/api/v1/models'),
      api('/api/v1/downloads'),
    ]);
    renderEndpoint(endpoint);
    renderModels(models);
    renderDownloads(downloads);
  } catch (error) {
    $('api-status').textContent = 'control plane error';
    $('api-status').className = 'pill unreachable';
  }
}

function fitPayload() {
  return {
    repo_id: $('repo-id').value.trim(),
    revision: $('revision').value.trim() || 'main',
    bits_per_weight: Number($('weight-bits').value),
    context_length: Number($('context-length').value),
    concurrent_sequences: Number($('concurrency').value),
  };
}

function authHeaders() {
  const token = $('admin-token').value.trim();
  return token ? {Authorization: `Bearer ${token}`} : {};
}

function renderFit(plan) {
  currentFit = plan;
  const result = $('fit-result');
  result.className = `fit-result ${plan.fits ? 'fit' : 'no-fit'}`;
  result.innerHTML = `<span class="fit-symbol">${plan.fits ? '✓' : '!'}</span><div><b>${plan.fits ? 'Fits this cluster' : 'Does not fit current capacity'}</b><small>${plan.weights_gib.toFixed(1)} GiB weights + ${plan.kv_cache_gib.toFixed(1)} GiB KV + ${plan.runtime_overhead_gib.toFixed(1)} GiB runtime = ${plan.total_required_gib.toFixed(1)} GiB · ${plan.minimum_nodes} node${plan.minimum_nodes === 1 ? '' : 's'} minimum · ${formatBytes(plan.repository_bytes)} download</small></div>`;
  $('download-model').disabled = !plan.fits && !$('allow-oversized').checked;
}

function renderFitError(message) {
  currentFit = null;
  $('fit-result').className = 'fit-result no-fit';
  $('fit-result').innerHTML = `<span class="fit-symbol">!</span><div><b>Fit check failed</b><small>${escapeHtml(message)}</small></div>`;
  $('download-model').disabled = true;
}

async function checkFit() {
  if (!$('repo-id').value.trim()) {
    renderFitError('Enter a Hugging Face model id.');
    return null;
  }
  $('check-fit').disabled = true;
  $('check-fit').textContent = 'Inspecting…';
  try {
    const plan = await api('/api/v1/models/fit', {method: 'POST', body: JSON.stringify(fitPayload())});
    renderFit(plan);
    return plan;
  } catch (error) {
    renderFitError(error.message);
    return null;
  } finally {
    $('check-fit').disabled = false;
    $('check-fit').textContent = 'Check fit';
  }
}

$('copy-endpoint').addEventListener('click', async () => {
  await navigator.clipboard.writeText($('api-endpoint').textContent);
  $('copy-endpoint').textContent = 'Copied';
  setTimeout(() => { $('copy-endpoint').textContent = 'Copy'; }, 1200);
});
$('check-fit').addEventListener('click', checkFit);
$('allow-oversized').addEventListener('change', () => {
  $('download-model').disabled = !currentFit || (!currentFit.fits && !$('allow-oversized').checked);
});
$('model-form').addEventListener('input', event => {
  if (event.target.id === 'allow-oversized') return;
  currentFit = null;
  $('download-model').disabled = true;
});
$('model-form').addEventListener('submit', async event => {
  event.preventDefault();
  const plan = currentFit || await checkFit();
  if (!plan) return;
  $('download-model').disabled = true;
  $('download-model').textContent = 'Queueing…';
  try {
    await api('/api/v1/models/download', {method: 'POST', headers: authHeaders(), body: JSON.stringify({...fitPayload(), allow_oversized: $('allow-oversized').checked})});
    $('download-model').textContent = 'Queued';
    await refreshLibrary();
  } catch (error) {
    renderFitError(error.message);
    $('download-model').textContent = 'Download model';
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

connect();
refreshLibrary();
setInterval(refreshLibrary, 2000);
