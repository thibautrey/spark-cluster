// Uses the shared dashboard API helper and its origin-scoped administrator credential.
let setupStage = 0;
let setupState = null;
let setupCandidates = [];
let setupSelection = new Map();
let setupChecks = null;
let setupJob = null;
let setupBusy = false;
let setupPolling = false;
let setupBootstrapPolling = false;
let hardwareTopology = null;
let setupDiscoveryState = 'idle';
const setupAuth = () => window.adminAuthHeaders();
const setupPayload = () => ({name: $('setup-name').value.trim(), topology: $('setup-topology').value, nodes: [...setupSelection.values()].map(node => ({address: node.address, username: node.username || '', fingerprint: node.fingerprint, role: node.role || 'worker', fabric_address: node.fabric_address || null}))});

function requiresDgxFabric() {
  return setupSelection.size > 1 && [...setupSelection.values()].every(node => node.machine_kind === 'dgx-spark');
}
function adaptSetupForMachines() {
  const nodes = [...setupSelection.values()];
  const dgx = requiresDgxFabric();
  $('hardware-assistant').hidden = !dgx;
  $('setup-sync-guide').hidden = !dgx;
  for (const option of $('setup-topology').options) {
    option.hidden = option.value === 'single' ? nodes.length !== 1 : option.value === 'switch' ? nodes.length < 2 : !dgx || (option.value === 'ring' && nodes.length !== 4);
    option.disabled = option.hidden;
  }
  if (!dgx && nodes.length > 1) $('setup-topology').value = 'switch';
  $('topology-choice-note').textContent = dgx
    ? 'DGX Spark cluster: choose the physical fabric layout and inspect its ConnectX links.'
    : nodes.length === 1 ? 'One machine, one or more GPUs: no cluster fabric is required.'
    : 'Machines share a compute pool. Model placement depends on available runtimes; combining GPU memory requires a compatible distributed runtime.';
  const labels = {'dgx-spark':'DGX Spark', 'linux-nvidia':'Linux · NVIDIA', 'linux-amd':'Linux · AMD', 'mac-apple-silicon':'Apple Silicon', 'windows-gpu':'Windows · GPU'};
  const instructions = {'metal-agent':'macOS: automatic installation is not available. Connect a separately configured Metal runtime through an existing pool controller.', 'windows-agent':'Windows: automatic installation is not available. Connect a separately configured runtime through an existing pool controller.', 'docker-rocm':'ROCm: automatic installation is not available. Connect a separately configured ROCm runtime through an existing pool controller.'};
  const summary = nodes.map(node => `<article class="setup-diagnostic ${node.install_supported === false ? 'warning' : 'pass'}"><span>⬡</span><div><b>${escapeHtml(node.hostname || node.address)} · ${escapeHtml(labels[node.machine_kind] || node.machine_kind || 'Hardware not inspected')}</b><small>${escapeHtml(node.platform || 'OS detected during readiness check')} · ${escapeHtml(node.architecture || 'architecture pending')}${node.accelerator_backend ? ` · ${escapeHtml(node.accelerator_backend.toUpperCase())}` : ''}</small><p>${escapeHtml((node.accelerators || []).join(', ') || 'GPU detection pending')}${node.memory_gib ? ` · ${Math.round(node.memory_gib)} GiB system/unified memory` : ''}</p><small>${escapeHtml(instructions[node.setup_strategy] || (node.install_supported ? 'Linux NVIDIA provisioning · k3s and GPU runtime' : node.machine_kind ? 'Automatic provisioning is unavailable for this platform.' : 'Run readiness checks to detect the platform and installation path.'))}</small></div></article>`).join('');
  $('setup-machine-summary').innerHTML = summary;
  $('setup-check-machine-summary').innerHTML = summary;
}

function setupError(message = '') {
  $('setup-error').hidden = !message;
  $('setup-error').textContent = message;
}
function renderSetupDiscoveryState() {
  const hasMachines = setupCandidates.length > 0 || setupSelection.size > 0;
  $('discovery-note').hidden = setupDiscoveryState === 'idle';
  $('setup-nodes').hidden = setupDiscoveryState !== 'complete' && !hasMachines;
  $('setup-footer').hidden = setupStage === 0 && !hasMachines;
}
function setupNavigate(stage) {
  setupStage = stage;
  document.querySelectorAll('[data-setup-page]').forEach(page => { page.hidden = Number(page.dataset.setupPage) !== stage; });
  [...$('setup-rail').children].forEach((item, index) => {
    item.className = index === stage ? 'active' : index < stage ? 'done' : '';
    if (index === stage) item.setAttribute('aria-current', 'step'); else item.removeAttribute('aria-current');
  });
  $('setup-back').hidden = stage === 0 || stage >= 3;
  $('setup-next').hidden = stage >= 3;
  $('setup-next').textContent = ['Choose machines →', 'Check readiness →', 'Install compute pool →'][stage] || '';
  $('setup-summary').textContent = `${setupSelection.size} machine${setupSelection.size === 1 ? '' : 's'} selected${setupJob ? ` · ${setupJob.status}` : ''}`;
  renderSetupDiscoveryState();
  setupButtons();
}
function setupButtons() {
  $('setup-next').disabled = setupBusy || !setupState?.enabled || !setupSelection.size ||
    (setupStage === 1 && ([...setupSelection.values()].some(node => node.access_status !== 'ready' || !node.username?.trim()) || (requiresDgxFabric() && (!hardwareTopology || hardwareTopology.verdict === 'error')))) ||
    (setupStage === 2 && (!setupChecks || setupChecks.some(check => check.status === 'error') || !$('setup-approve').checked));
  ['setup-discover', 'setup-attach', 'setup-check', 'setup-connect', 'setup-back', 'setup-resume', 'setup-hardware-scan'].forEach(id => { $(id).disabled = setupBusy || (id === 'setup-hardware-scan' && [...setupSelection.values()].some(node => node.access_status !== 'ready')); });
}
async function setupAction(action) {
  if (setupBusy) return;
  setupBusy = true;
  setupError();
  setupButtons();
  try { await action(); } catch (error) { setupError(error.message); }
  finally { setupBusy = false; setupButtons(); }
}
function invalidateSetup() {
  setupChecks = null;
  hardwareTopology = null;
  $('setup-approve').checked = false;
  $('setup-checks').innerHTML = '<div class="empty-state">Configuration changed. Run readiness checks again.</div>';
  renderHardwareTopology(expectedHardwareTopology());
  setupButtons();
}
function renderSetupCandidates() {
  $('setup-nodes').innerHTML = setupCandidates.length ? setupCandidates.map(node => `<label class="setup-node ${setupSelection.has(node.address) ? 'selected' : ''}"><input type="checkbox" data-select-node="${escapeHtml(node.address)}" ${setupSelection.has(node.address) ? 'checked' : ''} /><span class="setup-node-icon">⬡</span><span><b>${escapeHtml(node.hostname)}</b><small>${escapeHtml(node.address)} · ${escapeHtml(node.source)} · SSH reachable</small><small>${escapeHtml(node.gpu || 'Hardware detected during readiness check')} · host key verified during connection</small></span></label>`).join('') : '<div class="empty-state"><b>No SSH machine found</b><span>Check power, network isolation and SSH. Try an IPv4 address or /24 subnet.</span></div>';
  renderSetupDiscoveryState();
}
function renderSetupRoles() {
  const nodes = [...setupSelection.values()];
  if (!nodes.some(node => node.role === 'head') && nodes.length) nodes[0].role = 'head';
  if (nodes.length === 1) $('setup-topology').value = 'single';
  else if ($('setup-topology').value === 'single') $('setup-topology').value = 'switch';
  if (nodes.length !== 4 && $('setup-topology').value === 'ring') $('setup-topology').value = 'switch';
  adaptSetupForMachines();
  $('setup-roles').innerHTML = nodes.map(node => `<div class="setup-role"><div><b>${escapeHtml(node.hostname)}</b><small>${escapeHtml(node.address)}</small></div><label class="setup-checkbox"><input type="radio" name="setup-head" data-head="${escapeHtml(node.address)}" ${node.role === 'head' ? 'checked' : ''} /> Head</label><span class="setup-access-state ${node.access_status === 'ready' ? 'ready' : ''}">${node.access_status === 'ready' ? `✓ ${escapeHtml(node.username)} · SSH verified` : 'Waiting for connection'}</span>${requiresDgxFabric() ? `<label>Fabric IPv4<input data-fabric="${escapeHtml(node.address)}" aria-label="Fabric IPv4 ${escapeHtml(node.address)}" value="${escapeHtml(node.fabric_address || '')}" placeholder="Auto-detect from Sync" /></label>` : ''}</div>`).join('');
  renderBootstrapAccess();
  renderHardwareTopology(hardwareTopology || expectedHardwareTopology());
}

function renderBootstrapAccess() {
  const container = $('setup-bootstrap-list');
  if (!container) return;
  container.innerHTML = [...setupSelection.values()].map(node => {
    const status = node.access_status || 'creating';
    const title = status === 'ready' ? 'SSH verified automatically' : status === 'checking' ? 'Verifying SSH…' : status === 'error' ? 'Connection needs attention' : status === 'expired' ? 'Command expired' : status === 'creating' ? 'Generating a dedicated key…' : 'Run this command on the machine';
    const detail = status === 'ready' ? `${node.username}@${node.address} · ${node.fingerprint}` : node.access_error || (status === 'checking' ? 'The machine pinged the controller. Checking the host key and dedicated SSH key.' : 'The command expires after 15 minutes and can be replaced safely.');
    const command = node.access_command ? `<div class="setup-command"><code>${escapeHtml(node.access_command)}</code><button class="secondary" type="button" data-copy-bootstrap="${escapeHtml(node.address)}">Copy command</button></div>` : '';
    const regenerate = ['error','expired'].includes(status) ? `<button class="secondary setup-regenerate" type="button" data-regenerate-bootstrap="${escapeHtml(node.address)}">Generate a new command</button>` : '';
    return `<article class="setup-bootstrap-machine ${escapeHtml(status)}"><div class="setup-bootstrap-head"><span>${status === 'ready' ? '✓' : status === 'error' || status === 'expired' ? '!' : '⌁'}</span><div><b>${escapeHtml(node.hostname || node.address)}</b><small>${escapeHtml(title)}</small></div></div>${command}<p>${escapeHtml(detail)}</p>${regenerate}</article>`;
  }).join('');
}

async function createBootstrapAccess(node) {
  node.access_status = 'creating';
  node.access_error = '';
  node.access_command = '';
  renderBootstrapAccess();
  try {
    const enrollment = await api('/api/v1/setup/bootstrap', {method:'POST', headers:setupAuth(), body:JSON.stringify({address:node.address})});
    node.bootstrap_id = enrollment.id;
    node.access_status = enrollment.status;
    node.access_command = enrollment.command || '';
    node.access_error = enrollment.error || '';
    if (enrollment.username) node.username = enrollment.username;
    if (enrollment.fingerprint) node.fingerprint = enrollment.fingerprint;
  } catch (error) {
    node.access_status = 'error';
    node.access_error = error.message;
    renderBootstrapAccess();
    throw error;
  }
}

async function ensureBootstrapAccess() {
  for (const node of setupSelection.values()) if (!node.bootstrap_id) await createBootstrapAccess(node);
  renderSetupRoles();
  setupButtons();
}

async function pollBootstrapAccess() {
  if (setupBootstrapPolling || setupBusy || setupStage !== 1) return;
  const pending = [...setupSelection.values()].filter(node => node.bootstrap_id && !['ready','expired'].includes(node.access_status));
  if (!pending.length) return;
  setupBootstrapPolling = true;
  try {
    for (const node of pending) {
      const enrollment = await api(`/api/v1/setup/bootstrap/${node.bootstrap_id}`, {headers:setupAuth()});
      node.access_status = enrollment.status;
      node.access_error = enrollment.error || '';
      if (enrollment.username) node.username = enrollment.username;
      if (enrollment.fingerprint) node.fingerprint = enrollment.fingerprint;
    }
    renderSetupRoles();
    setupButtons();
  } catch (error) {
    setupError(`Access verification disconnected: ${error.message}`);
  } finally {
    setupBootstrapPolling = false;
  }
}

async function copyBootstrapCommand(node, button) {
  if (!node?.access_command) return;
  try {
    await navigator.clipboard.writeText(node.access_command);
  } catch {
    const input = document.createElement('textarea');
    input.value = node.access_command;
    input.style.position = 'fixed';
    input.style.opacity = '0';
    document.body.appendChild(input);
    input.select();
    document.execCommand('copy');
    input.remove();
  }
  button.textContent = 'Copied';
  setTimeout(() => { button.textContent = 'Copy command'; }, 1200);
}

function expectedHardwareTopology() {
  const nodes = [...setupSelection.values()];
  const topology = $('setup-topology').value;
  const portCount = topology === 'ring' || topology === 'direct' && nodes.length === 3 ? 2 : topology === 'single' ? 0 : 1;
  const hardwareNodes = nodes.map(node => ({address:node.address, hostname:node.hostname, role:node.role || 'worker', status:'pending', ports:Array.from({length:portCount}, (_,index) => ({name:`QSFP ${index + 1}`,status:'unknown',carrier:false,speed_gbps:0,rdma:true,addresses:[]}))}));
  const links = [];
  if (topology === 'switch') for (const node of nodes) links.push({source:node.address,target:'switch',status:'pending',message:'Expected cable to switch'});
  if (topology === 'direct') for (let left=0;left<nodes.length;left++) for (let right=left+1;right<nodes.length;right++) links.push({source:nodes[left].address,target:nodes[right].address,status:'pending',message:'Expected direct cable'});
  if (topology === 'ring' && nodes.length === 4) for (let index=0;index<nodes.length;index++) links.push({source:nodes[index].address,target:nodes[(index+1)%nodes.length].address,status:'pending',message:'Expected ring cable'});
  const expected = topology === 'single' ? 0 : links.length;
  const detail = topology === 'single' ? 'No fabric cable is required for one Spark.' : topology === 'switch' ? `Connect one cable from every Spark to the same ConnectX-7 switch (${expected} cables total).` : topology === 'ring' ? 'Connect each Spark to its two neighbours to close a four-cable ring.' : nodes.length === 3 ? 'Connect every Spark to the other two to form a triangle (3 cables total).' : 'Connect the two Sparks directly with one cable.';
  return {topology,verdict:'pending',expected_cables:expected,detected_cables:0,summary:'Expected physical layout — inspect the machines to verify it.',nodes:hardwareNodes,links,instructions:[{status:'info',title:'How to cable this layout',detail}]};
}

function hardwarePositions(topology) {
  const nodes = topology.nodes;
  const positions = {};
  if (topology.topology === 'switch') {
    const layouts = [[18,24],[82,24],[18,78],[82,78]];
    nodes.forEach((node,index) => { positions[node.address] = layouts[index] || [50,88]; });
    positions.switch = [50,51];
  } else if (nodes.length === 4) {
    const layouts = [[20,22],[80,22],[80,78],[20,78]];
    nodes.forEach((node,index) => { positions[node.address] = layouts[index]; });
  } else if (nodes.length === 3) {
    positions[nodes[0].address] = [50,20]; positions[nodes[1].address] = [20,76]; positions[nodes[2].address] = [80,76];
  } else if (nodes.length === 2) {
    positions[nodes[0].address] = [20,50]; positions[nodes[1].address] = [80,50];
  } else if (nodes.length === 1) positions[nodes[0].address] = [50,50];
  return positions;
}

function renderHardwareTopology(topology) {
  if (!topology || !requiresDgxFabric()) return;
  const positions = hardwarePositions(topology);
  const line = link => {
    const source = positions[link.source], target = positions[link.target];
    if (!source || !target) return '';
    const label = link.speed_gbps ? `${link.speed_gbps}G` : link.status === 'pending' ? 'cable' : 'missing';
    return `<line class="hardware-cable ${link.status}" x1="${source[0]}" y1="${source[1]}" x2="${target[0]}" y2="${target[1]}"/><text class="hardware-cable-label" x="${(source[0]+target[0])/2}" y="${(source[1]+target[1])/2-2}">${escapeHtml(label)}</text>`;
  };
  const device = node => {
    const [x,y] = positions[node.address];
    const ports = node.ports.length ? node.ports.map(port => `<span class="hardware-port ${port.status}"><i></i>${escapeHtml(port.name)}${port.speed_gbps ? ` · ${port.speed_gbps}G` : ''}</span>`).join('') : '<span class="hardware-port connected"><i></i>Management only</span>';
    const status = node.status === 'pass' ? 'Peer links verified' : node.status === 'warning' ? 'Link needs attention' : node.status === 'error' ? 'Cable not detected' : 'Waiting for inspection';
    return `<article class="hardware-device ${node.status}" style="left:${x}%;top:${y}%"><div class="hardware-device-head"><span class="hardware-device-icon">${node.role === 'head' ? 'H' : 'W'}</span><div><b>${escapeHtml(node.hostname)}</b><small>${escapeHtml(node.address)}</small></div></div><div class="hardware-ports">${ports}</div><div class="hardware-device-status">${status}</div></article>`;
  };
  const switchDevice = topology.topology === 'switch' ? `<article class="hardware-device switch ${topology.verdict}" style="left:50%;top:51%"><div class="hardware-device-icon">⇄</div><b>Ethernet switch</b><small>200G · one link per Spark</small></article>` : '';
  $('hardware-canvas').innerHTML = `<svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${topology.links.map(line).join('')}</svg>${topology.nodes.map(device).join('')}${switchDevice}`;
  $('hardware-summary').textContent = topology.summary;
  $('hardware-verdict').className = `pill ${topology.verdict}`;
  $('hardware-verdict').textContent = topology.verdict === 'pass' ? `${topology.detected_cables}/${topology.expected_cables} cables correct` : topology.verdict === 'warning' ? 'Check highlighted link' : topology.verdict === 'error' ? `${topology.detected_cables}/${topology.expected_cables} cables detected` : 'Expected layout';
  $('hardware-instructions').innerHTML = topology.instructions.map((item,index) => `<article class="hardware-instruction ${item.status}"><i>${item.status === 'pass' ? '✓' : item.status === 'error' ? '!' : index + 1}</i><div><b>${escapeHtml(item.title)}</b><p>${escapeHtml(item.detail)}</p></div></article>`).join('');
}
function renderSetupChecks(checks, target) {
  $(target).innerHTML = checks.map(check => `<article class="setup-diagnostic ${check.status}"><span>${check.status === 'pass' ? '✓' : '!'}</span><div><b>${escapeHtml(check.message)}</b><small>${escapeHtml(check.node)} · ${escapeHtml(check.code)}${check.automatic ? ' · automatic repair during installation' : ''}</small>${check.remedy ? `<p>${escapeHtml(check.remedy)}</p>` : ''}</div></article>`).join('');
}
function renderSetupJob(job) {
  setupJob = job;
  if (job.hardware_topology) hardwareTopology = job.hardware_topology;
  $('setup-job-title').textContent = job.mode === 'demo' ? 'Simulated compute pool installation' : 'Installing compute pool services';
  $('setup-job-status').textContent = job.status;
  $('setup-steps').innerHTML = job.steps.map((step, index) => `<div class="setup-step ${step.status}"><i>${step.status === 'done' ? '✓' : step.status === 'failed' ? '!' : index + 1}</i><div><b>${escapeHtml(step.label)}</b><small>${escapeHtml(step.message || step.status)}</small></div></div>`).join('');
  $('setup-logs').textContent = job.logs.map(log => `${new Date(log.timestamp).toLocaleTimeString()}  ${log.node}\n${log.message}`).join('\n\n');
  $('setup-logs').scrollTop = $('setup-logs').scrollHeight;
  const issues = job.checks.filter(check => check.status !== 'pass');
  renderSetupChecks(issues, 'setup-job-checks');
  $('setup-recovery').hidden = !['failed', 'blocked'].includes(job.status);
  $('setup-job-error').textContent = job.error || '';
  if (job.status === 'ready') {
    $('setup-open').href = job.dashboard_url || '#models';
    $('setup-open').textContent = job.mode === 'demo' ? 'Explore demo model launcher →' : 'Open pool dashboard →';
    document.querySelector('.setup-ready h3').textContent = job.mode === 'demo' ? 'Demo setup complete' : 'Your compute pool is ready';
  }
  setupNavigate(job.status === 'ready' ? 4 : 3);
}
async function loadSetup(restore = false, promptForAdmin = true) {
  setupState = await api('/api/v1/setup', {headers: setupAuth(), promptForAdmin});
  const connected = setupState.connected_cluster;
  $('setup-mode').textContent = connected ? `CONTROLLER · ${connected.cluster_name}` : setupState.mode === 'demo' ? 'DEMO · simulated machines' : setupState.enabled ? 'LIVE · SSH bootstrap' : 'Setup disabled';
  $('setup-access').textContent = connected ? `${connected.cluster_name} connected.` : setupState.enabled ? (setupState.mode === 'demo' ? 'Simulated discovery and installation. No machine is modified.' : 'Ready to configure machines.') : 'Token required to make changes.';
  if (restore && setupState.jobs.length) {
    const job = setupState.jobs[0];
    $('setup-name').value = job.request.name;
    $('setup-topology').value = job.request.topology;
    setupSelection = new Map(job.request.nodes.map(node => [node.address, {...node, hostname: node.address}]));
    renderSetupRoles();
    renderSetupJob(job);
  } else if (setupJob) {
    const job = setupState.jobs.find(item => item.id === setupJob.id);
    if (job) renderSetupJob(job);
  }
  setupButtons();
}
$('setup-connect').addEventListener('click', () => setupAction(() => loadSetup(true)));
$('setup-attach').addEventListener('click', () => setupAction(async () => {
  const address = $('setup-existing-address').value.trim();
  if (!address) throw new Error('Enter the management IPv4 of the pool controller.');
  const connection = await api('/api/v1/setup/connect', {method:'POST', headers:setupAuth(), body:JSON.stringify({address, port:Number($('setup-existing-port').value)})});
  await loadSetup();
  $('discovery-note').textContent = `Connected to ${connection.cluster_name} through ${connection.url}. Pool data and commands now use this controller.`;
  location.hash = '#overview';
}));
$('setup-discover').addEventListener('click', () => setupAction(async () => {
  if (!setupState) await loadSetup();
  setupDiscoveryState = 'scanning';
  $('discovery-note').textContent = 'Discovering LAN services and checking SSH…';
  renderSetupDiscoveryState();
  let result;
  try {
    result = await api('/api/v1/setup/discover', {method:'POST', headers:setupAuth(), body:JSON.stringify({subnet:$('setup-subnet').value.trim() || null, addresses:$('setup-address').value.trim() ? [$('setup-address').value.trim()] : []})});
  } catch (error) {
    setupDiscoveryState = 'complete';
    $('discovery-note').textContent = 'Scan stopped before any machines were found.';
    renderSetupCandidates();
    throw error;
  }
  setupCandidates = result.nodes;
  // Keep selected machines visible through subsequent discovery attempts.
  for (const node of setupSelection.values()) if (!setupCandidates.some(item => item.address === node.address)) setupCandidates.push(node);
  setupDiscoveryState = 'complete';
  $('discovery-note').textContent = result.notes.join(' ');
  renderSetupCandidates();
}));
$('setup-nodes').addEventListener('change', event => {
  const address = event.target.dataset.selectNode;
  if (!address) return;
  if (event.target.checked) setupSelection.set(address, {...setupCandidates.find(node => node.address === address), username:'', role:'worker', access_status:'pending'});
  else setupSelection.delete(address);
  invalidateSetup();
  renderSetupCandidates();
  setupNavigate(0);
});
$('setup-roles').addEventListener('input', event => {
  const target = event.target;
  if (target.dataset.head) for (const node of setupSelection.values()) node.role = node.address === target.dataset.head ? 'head' : 'worker';
  if (target.dataset.fabric) setupSelection.get(target.dataset.fabric).fabric_address = target.value.trim();
  invalidateSetup();
});
$('setup-name').addEventListener('input', invalidateSetup);
$('setup-topology').addEventListener('input', () => {
  adaptSetupForMachines();
  invalidateSetup();
  renderSetupRoles();
});
$('setup-approve').addEventListener('change', setupButtons);
$('setup-bootstrap-list').addEventListener('click', event => {
  const copyAddress = event.target.dataset.copyBootstrap;
  if (copyAddress) {
    copyBootstrapCommand(setupSelection.get(copyAddress), event.target);
    return;
  }
  const regenerateAddress = event.target.dataset.regenerateBootstrap;
  if (regenerateAddress) setupAction(async () => {
    const node = setupSelection.get(regenerateAddress);
    node.bootstrap_id = null;
    await createBootstrapAccess(node);
    renderSetupRoles();
  });
});
$('setup-hardware-scan').addEventListener('click', () => setupAction(async () => {
  $('setup-hardware-scan').textContent = 'Inspecting ports…';
  try {
    const result = await api('/api/v1/setup/hardware', {method:'POST', headers:setupAuth(), body:JSON.stringify(setupPayload())});
    setupChecks = result.checks;
    hardwareTopology = result.topology;
    for (const node of result.nodes) Object.assign(setupSelection.get(node.address), node);
    renderSetupRoles();
    renderSetupChecks(setupChecks, 'setup-checks');
  } finally {
    $('setup-hardware-scan').textContent = hardwareTopology ? 'Inspect again' : 'Inspect cables';
  }
}));
$('setup-back').addEventListener('click', () => { setupError(); setupNavigate(setupStage - 1); });
$('setup-next').addEventListener('click', () => setupAction(async () => {
  if (setupStage === 0) { renderSetupRoles(); setupNavigate(1); await ensureBootstrapAccess(); }
  else if (setupStage === 1) {
    $('setup-review').textContent = `${$('setup-name').value}: ${[...setupSelection.values()].map(node => `${node.username}@${node.address} (${node.role})`).join(', ')}.`;
    setupNavigate(2);
  } else if (setupStage === 2) {
    const job = await api('/api/v1/setup/jobs', {method:'POST', headers:setupAuth(), body:JSON.stringify(setupPayload())});
    renderSetupJob(job);
  }
}));
$('setup-check').addEventListener('click', () => setupAction(async () => {
  $('setup-checks').textContent = 'Checking selected machines…';
  const result = await api('/api/v1/setup/check', {method:'POST', headers:setupAuth(), body:JSON.stringify(setupPayload())});
  setupChecks = result.checks;
  hardwareTopology = result.topology;
  for (const node of result.nodes) Object.assign(setupSelection.get(node.address), node);
  renderSetupRoles();
  renderHardwareTopology(hardwareTopology);
  renderSetupChecks(setupChecks, 'setup-checks');
}));
$('setup-resume').addEventListener('click', () => setupAction(async () => {
  const job = await api(`/api/v1/setup/jobs/${setupJob.id}/resume`, {method:'POST', headers:setupAuth()});
  renderSetupJob(job);
}));
function reviseSetup() {
  setupJob = null;
  invalidateSetup();
  renderSetupRoles();
  setupNavigate(1);
}
$('setup-revise').addEventListener('click', reviseSetup);
$('setup-new').addEventListener('click', () => { setupJob = null; setupSelection.clear(); setupCandidates = []; setupDiscoveryState = 'idle'; invalidateSetup(); renderSetupCandidates(); setupNavigate(0); });
loadSetup(true, false).catch(error => {
  if (error.message === 'Invalid model administration token') {
    $('setup-mode').textContent = 'ADMIN TOKEN REQUIRED';
    $('setup-access').textContent = 'Enter the controller token to make changes.';
    return;
  }
  setupError(error.message);
});
setInterval(async () => {
  if (setupPolling || setupBusy || !setupJob || ['ready','failed','blocked'].includes(setupJob.status)) return;
  setupPolling = true;
  try { await loadSetup(); } catch (error) { setupError(`Progress disconnected: ${error.message}. Reconnect to recover saved setup.`); }
  finally { setupPolling = false; }
}, 2000);
setInterval(pollBootstrapAccess, 1500);
setupNavigate(0);

let setupSheetOpener = null;
let setupSheetCloseTimer = null;
function openSetupSheet(opener) {
  const wasHidden = $('setup-sheet').hidden;
  clearTimeout(setupSheetCloseTimer);
  setupSheetOpener = opener || setupSheetOpener;
  $('setup-sheet').hidden = false;
  document.body.classList.add('bottom-sheet-open');
  requestAnimationFrame(() => {
    $('setup-sheet').classList.add('open');
    if (wasHidden) $('setup-sheet-close').focus({preventScroll: true});
  });
}
function closeSetupSheet() {
  if ($('setup-sheet').hidden) return;
  $('setup-sheet').classList.remove('open');
  document.body.classList.remove('bottom-sheet-open');
  const opener = setupSheetOpener;
  setupSheetCloseTimer = setTimeout(() => {
    $('setup-sheet').hidden = true;
    if (opener?.isConnected) opener.focus({preventScroll: true});
  }, 220);
}
$('setup-open-sheet').addEventListener('click', event => openSetupSheet(event.currentTarget));
$('setup-sheet-close').addEventListener('click', closeSetupSheet);
$('setup-sheet-backdrop').addEventListener('click', closeSetupSheet);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !$('setup-sheet').hidden) closeSetupSheet();
});
window.addEventListener('hashchange', closeSetupSheet);
