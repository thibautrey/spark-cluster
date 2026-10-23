// Uses the shared dashboard API helper. Credentials stay in the current DOM only.
let setupStage = 0;
let setupState = null;
let setupCandidates = [];
let setupSelection = new Map();
let setupChecks = null;
let setupJob = null;
let setupBusy = false;
let setupPolling = false;
let hardwareTopology = null;
const setupAuth = () => $('setup-token').value.trim() ? {Authorization: `Bearer ${$('setup-token').value.trim()}`} : {};
const setupPayload = () => ({name: $('setup-name').value.trim(), topology: $('setup-topology').value, nodes: [...setupSelection.values()].map(node => ({address: node.address, username: node.username || 'nvidia', fingerprint: node.fingerprint, role: node.role || 'worker', fabric_address: node.fabric_address || null}))});

function setupError(message = '') {
  $('setup-error').hidden = !message;
  $('setup-error').textContent = message;
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
  $('setup-next').textContent = ['Choose machines →', 'Check readiness →', 'Install cluster →'][stage] || '';
  $('setup-summary').textContent = `${setupSelection.size} machine${setupSelection.size === 1 ? '' : 's'} selected${setupJob ? ` · ${setupJob.status}` : ''}`;
  setupButtons();
}
function setupButtons() {
  $('setup-next').disabled = setupBusy || !setupState?.enabled || !setupSelection.size ||
    (setupStage === 1 && (!$('setup-trust').checked || (setupSelection.size > 1 && (!hardwareTopology || hardwareTopology.verdict === 'error')))) ||
    (setupStage === 2 && (!setupChecks || setupChecks.some(check => check.status === 'error') || !$('setup-approve').checked));
  ['setup-discover', 'setup-attach', 'setup-check', 'setup-connect', 'setup-back', 'setup-resume', 'setup-hardware-scan'].forEach(id => { $(id).disabled = setupBusy || (id === 'setup-hardware-scan' && !$('setup-trust').checked); });
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
  $('setup-nodes').innerHTML = setupCandidates.length ? setupCandidates.map(node => `<label class="setup-node ${setupSelection.has(node.address) ? 'selected' : ''}"><input type="checkbox" data-select-node="${escapeHtml(node.address)}" ${setupSelection.has(node.address) ? 'checked' : ''} /><span class="setup-node-icon">⬡</span><span><b>${escapeHtml(node.hostname)}</b><small>${escapeHtml(node.address)} · ${escapeHtml(node.source)} · SSH reachable</small><code>${escapeHtml(node.fingerprint)}</code><small>${escapeHtml(node.gpu)}</small></span></label>`).join('') : '<div class="empty-state"><b>No SSH machine found</b><span>Check power, network isolation and SSH. Try an IPv4 address or /24 subnet.</span></div>';
}
function renderSetupRoles() {
  const nodes = [...setupSelection.values()];
  if (!nodes.some(node => node.role === 'head') && nodes.length) nodes[0].role = 'head';
  if (nodes.length === 1) $('setup-topology').value = 'single';
  else if ($('setup-topology').value === 'single') $('setup-topology').value = 'switch';
  if (nodes.length !== 4 && $('setup-topology').value === 'ring') $('setup-topology').value = 'switch';
  $('setup-roles').innerHTML = nodes.map(node => `<div class="setup-role"><div><b>${escapeHtml(node.hostname)}</b><small>${escapeHtml(node.address)}</small></div><label class="setup-checkbox"><input type="radio" name="setup-head" data-head="${escapeHtml(node.address)}" ${node.role === 'head' ? 'checked' : ''} /> Head</label><label>SSH user<input data-user="${escapeHtml(node.address)}" aria-label="SSH user ${escapeHtml(node.address)}" value="${escapeHtml(node.username || 'nvidia')}" pattern="[a-z_][a-z0-9_-]{0,31}" /></label><label>Fabric IPv4<input data-fabric="${escapeHtml(node.address)}" aria-label="Fabric IPv4 ${escapeHtml(node.address)}" value="${escapeHtml(node.fabric_address || '')}" placeholder="Auto-detect from Sync" /></label></div>`).join('');
  renderHardwareTopology(hardwareTopology || expectedHardwareTopology());
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
  if (!topology) return;
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
  $('setup-job-title').textContent = job.mode === 'demo' ? 'Simulated cluster installation' : 'Installing cluster services';
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
    $('setup-open').textContent = job.mode === 'demo' ? 'Explore demo model launcher →' : 'Open cluster dashboard →';
    document.querySelector('.setup-ready h3').textContent = job.mode === 'demo' ? 'Demo setup complete' : 'Your cluster services are ready';
  }
  setupNavigate(job.status === 'ready' ? 4 : 3);
}
async function loadSetup(restore = false) {
  setupState = await api('/api/v1/setup', {headers: setupAuth()});
  const connected = setupState.connected_cluster;
  $('setup-mode').textContent = connected ? `CONTROLLER · ${connected.cluster_name}` : setupState.mode === 'demo' ? 'DEMO · simulated machines' : setupState.enabled ? 'LIVE · SSH bootstrap' : 'Setup disabled';
  $('setup-access').textContent = connected ? `This dashboard controls ${connected.cluster_name} through ${connected.url}.` : setupState.enabled ? (setupState.mode === 'demo' ? 'Simulated discovery and installation. No machine is modified.' : 'Authenticated bootstrap connected. Use a trusted LAN.') : 'Enter a bridge address above, or enable SETUP_ENABLED to configure new machines.';
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
  if (!address) throw new Error('Enter the management IPv4 of the cluster head.');
  const connection = await api('/api/v1/setup/connect', {method:'POST', headers:setupAuth(), body:JSON.stringify({address, port:Number($('setup-existing-port').value)})});
  await loadSetup();
  $('discovery-note').textContent = `Connected to ${connection.cluster_name} through ${connection.url}. Cluster data and commands now use this controller.`;
  location.hash = '#overview';
}));
$('setup-discover').addEventListener('click', () => setupAction(async () => {
  if (!setupState) await loadSetup();
  $('discovery-note').textContent = 'Discovering LAN services and checking SSH…';
  const result = await api('/api/v1/setup/discover', {method:'POST', headers:setupAuth(), body:JSON.stringify({subnet:$('setup-subnet').value.trim() || null, addresses:$('setup-address').value.trim() ? [$('setup-address').value.trim()] : []})});
  setupCandidates = result.nodes;
  // Keep selected machines visible through subsequent discovery attempts.
  for (const node of setupSelection.values()) if (!setupCandidates.some(item => item.address === node.address)) setupCandidates.push(node);
  renderSetupCandidates();
  $('discovery-note').textContent = result.notes.join(' ');
}));
$('setup-nodes').addEventListener('change', event => {
  const address = event.target.dataset.selectNode;
  if (!address) return;
  if (event.target.checked) setupSelection.set(address, {...setupCandidates.find(node => node.address === address), username:'nvidia', role:'worker'});
  else setupSelection.delete(address);
  $('setup-trust').checked = false;
  invalidateSetup();
  renderSetupCandidates();
  setupNavigate(0);
});
$('setup-roles').addEventListener('input', event => {
  const target = event.target;
  if (target.dataset.head) for (const node of setupSelection.values()) node.role = node.address === target.dataset.head ? 'head' : 'worker';
  if (target.dataset.user) setupSelection.get(target.dataset.user).username = target.value;
  if (target.dataset.fabric) setupSelection.get(target.dataset.fabric).fabric_address = target.value.trim();
  invalidateSetup();
});
$('setup-name').addEventListener('input', invalidateSetup);
$('setup-topology').addEventListener('input', () => {
  const topology = $('setup-topology').value;
  $('topology-choice-note').textContent = topology === 'switch'
    ? 'Best decode scaling and required for independent model pools. NVIDIA measured 36.34 tok/s on four Sparks with RoCEnante versus 32.82 tok/s on a physical ring for Qwen3.8-27B NVFP4.'
    : topology === 'ring'
      ? 'Official switchless four-Spark layout. NCCL follows the ring directly; choose a switch later for any-to-any traffic or independent model pools.'
      : topology === 'direct'
        ? 'Direct point-to-point fabric for two Sparks, or a three-Spark triangle.'
        : 'No fabric cable is required for a standalone Spark.';
  invalidateSetup();
  renderSetupRoles();
});
['setup-trust','setup-approve'].forEach(id => $(id).addEventListener('change', setupButtons));
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
  if (setupStage === 0) { renderSetupRoles(); setupNavigate(1); }
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
$('setup-new').addEventListener('click', () => { setupJob = null; setupSelection.clear(); setupCandidates = []; invalidateSetup(); renderSetupCandidates(); setupNavigate(0); });
loadSetup(true).catch(error => {
  if (error.message === 'Invalid model administration token') {
    $('setup-mode').textContent = 'ADMIN TOKEN REQUIRED';
    $('setup-access').textContent = 'Enter the administration token to view or change controller setup. Cluster metrics will appear after a bridge is attached.';
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
setupNavigate(0);
