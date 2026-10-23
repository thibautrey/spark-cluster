// Uses the shared dashboard API helper. Credentials stay in the current DOM only.
let setupStage = 0;
let setupState = null;
let setupCandidates = [];
let setupSelection = new Map();
let setupChecks = null;
let setupJob = null;
let setupBusy = false;
let setupPolling = false;
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
    (setupStage === 1 && !$('setup-trust').checked) ||
    (setupStage === 2 && (!setupChecks || setupChecks.some(check => check.status === 'error') || !$('setup-approve').checked));
  ['setup-discover', 'setup-attach', 'setup-check', 'setup-connect', 'setup-back', 'setup-resume'].forEach(id => { $(id).disabled = setupBusy; });
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
  $('setup-approve').checked = false;
  $('setup-checks').innerHTML = '<div class="empty-state">Configuration changed. Run readiness checks again.</div>';
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
  $('setup-roles').innerHTML = nodes.map(node => `<div class="setup-role"><div><b>${escapeHtml(node.hostname)}</b><small>${escapeHtml(node.address)}</small></div><label class="setup-checkbox"><input type="radio" name="setup-head" data-head="${escapeHtml(node.address)}" ${node.role === 'head' ? 'checked' : ''} /> Head</label><label>SSH user<input data-user="${escapeHtml(node.address)}" aria-label="SSH user ${escapeHtml(node.address)}" value="${escapeHtml(node.username || 'nvidia')}" pattern="[a-z_][a-z0-9_-]{0,31}" /></label><label>Fabric IPv4<input data-fabric="${escapeHtml(node.address)}" aria-label="Fabric IPv4 ${escapeHtml(node.address)}" value="${escapeHtml(node.fabric_address || '')}" placeholder="Auto-detect from Sync" /></label></div>`).join('');
}
function renderSetupChecks(checks, target) {
  $(target).innerHTML = checks.map(check => `<article class="setup-diagnostic ${check.status}"><span>${check.status === 'pass' ? '✓' : '!'}</span><div><b>${escapeHtml(check.message)}</b><small>${escapeHtml(check.node)} · ${escapeHtml(check.code)}${check.automatic ? ' · automatic repair during installation' : ''}</small>${check.remedy ? `<p>${escapeHtml(check.remedy)}</p>` : ''}</div></article>`).join('');
}
function renderSetupJob(job) {
  setupJob = job;
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
['setup-name','setup-topology'].forEach(id => $(id).addEventListener('input', invalidateSetup));
['setup-trust','setup-approve'].forEach(id => $(id).addEventListener('change', setupButtons));
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
  for (const node of result.nodes) Object.assign(setupSelection.get(node.address), node);
  renderSetupRoles();
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
    $('setup-access').textContent = 'Cluster data is available. Enter the administration token to view or change controller setup.';
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
