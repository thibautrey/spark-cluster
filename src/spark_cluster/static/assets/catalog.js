/* Native Hugging Face browser. All Hub requests go through the controller. */
let catalogModels = [];
let catalogCursor = null;
let catalogSelection = null;
let catalogRequest = 0;
let detailRequest = 0;
let catalogTimer;
let catalogJobs = [];

function catalogState(id) {
  const job = catalogJobs.find(item => item.repo_id === id && !['failed', 'ready'].includes(item.status));
  if (job) return job.status.replaceAll('_', ' ');
  if (localModels.some(item => item.id === id)) return 'on cluster';
  if (catalogJobs.some(item => item.repo_id === id && item.status === 'ready')) return 'downloaded';
  return '';
}

function renderCatalogCards() {
  $('catalog-grid').innerHTML = catalogModels.length ? catalogModels.map(model => `
    <button type="button" class="catalog-card ${catalogSelection === model.id ? 'selected' : ''}" data-repo="${escapeHtml(model.id)}" aria-pressed="${catalogSelection === model.id}">
      <span class="catalog-card-head"><span class="catalog-avatar">${escapeHtml(model.author[0].toUpperCase())}</span><span class="catalog-author">${escapeHtml(model.author)}</span>${model.gated ? '<span class="catalog-gated">License</span>' : ''}</span>
      <b>${escapeHtml(model.id.split('/').pop())}</b>
      <span class="catalog-tags">${[model.task, model.library, model.parameters ? `${formatNumber(model.parameters)} parameters` : null].filter(Boolean).map(tag => `<span>${escapeHtml(tag)}</span>`).join('')}</span>
      <span class="catalog-card-foot"><span>↓ ${formatNumber(model.downloads)}</span><span>♡ ${formatNumber(model.likes)}</span>${model.updated ? `<span>${escapeHtml(new Date(model.updated).toLocaleDateString())}</span>` : ''}<span class="catalog-card-state">${escapeHtml(catalogState(model.id))}</span></span>
    </button>`).join('') : '<div class="catalog-empty">No models match these filters.<br>Try another search or reset the filters.</div>';
}

async function loadCatalog(append = false) {
  const request = ++catalogRequest;
  clearTimeout(catalogTimer);
  $('catalog-status').textContent = append ? 'Loading more models…' : 'Searching Hugging Face…';
  $('catalog-grid').setAttribute('aria-busy', 'true');
  $('catalog-more').disabled = true;
  $('catalog-retry').hidden = true;
  if (!append) {
    catalogModels = [];
    catalogCursor = null;
    $('catalog-grid').innerHTML = '<div class="catalog-empty">Discovering models…</div>';
    $('catalog-more').hidden = true;
  }
  const params = new URLSearchParams({query: $('catalog-query').value.trim(), sort: $('catalog-sort').value, task: $('catalog-task').value, library: $('catalog-library').value, access: $('catalog-access').value});
  if (append && catalogCursor) params.set('cursor', catalogCursor);
  try {
    const page = await api(`/api/v1/catalog/models?${params}`);
    if (request !== catalogRequest) return;
    const existing = new Set(catalogModels.map(model => model.id));
    catalogModels.push(...page.models.filter(model => !existing.has(model.id)));
    catalogCursor = page.cursor;
    renderCatalogCards();
    $('catalog-status').textContent = `${catalogModels.length} models${catalogCursor ? ' · more available' : ''} · Hugging Face Hub`;
    $('catalog-more').hidden = !catalogCursor;
  } catch (error) {
    if (request !== catalogRequest) return;
    $('catalog-status').textContent = error.message;
    $('catalog-retry').hidden = false;
    if (!append) $('catalog-grid').innerHTML = '<div class="catalog-empty">The catalog could not be loaded. Retry, or use a repository ID.</div>';
  } finally {
    if (request === catalogRequest) {
      $('catalog-grid').setAttribute('aria-busy', 'false');
      $('catalog-more').disabled = false;
    }
  }
}

function resetModelFit() {
  ++fitRequestVersion;
  currentFit = null;
  $('check-fit').disabled = false;
  $('check-fit').textContent = 'Check fit';
  $('download-model').disabled = true;
  $('download-model').textContent = 'Download once';
  $('fit-result').className = 'fit-result neutral';
  $('fit-result').innerHTML = '<span class="fit-symbol">?</span><div><b>Run an intelligent fit check</b><small>Architecture, memory tiers, engine support, and storage.</small></div>';
  $('fit-details').hidden = true;
}

async function inspectCatalogModel(repo, revision = 'main') {
  const request = ++detailRequest;
  catalogSelection = repo;
  $('repo-id').value = repo;
  $('revision').value = revision;
  resetModelFit();
  renderCatalogCards();
  $('catalog-placeholder').hidden = true;
  $('catalog-detail').hidden = false;
  $('model-form').hidden = true;
  $('manual-repo-label').hidden = true;
  $('catalog-detail').innerHTML = '<div class="catalog-detail-head"><p>Inspecting repository files…</p></div>';
  if (window.innerWidth <= 800) $('catalog-inspector').scrollIntoView({behavior: 'smooth', block: 'start'});
  try {
    const model = await api(`/api/v1/catalog/models/${repo.split('/').map(encodeURIComponent).join('/')}?revision=${encodeURIComponent(revision)}`);
    if (request !== detailRequest) return;
    const facts = [['Repository size', `${model.size_complete ? '' : 'At least '}${formatBytes(model.repository_bytes)}`], ['Parameters', model.parameters ? formatNumber(model.parameters) : 'Not reported'], ['Architecture', model.architecture || model.library || 'Not reported'], ['License', model.license || 'See model license']];
    $('catalog-detail').innerHTML = `<div class="catalog-detail-head"><p class="eyebrow">${escapeHtml(model.author)}</p><h3>${escapeHtml(model.id.split('/').pop())}</h3><div class="catalog-tags">${[model.task, model.library].filter(Boolean).map(tag => `<span>${escapeHtml(tag)}</span>`).join('')}</div><dl class="catalog-facts">${facts.map(([title, value]) => `<div><dt>${title}</dt><dd>${escapeHtml(value)}</dd></div>`).join('')}</dl>${model.gated ? '<p class="catalog-gated">License approval required. Accept access on Hugging Face and set SPARK_CLUSTER_HF_TOKEN on the controller.</p>' : ''}<p>↓ ${formatNumber(model.downloads)} downloads · ♡ ${formatNumber(model.likes)} likes</p><a href="https://huggingface.co/${model.id.split('/').map(encodeURIComponent).join('/')}" target="_blank" rel="noopener noreferrer">Model card &amp; license ↗</a></div><details class="catalog-files"><summary>${model.files.length} repository files · preview</summary><div class="catalog-file-list">${model.files.map(file => `<div class="catalog-file"><span>${escapeHtml(file.name)}</span><span>${file.size === null ? 'Unknown size' : formatBytes(file.size)}</span></div>`).join('')}</div></details>`;
    $('model-form').hidden = false;
  } catch (error) {
    if (request !== detailRequest) return;
    $('catalog-detail').innerHTML = `<div class="catalog-detail-head"><h3>${escapeHtml(repo)}</h3><p>${escapeHtml(error.message)}</p><button id="catalog-detail-retry" type="button" class="secondary">Retry inspection</button></div>`;
    $('catalog-detail-retry').addEventListener('click', () => inspectCatalogModel(repo, revision));
  }
}

$('catalog-grid').addEventListener('click', event => {
  const button = event.target.closest('[data-repo]');
  if (button) inspectCatalogModel(button.dataset.repo);
});
$('catalog-query').addEventListener('input', () => {
  ++catalogRequest; // Invalidate requests immediately, including during debounce.
  clearTimeout(catalogTimer);
  $('catalog-more').hidden = true;
  catalogTimer = setTimeout(() => loadCatalog(), 350);
});
$('catalog-query').addEventListener('keydown', event => {
  if (event.key === 'Enter') { event.preventDefault(); loadCatalog(); }
});
['catalog-sort', 'catalog-task', 'catalog-library', 'catalog-access'].forEach(id => $(id).addEventListener('change', () => loadCatalog()));
$('catalog-reset').addEventListener('click', () => {
  $('catalog-query').value = '';
  $('catalog-sort').value = 'trendingScore';
  $('catalog-task').value = 'text-generation';
  $('catalog-library').value = '';
  $('catalog-access').value = '';
  loadCatalog();
});
$('catalog-retry').addEventListener('click', () => loadCatalog());
$('catalog-more').addEventListener('click', () => loadCatalog(true));
$('catalog-manual').addEventListener('click', () => {
  ++detailRequest;
  catalogSelection = null;
  resetModelFit();
  $('repo-id').value = $('catalog-query').value.includes('/') ? $('catalog-query').value.trim() : '';
  $('revision').value = 'main';
  $('catalog-placeholder').hidden = true;
  $('catalog-detail').hidden = true;
  $('model-form').hidden = false;
  $('manual-repo-label').hidden = false;
  renderCatalogCards();
  $('repo-id').focus();
});
$('revision').addEventListener('change', () => {
  if (catalogSelection) inspectCatalogModel(catalogSelection, $('revision').value.trim() || 'main');
});
document.addEventListener('library-updated', event => {
  const previous = JSON.stringify(catalogModels.map(model => catalogState(model.id)));
  catalogJobs = event.detail;
  if (previous !== JSON.stringify(catalogModels.map(model => catalogState(model.id)))) renderCatalogCards();
});
loadCatalog();
