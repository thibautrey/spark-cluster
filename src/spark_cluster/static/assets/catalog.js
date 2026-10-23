/* Native Hugging Face browser. All Hub requests go through the controller. */
let catalogModels = [];
let catalogCursor = null;
let catalogSelection = null;
let catalogRequest = 0;
let detailRequest = 0;
let catalogTimer;
let catalogJobs = [];
let catalogFits = new Map();
let catalogFitRequest = 0;
let catalogCapacityLoading = false;
let catalogBaseStatus = '';
let catalogSheetOpener = null;
let catalogSheetCloseTimer = null;

function openCatalogSheet(opener, title = 'Model details') {
  const wasHidden = $('catalog-sheet').hidden;
  clearTimeout(catalogSheetCloseTimer);
  if (opener) catalogSheetOpener = opener;
  $('catalog-sheet-title').textContent = title;
  $('catalog-sheet').hidden = false;
  document.body.classList.add('bottom-sheet-open');
  requestAnimationFrame(() => {
    $('catalog-sheet').classList.add('open');
    if (wasHidden) $('catalog-sheet-close').focus({preventScroll: true});
  });
}

function closeCatalogSheet() {
  if ($('catalog-sheet').hidden) return;
  ++detailRequest;
  $('catalog-sheet').classList.remove('open');
  document.body.classList.remove('bottom-sheet-open');
  const opener = catalogSheetOpener;
  catalogSheetCloseTimer = setTimeout(() => {
    $('catalog-sheet').hidden = true;
    if (opener?.isConnected) opener.focus({preventScroll: true});
  }, 220);
}

function catalogState(id) {
  const job = catalogJobs.find(item => item.repo_id === id && !['failed', 'completed'].includes(item.status));
  if (job) return job.status.replaceAll('_', ' ');
  if (localModels.some(item => item.id === id)) return 'on cluster';
  if (catalogJobs.some(item => item.repo_id === id && item.status === 'completed')) return 'downloaded';
  return '';
}

function catalogMonogram(model) {
  const name = model.id.split('/').pop().replace(/[^A-Za-z0-9]+/g, ' ').trim();
  const initials = name.split(/\s+/).slice(0, 2).map(part => part[0]).join('');
  return (initials || model.author[0] || 'HF').toUpperCase();
}

function catalogImage(model, detail = false) {
  return `<span class="catalog-media ${detail ? 'catalog-detail-media' : ''}">
    <span class="catalog-image-fallback" aria-hidden="true"><i>${escapeHtml(catalogMonogram(model))}</i><small>Hugging Face model</small></span>
    <img src="${escapeHtml(model.thumbnail_url)}" alt="Hugging Face preview for ${escapeHtml(model.id)}" ${detail ? '' : 'loading="lazy"'} decoding="async" referrerpolicy="no-referrer">
    <span class="catalog-media-source">Hugging Face</span>
    ${model.gated ? '<span class="catalog-media-badge gated">License required</span>' : ''}
    ${catalogState(model.id) ? `<span class="catalog-media-badge state">${escapeHtml(catalogState(model.id))}</span>` : ''}
  </span>`;
}

function selectedCapacity() {
  return Number($('catalog-capacity').value || 0);
}

function fitBadge(model) {
  const fit = catalogFits.get(model.id);
  if (!fit) return '';
  const label = fit.memory_fits ? 'Full memory' : fit.placement_strategy.replaceAll('-', ' ');
  const engine = fit.launch_supported ? '' : ' · engine required';
  return `<span class="catalog-fit-badge"><i>✓</i>${escapeHtml(label)} · ${fit.minimum_nodes} Spark${fit.minimum_nodes === 1 ? '' : 's'}${engine}</span>`;
}

function visibleCatalogModels() {
  if (!selectedCapacity()) return catalogModels;
  return catalogModels.filter(model => catalogFits.get(model.id)?.compatible);
}

function renderCatalogCards() {
  const visibleModels = visibleCatalogModels();
  if (catalogCapacityLoading) {
    $('catalog-grid').innerHTML = '<div class="catalog-empty catalog-analyzing"><i></i><b>Calculating model capacity…</b><span>Inspecting repository precision, weights, KV cache, and known hybrid placements.</span></div>';
    return;
  }
  $('catalog-grid').innerHTML = visibleModels.length ? visibleModels.map(model => `
    <button type="button" class="catalog-card ${catalogSelection === model.id ? 'selected' : ''}" data-repo="${escapeHtml(model.id)}" aria-pressed="${catalogSelection === model.id}">
      ${catalogImage(model)}
      <span class="catalog-card-content">
        <span class="catalog-card-head"><span class="catalog-avatar">${escapeHtml(model.author[0].toUpperCase())}</span><span class="catalog-author">${escapeHtml(model.author)}</span><span class="catalog-card-arrow">↗</span></span>
        <b class="catalog-card-title">${escapeHtml(model.id.split('/').pop())}</b>
        <span class="catalog-tags">${[model.task, model.library, model.parameters ? `${formatNumber(model.parameters)} params` : null].filter(Boolean).slice(0, 3).map(tag => `<span>${escapeHtml(tag)}</span>`).join('')}</span>
        ${fitBadge(model)}
        <span class="catalog-card-foot"><span title="Downloads">↓ ${formatNumber(model.downloads)}</span><span title="Likes">♡ ${formatNumber(model.likes)}</span>${model.updated ? `<span class="catalog-updated">${escapeHtml(new Date(model.updated).toLocaleDateString())}</span>` : ''}</span>
      </span>
    </button>`).join('') : `<div class="catalog-empty">${selectedCapacity() ? `No analyzed model fits ${selectedCapacity()} Spark${selectedCapacity() === 1 ? '' : 's'} at this context.<br>Try more machines, a shorter context, or load more models.` : 'No models match these filters.<br>Try another search or reset the filters.'}</div>`;
}

async function applyCapacityFilter() {
  const sparkCount = selectedCapacity();
  const request = ++catalogFitRequest;
  if (!sparkCount) {
    catalogFits = new Map();
    catalogCapacityLoading = false;
    renderCatalogCards();
    $('catalog-status').textContent = catalogBaseStatus;
    return;
  }
  catalogCapacityLoading = true;
  renderCatalogCards();
  $('catalog-status').textContent = `Analyzing ${catalogModels.length} models for ${sparkCount} Spark${sparkCount === 1 ? '' : 's'}…`;
  try {
    const response = await api('/api/v1/catalog/fit', {method: 'POST', body: JSON.stringify({
      models: catalogModels.map(model => ({id: model.id, commit: model.commit})),
      spark_count: sparkCount,
      context_length: Number($('catalog-fit-context').value),
      concurrent_sequences: 1,
      kv_cache_dtype: 'auto',
    })});
    if (request !== catalogFitRequest) return;
    catalogFits = new Map(response.results.map(result => [result.repo_id, result]));
    catalogCapacityLoading = false;
    renderCatalogCards();
    const compatible = response.results.filter(result => result.compatible).length;
    const cached = response.results.filter(result => result.cached).length;
    const unavailable = response.results.filter(result => result.error).length;
    const context = Number($('catalog-fit-context').value) / 1024;
    $('catalog-status').textContent = `${compatible} of ${catalogModels.length} fit ${sparkCount} Spark${sparkCount === 1 ? '' : 's'} at ${context >= 1024 ? `${context / 1024}M` : `${context}k`} context${cached ? ` · ${cached} cached` : ''}${unavailable ? ` · ${unavailable} unavailable` : ''}`;
  } catch (error) {
    if (request !== catalogFitRequest) return;
    catalogCapacityLoading = false;
    catalogFits = new Map();
    renderCatalogCards();
    $('catalog-status').textContent = `Capacity filter unavailable · ${error.message}`;
  }
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
    catalogBaseStatus = `${catalogModels.length} models${catalogCursor ? ' · more available' : ''} · Hugging Face Hub`;
    if (selectedCapacity()) await applyCapacityFilter();
    else {
      renderCatalogCards();
      $('catalog-status').textContent = catalogBaseStatus;
    }
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

async function inspectCatalogModel(repo, revision = 'main', opener = null) {
  const request = ++detailRequest;
  catalogSelection = repo;
  $('repo-id').value = repo;
  $('revision').value = revision;
  resetModelFit();
  renderCatalogCards();
  openCatalogSheet(opener, repo.split('/').pop());
  $('catalog-detail').hidden = false;
  $('model-form').hidden = true;
  $('manual-repo-label').hidden = true;
  $('catalog-sheet-body').className = 'catalog-sheet-body is-loading';
  $('catalog-detail').innerHTML = '<div class="catalog-detail-head"><p>Inspecting repository files…</p></div>';
  try {
    const model = await api(`/api/v1/catalog/models/${repo.split('/').map(encodeURIComponent).join('/')}?revision=${encodeURIComponent(revision)}`);
    if (request !== detailRequest) return;
    const facts = [['Repository size', `${model.size_complete ? '' : 'At least '}${formatBytes(model.repository_bytes)}`], ['Parameters', model.parameters ? formatNumber(model.parameters) : 'Not reported'], ['Architecture', model.architecture || model.library || 'Not reported'], ['License', model.license || 'See model license']];
    $('catalog-detail').innerHTML = `${catalogImage(model, true)}<div class="catalog-detail-head"><div class="catalog-detail-identity"><span class="catalog-avatar">${escapeHtml(model.author[0].toUpperCase())}</span><div><p class="eyebrow">${escapeHtml(model.author)}</p><h3>${escapeHtml(model.id.split('/').pop())}</h3></div></div><div class="catalog-tags">${[model.task, model.library].filter(Boolean).map(tag => `<span>${escapeHtml(tag)}</span>`).join('')}</div><div class="catalog-popularity"><span>↓ <b>${formatNumber(model.downloads)}</b><small>downloads</small></span><span>♡ <b>${formatNumber(model.likes)}</b><small>likes</small></span></div><dl class="catalog-facts">${facts.map(([title, value]) => `<div><dt>${title}</dt><dd>${escapeHtml(value)}</dd></div>`).join('')}</dl>${model.gated ? '<p class="catalog-access-note">License approval is required. Accept access on Hugging Face and configure the controller token.</p>' : ''}<a class="catalog-hf-link" href="https://huggingface.co/${model.id.split('/').map(encodeURIComponent).join('/')}" target="_blank" rel="noopener noreferrer">Open model card &amp; license <span>↗</span></a></div><details class="catalog-files"><summary><span>${model.files.length} repository files</span><small>Show file manifest</small></summary><div class="catalog-file-list">${model.files.map(file => `<div class="catalog-file"><span>${escapeHtml(file.name)}</span><span>${file.size === null ? 'Unknown size' : formatBytes(file.size)}</span></div>`).join('')}</div></details>`;
    $('catalog-sheet-body').className = 'catalog-sheet-body';
    $('model-form').hidden = false;
  } catch (error) {
    if (request !== detailRequest) return;
    $('catalog-sheet-body').className = 'catalog-sheet-body is-loading';
    $('catalog-detail').innerHTML = `<div class="catalog-detail-head"><h3>${escapeHtml(repo)}</h3><p>${escapeHtml(error.message)}</p><button id="catalog-detail-retry" type="button" class="secondary">Retry inspection</button></div>`;
    $('catalog-detail-retry').addEventListener('click', () => inspectCatalogModel(repo, revision));
  }
}

$('catalog-grid').addEventListener('click', event => {
  const button = event.target.closest('[data-repo]');
  if (button) inspectCatalogModel(button.dataset.repo, 'main', button);
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
['catalog-capacity', 'catalog-fit-context'].forEach(id => $(id).addEventListener('change', () => applyCapacityFilter()));
$('catalog-reset').addEventListener('click', () => {
  $('catalog-query').value = '';
  $('catalog-sort').value = 'trendingScore';
  $('catalog-task').value = 'text-generation';
  $('catalog-library').value = '';
  $('catalog-access').value = '';
  $('catalog-capacity').value = '';
  $('catalog-fit-context').value = '32768';
  catalogFits = new Map();
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
  $('catalog-detail').hidden = true;
  $('model-form').hidden = false;
  $('manual-repo-label').hidden = false;
  $('catalog-sheet-body').className = 'catalog-sheet-body manual-mode';
  openCatalogSheet($('catalog-manual'), 'Add a Hugging Face repository');
  renderCatalogCards();
  requestAnimationFrame(() => $('repo-id').focus());
});
$('catalog-sheet-close').addEventListener('click', closeCatalogSheet);
$('catalog-sheet-backdrop').addEventListener('click', closeCatalogSheet);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !$('catalog-sheet').hidden) closeCatalogSheet();
});
window.addEventListener('hashchange', closeCatalogSheet);
$('revision').addEventListener('change', () => {
  if (catalogSelection) inspectCatalogModel(catalogSelection, $('revision').value.trim() || 'main');
});
document.addEventListener('library-updated', event => {
  const previous = JSON.stringify(catalogModels.map(model => catalogState(model.id)));
  catalogJobs = event.detail;
  if (previous !== JSON.stringify(catalogModels.map(model => catalogState(model.id)))) renderCatalogCards();
});
loadCatalog();

// Keep private, missing, or temporarily unavailable Hub thumbnails graceful.
document.addEventListener('error', event => {
  if (event.target.matches?.('.catalog-media img')) event.target.classList.add('failed');
}, true);
