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

function renderCatalogCards() {
  $('catalog-grid').innerHTML = catalogModels.length ? catalogModels.map(model => `
    <button type="button" class="catalog-card ${catalogSelection === model.id ? 'selected' : ''}" data-repo="${escapeHtml(model.id)}" aria-pressed="${catalogSelection === model.id}">
      ${catalogImage(model)}
      <span class="catalog-card-content">
        <span class="catalog-card-head"><span class="catalog-avatar">${escapeHtml(model.author[0].toUpperCase())}</span><span class="catalog-author">${escapeHtml(model.author)}</span><span class="catalog-card-arrow">↗</span></span>
        <b class="catalog-card-title">${escapeHtml(model.id.split('/').pop())}</b>
        <span class="catalog-tags">${[model.task, model.library, model.parameters ? `${formatNumber(model.parameters)} params` : null].filter(Boolean).slice(0, 3).map(tag => `<span>${escapeHtml(tag)}</span>`).join('')}</span>
        <span class="catalog-card-foot"><span title="Downloads">↓ ${formatNumber(model.downloads)}</span><span title="Likes">♡ ${formatNumber(model.likes)}</span>${model.updated ? `<span class="catalog-updated">${escapeHtml(new Date(model.updated).toLocaleDateString())}</span>` : ''}</span>
      </span>
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
    $('catalog-detail').innerHTML = `${catalogImage(model, true)}<div class="catalog-detail-head"><div class="catalog-detail-identity"><span class="catalog-avatar">${escapeHtml(model.author[0].toUpperCase())}</span><div><p class="eyebrow">${escapeHtml(model.author)}</p><h3>${escapeHtml(model.id.split('/').pop())}</h3></div></div><div class="catalog-tags">${[model.task, model.library].filter(Boolean).map(tag => `<span>${escapeHtml(tag)}</span>`).join('')}</div><div class="catalog-popularity"><span>↓ <b>${formatNumber(model.downloads)}</b><small>downloads</small></span><span>♡ <b>${formatNumber(model.likes)}</b><small>likes</small></span></div><dl class="catalog-facts">${facts.map(([title, value]) => `<div><dt>${title}</dt><dd>${escapeHtml(value)}</dd></div>`).join('')}</dl>${model.gated ? '<p class="catalog-access-note">License approval is required. Accept access on Hugging Face and configure the controller token.</p>' : ''}<a class="catalog-hf-link" href="https://huggingface.co/${model.id.split('/').map(encodeURIComponent).join('/')}" target="_blank" rel="noopener noreferrer">Open model card &amp; license <span>↗</span></a></div><details class="catalog-files"><summary><span>${model.files.length} repository files</span><small>Show file manifest</small></summary><div class="catalog-file-list">${model.files.map(file => `<div class="catalog-file"><span>${escapeHtml(file.name)}</span><span>${file.size === null ? 'Unknown size' : formatBytes(file.size)}</span></div>`).join('')}</div></details>`;
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

// Keep private, missing, or temporarily unavailable Hub thumbnails graceful.
document.addEventListener('error', event => {
  if (event.target.matches?.('.catalog-media img')) event.target.classList.add('failed');
}, true);
