/* Render untrusted Hub model cards. Parser/sanitizer are vendored for LAN use. */
function renderModelReadme(content, model) {
  if (!window.marked || !window.DOMPurify) return escapeHtml(content);
  // Hub YAML describes metadata, rather than documentation for the reader.
  const markdown = content.replace(/^\uFEFF?---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/, '');
  const fragment = DOMPurify.sanitize(marked.parse(markdown, {gfm: true, breaks: false}), {
    RETURN_DOM_FRAGMENT: true,
    ALLOWED_TAGS: ['p', 'br', 'hr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'img',
      'strong', 'b', 'em', 'i', 'del', 's', 'ul', 'ol', 'li', 'blockquote', 'pre', 'code',
      'table', 'thead', 'tbody', 'tr', 'th', 'td', 'details', 'summary', 'div', 'span',
      'sup', 'sub', 'kbd', 'picture', 'source'],
    ALLOWED_ATTR: ['href', 'src', 'alt', 'title', 'colspan', 'rowspan', 'start', 'open'],
  });
  const repo = model.id.split('/').map(encodeURIComponent).join('/');
  const revision = encodeURIComponent(model.revision || 'main');
  const directory = (model.readme_path || 'README.md').split('/').slice(0, -1)
    .map(encodeURIComponent).join('/');
  const base = `https://huggingface.co/${repo}`;
  fragment.querySelectorAll('a, img').forEach(element => {
    const image = element.tagName === 'IMG';
    const attribute = image ? 'src' : 'href';
    const value = element.getAttribute(attribute);
    if (!value) return;
    if (!image && value.startsWith('#')) return;
    try {
      const url = new URL(value, `${base}/${image ? 'resolve' : 'blob'}/${revision}/${directory ? `${directory}/` : ''}`);
      if (!['https:', 'http:', ...(!image ? ['mailto:'] : [])].includes(url.protocol)) {
        element.removeAttribute(attribute);
        return;
      }
      element.setAttribute(attribute, url.href);
    } catch {
      element.removeAttribute(attribute);
      return;
    }
    if (image) {
      element.loading = 'lazy';
      element.referrerPolicy = 'no-referrer';
    } else {
      element.target = '_blank';
      element.rel = 'noopener noreferrer';
    }
  });
  fragment.querySelectorAll('pre').forEach(element => element.tabIndex = 0);
  const wrapper = document.createElement('div');
  wrapper.append(fragment);
  return wrapper.innerHTML;
}
