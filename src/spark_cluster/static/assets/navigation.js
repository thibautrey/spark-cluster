const dashboardRoutes = {
  setup: {title: 'Cluster setup'},
  overview: {title: 'Cluster overview'},
  api: {title: 'Inference API'},
  models: {title: 'Model library'},
  launch: {title: 'Launch a model'},
  logs: {title: 'Runtime logs'},
  topology: {title: 'Cluster topology'},
  cache: {title: 'KV cache'},
  memory: {title: 'Memory'},
};

function selectedDashboardRoute() {
  const route = location.hash.slice(1);
  return dashboardRoutes[route] ? route : 'overview';
}

function showDashboardRoute({resetScroll = true} = {}) {
  const route = selectedDashboardRoute();
  document.body.dataset.route = route;
  document.querySelectorAll('nav a[href^="#"]').forEach(link => {
    const selected = link.getAttribute('href') === `#${route}`;
    link.classList.toggle('active', selected);
    if (selected) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  document.getElementById('page-title').textContent = dashboardRoutes[route].title;
  document.getElementById('route-select').value = route;
  document.title = `${dashboardRoutes[route].title} — Spark Cluster`;
  if (resetScroll) requestAnimationFrame(() => window.scrollTo({top: 0, behavior: 'auto'}));
}

document.getElementById('route-select').addEventListener('change', event => {
  location.hash = `#${event.target.value}`;
});
window.addEventListener('hashchange', () => showDashboardRoute());
if (!dashboardRoutes[location.hash.slice(1)]) history.replaceState(null, '', '#overview');
showDashboardRoute({resetScroll: false});
