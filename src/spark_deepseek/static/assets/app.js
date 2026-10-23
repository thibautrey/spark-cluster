const $ = (id) => document.getElementById(id);
const history = [];
const formatNumber = (value) => value >= 1e9 ? `${(value/1e9).toFixed(1)}B` : value >= 1e6 ? `${(value/1e6).toFixed(1)}M` : value >= 1e3 ? `${(value/1e3).toFixed(1)}k` : `${Math.round(value)}`;
const pct = (used, total) => total ? Math.min(100, used / total * 100) : 0;

function render(snapshot) {
  const memoryUsed = snapshot.nodes.reduce((sum, n) => sum + n.memory.used_gib, 0);
  const memoryTotal = snapshot.nodes.reduce((sum, n) => sum + n.memory.total_gib, 0);
  $('cluster-name').textContent = snapshot.cluster_name.toUpperCase();
  $('timestamp').textContent = new Date(snapshot.timestamp).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
  $('output-tps').textContent = Math.round(snapshot.total_output_tokens_per_second);
  $('hit-rate').textContent = snapshot.cache_hit_rate.toFixed(1);
  $('hit-ring').style.background = `conic-gradient(var(--green) ${snapshot.cache_hit_rate}%,#202a38 0)`;
  $('memory-used').textContent = Math.round(memoryUsed);
  $('memory-total').textContent = `${Math.round(memoryTotal)} GiB total · ${pct(memoryUsed,memoryTotal).toFixed(0)}% allocated`;
  $('memory-bar').style.width = `${pct(memoryUsed,memoryTotal)}%`;
  $('prefix-reuse').textContent = formatNumber(snapshot.prefix_reuse_tokens);
  $('model-short').textContent = snapshot.runtime.model.split('/').pop().replace('DeepSeek-','');
  $('runtime-context').textContent = `${Math.round(snapshot.runtime.context_length/1024)}k`;
  $('runtime-parallel').textContent = `${snapshot.runtime.tensor_parallel} × ${snapshot.runtime.pipeline_parallel}`;
  $('runtime-status').textContent = snapshot.status[0].toUpperCase() + snapshot.status.slice(1);
  $('node-count').textContent = `${snapshot.nodes.length} nodes`;
  renderTopology(snapshot.nodes);
  renderCache(snapshot.cache_tiers);
  renderMemory(snapshot.nodes);
  history.push(snapshot.total_output_tokens_per_second);
  if (history.length > 40) history.shift();
  drawSparkline(history);
}

function renderTopology(nodes) {
  $('topology-map').innerHTML = '<div class="fabric-line"></div>' + nodes.map(node => `
    <div class="node">
      <div class="node-top"><span class="node-icon">${node.role === 'head' ? 'H' : 'W'}</span><i class="node-state"></i></div>
      <h3>${node.hostname}</h3><p>${node.role.toUpperCase()} · ${node.address}</p>
      <div class="node-stats"><span>GPU<b>${node.gpu_utilization.toFixed(0)}%</b></span><span>MEM<b>${node.memory.used_gib.toFixed(0)} / ${node.memory.total_gib.toFixed(0)}</b></span><span>OUTPUT<b>${node.output_tokens_per_second.toFixed(0)} t/s</b></span><span>TEMP<b>${node.temperature_c.toFixed(0)}°C</b></span></div>
    </div>`).join('');
}

function renderCache(tiers) {
  $('cache-tiers').innerHTML = tiers.map(tier => `<div class="cache-row">
    <div class="cache-row-head"><b>${tier.name}</b><span>${tier.used_gib.toFixed(0)} / ${tier.capacity_gib.toFixed(0)} GiB</span></div>
    <div class="cache-track"><i style="width:${pct(tier.used_gib,tier.capacity_gib)}%"></i></div>
    <div class="cache-meta"><span>${((tier.hits/(tier.hits+tier.misses))*100).toFixed(1)}% hit</span><span>${tier.read_latency_ms.toFixed(1)} ms</span></div>
  </div>`).join('');
}

function renderMemory(nodes) {
  $('memory-grid').innerHTML = nodes.map(node => {
    const model = Math.round(node.memory.model_gib/8), kv = Math.round(node.memory.kv_cache_gib/3);
    const runtime = Math.max(1, Math.round((node.memory.used_gib-node.memory.model_gib-node.memory.kv_cache_gib)/3));
    const free = Math.max(1, 16-model-kv-runtime);
    const blocks = (klass,count) => Array(count).fill(`<i class="${klass}"></i>`).join('');
    return `<div><div class="memory-node-head"><b>${node.hostname}</b><span>${node.memory.used_gib.toFixed(0)} / ${node.memory.total_gib.toFixed(0)} GiB</span></div><div class="memory-block">${blocks('model',model)}${blocks('kv',kv)}${blocks('runtime',runtime)}${blocks('free',free)}</div><div class="memory-labels"><span><i class="m"></i>model</span><span><i class="k"></i>KV</span><span><i class="r"></i>runtime</span><span><i class="f"></i>free</span></div></div>`;
  }).join('');
}

function drawSparkline(values) {
  const canvas = $('throughput-spark'), ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height; ctx.clearRect(0,0,w,h); if(values.length < 2) return;
  const min=Math.min(...values)-4,max=Math.max(...values)+4;
  ctx.beginPath(); values.forEach((v,i)=>{const x=i/(values.length-1)*w,y=h-(v-min)/(max-min)*h;(i?ctx.lineTo(x,y):ctx.moveTo(x,y));});
  ctx.strokeStyle='#63f2b3';ctx.lineWidth=2;ctx.stroke();ctx.lineTo(w,h);ctx.lineTo(0,h);ctx.closePath();
  const gradient=ctx.createLinearGradient(0,0,0,h);gradient.addColorStop(0,'rgba(99,242,179,.24)');gradient.addColorStop(1,'rgba(99,242,179,0)');ctx.fillStyle=gradient;ctx.fill();
}

function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/api/v1/stream`);
  socket.onopen=()=>{$('connection-dot').style.background='var(--green)';$('connection-label').textContent='Live stream';};
  socket.onmessage=(event)=>render(JSON.parse(event.data));
  socket.onclose=()=>{$('connection-dot').style.background='var(--amber)';$('connection-label').textContent='Reconnecting';setTimeout(connect,1500);};
}
connect();
