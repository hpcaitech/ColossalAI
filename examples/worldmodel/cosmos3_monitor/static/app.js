/* Cosmos3 training monitor — a small, dependency-free SPA over the read-only JSON API. */
'use strict';

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  runs: [],
  total: 0,
  health: null,
  search: '',
  token: null,
  run: null,
  manifest: null,
  tab: 'overview',
  series: null,
  records: null,
  events: [],
  checkpoints: null,
  system: null,
  ranks: null,
  alerts: null,
  charts: new Map(),
  stream: null,
  detailTimer: null,
  runsTimer: null,
  focusStep: null,
  chartLog: false,
  xMode: 'step',
  range: 0,
  smooth: 0,
  showAnnotations: true,
  paused: false,
  compareTokens: new Set(),
  detailStep: null,
  detail: null,
  theme: localStorage.getItem('c3-theme') || 'dark',
};

const SERIES_KEYS = [
  'loss/total', 'loss/vision', 'loss/action', 'optim/lr', 'optim/grad_norm',
  'perf/step_s', 'perf/samples_per_s', 'perf/mfu', 'perf/skew',
  'mem/device_used_gib', 'mem/peak_reserved_gib', 'mem/allocated_gib', 'mem/device_total_gib',
  'data/samples', 'data/ar_tokens', 'data/dm_tokens', 'data/wait_s', 'data/vae_s', 'data/epoch',
  'perf/forward_s', 'perf/backward_s', 'perf/optim_s',
].join(',');

function esc(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function palette() {
  return [1, 2, 3, 4, 5, 6].map((i) => css('--series-' + i));
}

function num(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return '—';
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  if (Math.abs(n) >= 1e6) return n.toExponential(2);
  return n.toFixed(digits);
}

function int(value) {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  return Math.round(Number(value)).toLocaleString('en-US');
}

function bytes(value) {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  let n = Number(value);
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${n.toFixed(n >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function duration(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return '—';
  const s = Math.max(0, Number(seconds));
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}d ${Math.round((s % 86400) / 3600)}h`;
}

function ago(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return '—';
  return `${duration(seconds)} ago`;
}

function clock(ts) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleString();
}

function elapsed(value) {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  const total = Math.max(0, Number(value));
  if (total < 60) return `${total.toFixed(total < 10 ? 1 : 0)}s`;
  if (total < 3600) return `${Math.floor(total / 60)}:${String(Math.floor(total % 60)).padStart(2, '0')}`;
  return `${Math.floor(total / 3600)}:${String(Math.floor((total % 3600) / 60)).padStart(2, '0')}:${String(Math.floor(total % 60)).padStart(2, '0')}`;
}

function pct(value, digits = 1) {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function statusPill(name) {
  const label = {
    running: '运行中', completed: '已完成', attention: '需关注', failed: '失败',
    initializing: '初始化', unknown: '未知', stale: '过期',
  }[name] || name || '未知';
  return `<span class="status ${esc(name || 'unknown')}"><span class="dot"></span>${esc(label)}</span>`;
}

function severityChip(sev) {
  const cls = sev === 'critical' || sev === 'error' ? 'bad' : sev === 'warn' ? 'warn' : 'ok';
  const icon = sev === 'critical' || sev === 'error' ? '!' : sev === 'warn' ? '△' : 'i';
  return `<span class="chip ${cls}">${icon} ${esc(sev || 'info')}</span>`;
}

async function api(path) {
  const response = await fetch(path, { cache: 'no-store' });
  let body = null;
  try { body = await response.json(); } catch (_) { /* empty body */ }
  if (!response.ok) {
    throw new Error((body && body.detail) || `${response.status} ${response.statusText}`);
  }
  return body ? body.data : null;
}

function toast(message, kind = '') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  $('#toast').appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

/* --------------------------------------------------------------------------------------------------------- */
/* Sparkline / small SVG                                                                                       */
/* --------------------------------------------------------------------------------------------------------- */
function sparkline(values, color, width = 92, height = 24) {
  if (!values || values.length < 2) return '<span class="muted small">—</span>';
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const step = width / (values.length - 1);
  const points = values.map((v, i) => `${(i * step).toFixed(1)},${(height - 2 - ((v - min) / span) * (height - 4)).toFixed(1)}`).join(' ');
  return `<svg class="sparkline" viewBox="0 0 ${width} ${height}" aria-hidden="true"><polyline fill="none" stroke="${color}" stroke-width="1.5" points="${points}"/></svg>`;
}

/* --------------------------------------------------------------------------------------------------------- */
/* Charts                                                                                                     */
/* --------------------------------------------------------------------------------------------------------- */
function baseOption() {
  const p = palette();
  return {
    backgroundColor: 'transparent',
    animation: false,
    color: p,
    textStyle: { color: css('--text-2'), fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif', fontSize: 11 },
    grid: { left: 54, right: 28, top: 34, bottom: 34, containLabel: false },
    tooltip: {
      trigger: 'axis',
      backgroundColor: css('--surface'),
      borderColor: css('--border-strong'),
      textStyle: { color: css('--text'), fontSize: 11 },
      axisPointer: { type: 'cross', lineStyle: { color: css('--muted'), width: 1 } },
      valueFormatter: (value) => (typeof value === 'number' ? value.toLocaleString('en-US', { maximumFractionDigits: 4 }) : value),
    },
    xAxis: {
      type: 'value',
      // Integer ticks are mandatory when the axis is an optimizer step.  In elapsed-time mode the axis is
      // seconds from the first record and gets a clock formatter instead.
      minInterval: state.xMode === 'step' ? 1 : null,
      axisLine: { lineStyle: { color: css('--grid') } },
      axisLabel: {
        color: css('--muted'), fontSize: 10,
        formatter: state.xMode === 'step' ? (v) => (v >= 1000 ? `${v / 1000}k` : v) : (v) => elapsed(v),
      },
      splitLine: { show: false },
      scale: true,
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      axisLabel: { color: css('--muted'), fontSize: 10 },
      splitLine: { lineStyle: { color: css('--grid'), width: 1 } },
      scale: true,
    },
    legend: {
      // The unit lives in the card header.  Keeping it out of the ECharts axis name is what stops the
      // "device used" / "GiB" and "skew" / "/ mean" collisions in the top-left corner.
      type: 'scroll', top: 0, left: 0, right: 0,
      icon: 'roundRect', itemWidth: 9, itemHeight: 9,
      textStyle: { color: css('--text-2'), fontSize: 11 }, itemGap: 10,
      pageIconSize: 8, pageTextStyle: { color: css('--muted'), fontSize: 10 },
    },
    dataZoom: [{ type: 'inside', filterMode: 'none' }],
  };
}

function line(name, points, color, extra = {}) {
  return Object.assign({
    name,
    type: 'line',
    showSymbol: false,
    symbol: 'circle',
    symbolSize: 6,
    sampling: 'lttb',
    lineStyle: { width: 2, color },
    itemStyle: { color },
    data: points,
    connectNulls: false,
  }, extra);
}

function pointsOf(series, key) {
  const item = series && series[key];
  if (!item) return [];
  let pairs = item.x.map((x, i) => [x, item.y[i]]).filter((pair) => pair[1] != null);
  if (state.range && state.xMode === 'step' && pairs.length) {
    const last = Math.max(...pairs.map((pair) => pair[0]));
    const first = last - state.range + 1;
    pairs = pairs.filter((pair) => pair[0] >= first);
  } else if (state.range && state.xMode === 'time' && pairs.length) {
    pairs = pairs.slice(-state.range);
  }
  return pairs;
}

function smoothPoints(points, window) {
  if (!window) return points;
  const alpha = 2 / (Number(window) + 1);
  let ema = null;
  return points.map(([x, y]) => {
    if (y == null) return [x, null];
    ema = ema == null ? y : alpha * y + (1 - alpha) * ema;
    return [x, ema];
  });
}

function annotationLines() {
  if (!state.showAnnotations || state.xMode !== 'step') return [];
  const rows = [];
  if (state.focusStep != null) {
    rows.push({ xAxis: state.focusStep, lineStyle: { color: css('--warning'), type: 'dashed', width: 1 }, label: { formatter: `step ${state.focusStep}`, color: css('--warning'), fontSize: 10 } });
  }
  const seen = new Set();
  for (const checkpoint of (state.checkpoints && state.checkpoints.checkpoints) || []) {
    const step = Number(checkpoint.step);
    if (!Number.isFinite(step) || seen.has(`c${step}`)) continue;
    seen.add(`c${step}`);
    rows.push({ xAxis: step, lineStyle: { color: css('--good'), type: 'dotted', width: 1 }, label: { formatter: 'ckpt', color: css('--good'), fontSize: 9 } });
  }
  for (const event of (state.events || []).filter((item) => ['warn', 'error', 'critical'].includes(item.severity)).slice(-24)) {
    const step = Number(event.step);
    if (!Number.isFinite(step) || seen.has(`e${step}${event.kind}`)) continue;
    seen.add(`e${step}${event.kind}`);
    rows.push({ xAxis: step, lineStyle: { color: css('--warning'), type: 'dashed', width: 1 }, label: { formatter: String(event.kind || 'event'), color: css('--warning'), fontSize: 9 } });
  }
  return rows;
}

function markLines() {
  return annotationLines();
}

function makeChart(id, option) {
  const el = document.getElementById(id);
  if (!el || typeof echarts === 'undefined') return null;
  const existing = echarts.getInstanceByDom(el);
  const chart = existing || echarts.init(el, null, { renderer: 'canvas' });
  // Replacing every option resets the user's zoom on each 5 s refresh.  Keep the viewport and replace only the
  // data series; a tab switch still disposes the instance and starts clean.
  chart.setOption(option, existing ? { notMerge: false, replaceMerge: ['series'] } : true);
  if (!chart.__c3Click) {
    chart.on('click', (params) => {
      if (state.xMode !== 'step') return;
      const value = params && params.value;
      const step = Array.isArray(value) ? value[0] : value;
      if (Number.isFinite(Number(step))) openStepDetail(Math.round(Number(step)));
    });
    chart.__c3Click = true;
  }
  state.charts.set(id, chart);
  return chart;
}

function connectCharts() {
  const charts = Array.from(state.charts.values());
  if (charts.length > 1) echarts.connect('cosmos3-run');
}

function disposeCharts() {
  for (const chart of state.charts.values()) {
    try { chart.dispose(); } catch (_) { /* the DOM may already be gone */ }
  }
  state.charts.clear();
}

function renderLossChart(id = 'chart-loss', tall = false) {
  if (!state.series) return;
  const p = palette();
  const option = baseOption();
  option.yAxis.type = state.chartLog ? 'log' : 'value';
  const total = pointsOf(state.series, 'loss/total');
  option.series = [
    line('total', total, p[0], { lineStyle: { width: 2.25, color: p[0] }, markLine: { symbol: 'none', data: markLines(), silent: true } }),
    line('vision', pointsOf(state.series, 'loss/vision'), p[1]),
    line('action', pointsOf(state.series, 'loss/action'), p[2]),
  ];
  if (state.smooth) {
    option.series.push(line(`total (EMA ${state.smooth})`, smoothPoints(total, state.smooth), css('--text-2'), {
      lineStyle: { width: 2.5, color: css('--text-2'), type: 'dashed' }, itemStyle: { color: css('--text-2') },
    }));
  }
  makeChart(id, option);
}

function renderStepChart(id = 'chart-step') {
  if (!state.series) return;
  const p = palette();
  const option = baseOption();
  const parts = [
    ['wait', 'data/wait_s', p[4]],
    ['VAE', 'data/vae_s', p[3]],
    ['forward', 'perf/forward_s', p[0]],
    ['backward', 'perf/backward_s', p[1]],
    ['optim', 'perf/optim_s', p[2]],
  ];
  option.series = parts.map(([name, key, color]) => line(name, pointsOf(state.series, key), color, {
    stack: 'step', areaStyle: { opacity: 0.28, color }, lineStyle: { width: 1, color },
  }));
  option.series.push(line('total', pointsOf(state.series, 'perf/step_s'), css('--text-2'), {
    stack: undefined, lineStyle: { width: 1.5, type: 'dashed', color: css('--text-2') }, markLine: { symbol: 'none', data: markLines(state.focusStep), silent: true },
  }));
  makeChart(id, option);
}

function renderThroughputChart(id = 'chart-throughput') {
  if (!state.series) return;
  const option = baseOption();
  option.series = [line('throughput', pointsOf(state.series, 'perf/samples_per_s'), palette()[2])];
  makeChart(id, option);
}

function renderMfuChart(id = 'chart-mfu') {
  if (!state.series) return;
  const option = baseOption();
  const points = pointsOf(state.series, 'perf/mfu').map(([x, y]) => [x, y * 100]);
  option.series = [line('MFU', points, palette()[0])];
  makeChart(id, option);
}

function renderLrChart(id = 'chart-lr') {
  if (!state.series) return;
  const option = baseOption();
  option.series = [line('learning rate', pointsOf(state.series, 'optim/lr'), palette()[0])];
  makeChart(id, option);
}

function renderGradChart(id = 'chart-grad') {
  if (!state.series) return;
  const option = baseOption();
  option.series = [line('grad norm', pointsOf(state.series, 'optim/grad_norm'), palette()[1])];
  makeChart(id, option);
}

function renderMemoryChart(id = 'chart-mem') {
  if (!state.series) return;
  const p = palette();
  const option = baseOption();
  option.series = [
    line('device used', pointsOf(state.series, 'mem/device_used_gib'), p[0]),
    line('peak reserved', pointsOf(state.series, 'mem/peak_reserved_gib'), p[1]),
    line('allocated', pointsOf(state.series, 'mem/allocated_gib'), p[2]),
    line('device total', pointsOf(state.series, 'mem/device_total_gib'), css('--muted'), { lineStyle: { width: 1.5, type: 'dashed', color: css('--muted') } }),
  ];
  makeChart(id, option);
}

function renderSkewChart(id = 'chart-skew') {
  if (!state.series) return;
  const option = baseOption();
  option.series = [line('skew', pointsOf(state.series, 'perf/skew'), palette()[4], {
    markLine: {
      symbol: 'none', silent: true,
      data: [{ yAxis: 1.2, lineStyle: { color: css('--warning'), type: 'dashed' }, label: { formatter: '1.2', color: css('--warning'), fontSize: 10 } }],
    },
  })];
  makeChart(id, option);
}

function renderDataChart(id = 'chart-data') {
  if (!state.series) return;
  const p = palette();
  const option = baseOption();
  option.yAxis.type = 'log';
  option.series = [
    line('samples', pointsOf(state.series, 'data/samples'), p[0]),
    line('AR tokens', pointsOf(state.series, 'data/ar_tokens'), p[1]),
    line('DM tokens', pointsOf(state.series, 'data/dm_tokens'), p[2]),
  ];
  makeChart(id, option);
}

function checkpointFactor() {
  const args = (state.manifest && state.manifest.args) || {};
  const mode = String(args.activation_checkpointing || args.checkpoint_mode || 'full').toLowerCase();
  return mode === 'none' ? 2 : 3;
}

function renderStageMfuChart(id = 'chart-stage-mfu') {
  if (!state.series) return;
  const p = palette();
  const samples = state.series['data/samples'];
  const world = (state.run && state.run.world) || 0;
  if (!samples || !world) return;
  const index = new Map(samples.x.map((x, i) => [Math.round(x), i]));
  const factor = checkpointFactor();
  const peak = 353.9;
  const points = (key, flopsPerSample) => {
    const source = state.series[key];
    if (!source) return [];
    const result = [];
    for (let i = 0; i < source.x.length; i += 1) {
      const sample = samples.y[index.get(Math.round(source.x[i]))];
      const seconds = source.y[i];
      if (sample == null || !seconds) continue;
      result.push([source.x[i], (sample * flopsPerSample * 1e12) / seconds / (world * peak * 1e12) * 100]);
    }
    return result;
  };
  const option = baseOption();
  option.yAxis.name = '';
  option.yAxis.axisLabel.formatter = (value) => `${value.toFixed(0)}%`;
  option.series = [
    line('forward', points(state.series['perf/forward_s'] ? 'perf/forward_s' : 'perf/fwd_s', 20.34), p[0]),
    line('backward', points(state.series['perf/backward_s'] ? 'perf/backward_s' : 'perf/bwd_s', 20.34 * factor), p[1]),
  ];
  const vae = points('data/vae_s', 33.0);
  if (vae.length) option.series.push(line('VAE', vae, p[2]));
  makeChart(id, option);
}

function renderEpochChart(id = 'chart-epoch') {
  if (!state.series) return;
  const option = baseOption();
  option.yAxis.name = '';
  option.series = [line('epoch', pointsOf(state.series, 'data/epoch'), palette()[3], { step: 'end' })];
  makeChart(id, option);
}

function renderCheckpointCharts() {
  const rows = (state.checkpoints && state.checkpoints.checkpoints) || [];
  if (!rows.length) return;
  const p = palette();
  const sizeOption = baseOption();
  sizeOption.series = [{
    name: 'checkpoint size', type: 'bar', barMaxWidth: 18,
    itemStyle: { color: p[0], borderRadius: [4, 4, 0, 0] },
    data: rows.filter((r) => r.bytes != null).map((r) => [Number(r.step) || 0, Number(r.bytes)]),
  }];
  makeChart('chart-ckpt-size', sizeOption);

  const durationOption = baseOption();
  durationOption.series = [{
    name: 'save duration', type: 'bar', barMaxWidth: 18,
    itemStyle: { color: p[2], borderRadius: [4, 4, 0, 0] },
    data: rows.filter((r) => r.duration_ms != null).map((r) => [Number(r.step) || 0, Number(r.duration_ms) / 1000]),
  }];
  makeChart('chart-ckpt-duration', durationOption);
}

function renderSystemCharts(host) {
  const entry = state.system && state.system.hosts ? state.system.hosts[host] : null;
  if (!entry) return;
  const p = palette();
  const devices = Object.keys(entry.devices || {}).sort((a, b) => Number(a) - Number(b));
  const hbmOption = baseOption();
  hbmOption.series = devices.map((index, i) => {
    const values = (entry.devices[index].hbm_used_mib || []).map(([t, v]) => [t, v / 1024]);
    return line(`NPU ${index}`, values, p[i % p.length], { xAxisIndex: 0 });
  });
  hbmOption.xAxis.type = 'value';
  hbmOption.xAxis.axisLabel.formatter = (v) => new Date(v * 1000).toLocaleTimeString();
  makeChart('chart-sys-hbm', hbmOption);

  const tempOption = baseOption();
  tempOption.series = devices.map((index, i) => line(`NPU ${index}`, (entry.devices[index].temperature_c || []).map(([t, v]) => [t, v]), p[i % p.length]));
  tempOption.xAxis.axisLabel.formatter = (v) => new Date(v * 1000).toLocaleTimeString();
  makeChart('chart-sys-temp', tempOption);
}

/* --------------------------------------------------------------------------------------------------------- */
/* Runs list                                                                                                  */
/* --------------------------------------------------------------------------------------------------------- */
async function loadHealth() {
  try {
    state.health = await api('/api/health');
    const h = state.health;
    $('#health').innerHTML = `${h.runs} runs · scan ${(h.last_scan_duration_ms || 0).toFixed(0)}ms · <span class="${h.last_error ? 'bad' : 'muted'}">${h.last_error ? esc(h.last_error) : 'ok'}</span>`;
  } catch (error) {
    $('#health').innerHTML = `<span class="bad">${esc(error.message)}</span>`;
  }
}

async function loadRuns() {
  try {
    const data = await api(`/api/runs?q=${encodeURIComponent(state.search)}&limit=500`);
    state.runs = data.runs || [];
    state.total = data.total || 0;
    renderSidebar();
    if (!state.token) renderRunsPage();
  } catch (error) {
    $('#run-list').innerHTML = `<div class="bad small">${esc(error.message)}</div>`;
  }
}

function renderSidebar() {
  const active = state.token;
  const rows = state.runs.map((run) => {
    const cls = run.token === active ? 'active' : '';
    const step = run.step != null ? `step ${int(run.step)}` : 'no steps';
    const loss = run.last && run.last.loss != null ? `loss ${num(run.last.loss, 4)}` : '';
    const alert = run.alert_total ? `<span class="bad">${run.alert_total} alert</span>` : '';
    return `<a class="run-item ${cls}" href="#/run/${run.token}/overview" data-token="${run.token}">
      <div class="ri-top">${statusPill(run.state)}<span class="ri-name">${esc(run.display_name)}</span></div>
      <div class="ri-meta"><span>${esc(step)}</span><span>${esc(loss)}</span>${alert ? `<span>${alert}</span>` : ''}</div>
    </a>`;
  }).join('');
  $('#run-list').innerHTML = rows || '<div class="empty small">没有发现 run<br>用 <code>--roots</code> 指定目录</div>';
}

function renderRunsPage() {
  const p = palette();
  const rows = state.runs.map((run) => {
    const last = run.last || {};
    const hbm = last.hbm_fraction != null ? `${pct(last.hbm_fraction, 0)}` : '—';
    const progress = run.progress != null ? Math.max(0, Math.min(1, run.progress)) : 0;
    const alerts = run.alert_total ? `<span class="chip bad">${run.alert_total}</span>` : '<span class="muted">—</span>';
    return `<tr>
      <td><input class="cmp-select" type="checkbox" data-token="${run.token}" ${state.compareTokens.has(run.token) ? 'checked' : ''} title="加入对比"></td>
      <td><a href="#/run/${run.token}/overview"><strong>${esc(run.display_name)}</strong></a><div class="muted small ellipsis">${esc(run.dir)}</div></td>
      <td>${statusPill(run.state)}${run.stale ? ' <span class="chip warn">stale</span>' : ''}</td>
      <td class="num">${int(run.step)}${run.max_steps ? `<span class="muted"> / ${int(run.max_steps)}</span>` : ''}<div class="progress" style="margin-top:4px"><span style="width:${(progress * 100).toFixed(1)}%"></span></div></td>
      <td class="num">${num(last.loss, 4)}</td>
      <td class="num">${num(last.step_s, 1)}<span class="muted">s</span></td>
      <td class="num">${num(last.samples_per_s, 1)}</td>
      <td class="num">${last.mfu != null ? `${num(last.mfu * 100, 1)}%` : '—'}</td>
      <td class="num">${hbm}</td>
      <td class="num">${sparkline(run.sparkline && run.sparkline.loss, p[0])}</td>
      <td class="num">${alerts}</td>
      <td class="num"><span class="muted small">${ago(run.age_s)}</span></td>
    </tr>`;
  }).join('');
  const table = state.runs.length ? `<div class="card"><div class="card-head"><h2>Runs <span class="muted small">${state.total}</span></h2><div class="hint-inline">点击任一行查看配置、曲线、checkpoint 与告警</div></div>
    <div class="table-scroll"><table class="table"><thead><tr>
      <th></th><th>run</th><th>状态</th><th class="num">step</th><th class="num">loss</th><th class="num">step time</th>
      <th class="num">samples/s</th><th class="num">MFU</th><th class="num">HBM</th><th class="num">loss 趋势</th><th class="num">alerts</th><th class="num">更新</th>
    </tr></thead><tbody>${rows}</tbody></table></div></div>` :
    `<div class="card empty"><h2>还没有发现 run</h2><p>用 <code>--roots</code> 指向包含 <code>metrics/run.json</code> 或 <code>metrics/aggregated.jsonl</code> 的目录。例如：</p><pre class="mono">python -m examples.worldmodel.cosmos3_monitor serve --roots /mnt/disk0/jiangzs/nzf/bench_online/runs</pre></div>`;
  $('#crumbs').innerHTML = '<strong>Runs</strong>';
  $('#view').innerHTML = `<div class="page-head"><div><h1>训练运行</h1><div class="subtle">只读索引 · 数据来自每个 run 目录的 JSONL</div></div><div class="spacer"></div><div class="page-actions"><span class="chip">${state.total} runs</span><button id="compare-selected" class="primary" ${state.compareTokens.size < 2 ? 'disabled' : ''}>对比选中 (${state.compareTokens.size})</button><button id="refresh-runs" class="ghost">刷新</button></div></div>${table}`;
  const button = $('#refresh-runs');
  if (button) button.onclick = () => { loadRuns(); loadHealth(); };
  const updateCompare = () => {
    const target = $('#compare-selected');
    if (!target) return;
    target.disabled = state.compareTokens.size < 2;
    target.textContent = `对比选中 (${state.compareTokens.size})`;
  };
  document.querySelectorAll('.cmp-select').forEach((box) => {
    box.onchange = () => {
      if (box.checked) state.compareTokens.add(box.dataset.token);
      else state.compareTokens.delete(box.dataset.token);
      updateCompare();
    };
  });
  const compareButton = $('#compare-selected');
  if (compareButton) compareButton.onclick = () => {
    if (state.compareTokens.size < 2) return;
    location.hash = `#/compare?runs=${Array.from(state.compareTokens).join(',')}`;
  };
}

/* --------------------------------------------------------------------------------------------------------- */
/* Run detail                                                                                                 */
/* --------------------------------------------------------------------------------------------------------- */
async function openRun(token, tab) {
  if (state.token !== token) {
    state.run = null;
    state.series = null;
    state.checkpoints = null;
    state.events = [];
    state.alerts = null;
    state.system = null;
    state.ranks = null;
    state.records = null;
    state.focusStep = null;
    disposeCharts();
    closeStream();
  }
  state.token = token;
  state.tab = tab || 'overview';
  history.replaceState(null, '', `#/run/${token}/${state.tab}`);
  renderSidebar();
  await loadRunData();
  renderRunView();
  openStream(token);
  if (!state.detailTimer) state.detailTimer = setInterval(refreshDetailQuiet, 5000);
}

async function loadRunData() {
  const token = state.token;
  if (!token) return;
  const detail = await api(`/api/runs/${token}?include=manifest`);
  if (state.token !== token) return;
  state.run = detail;
  state.manifest = detail.manifest || null;
  const [series, events, checkpoints, alerts, system, ranks] = await Promise.all([
    api(`/api/runs/${token}/series?keys=${encodeURIComponent(SERIES_KEYS)}&max_points=5000&x=${state.xMode}`).catch(() => null),
    api(`/api/runs/${token}/events?limit=400`).catch(() => ({ events: [] })),
    api(`/api/runs/${token}/checkpoints`).catch(() => null),
    api(`/api/runs/${token}/alerts`).catch(() => null),
    api(`/api/runs/${token}/system`).catch(() => null),
    api(`/api/runs/${token}/ranks`).catch(() => null),
  ]);
  if (state.token !== token) return;
  state.series = series ? series.series : null;
  state.events = (events && events.events) || [];
  state.checkpoints = checkpoints;
  state.alerts = alerts;
  state.system = system;
  state.ranks = ranks;
}

async function refreshDetailQuiet() {
  if (!state.token || document.hidden || state.paused) return;
  try {
    await loadRunData();
    if (state.tab === 'overview' || state.tab === 'charts') {
      updateHeaderMeta();
      renderKpis();
      if (state.tab === 'charts') refreshCharts();
      else { renderLossChart('chart-loss-overview'); renderStepChart('chart-step-overview'); }
    } else if (state.tab === 'records' || state.tab === 'events' || state.tab === 'checkpoints' || state.tab === 'system' || state.tab === 'topology') {
      renderRunView();
    } else {
      updateHeaderMeta();
    }
  } catch (_) { /* keep the last good view; the stale badge explains it */ }
}

function updateHeaderMeta() {
  const run = state.run;
  if (!run) return;
  const status = $('#run-status');
  if (status) status.innerHTML = statusPill(run.state);
  const age = $('#run-age');
  if (age) age.textContent = ago(run.age_s);
  const progress = run.progress != null ? Math.max(0, Math.min(1, run.progress)) : 0;
  const bar = $('#run-progress');
  if (bar) bar.style.width = `${(progress * 100).toFixed(1)}%`;
  const step = $('#run-step');
  if (step) step.textContent = `${int(run.step)}${run.max_steps ? ` / ${int(run.max_steps)}` : ''}`;
  const label = $('#run-progress-label');
  if (label) label.textContent = run.progress != null ? `${(progress * 100).toFixed(1)}%` : '';
}

function runCrumbs() {
  const run = state.run || {};
  $('#crumbs').innerHTML = `<a href="#/runs">Runs</a> <span>/</span> <strong>${esc(run.display_name || run.run_id || state.token)}</strong>`;
}

function renderRunHeader() {
  const run = state.run;
  if (!run) return;
  runCrumbs();
  const last = run.last || {};
  const step = run.step || 0;
  const total = run.max_steps;
  const progress = run.progress != null ? Math.max(0, Math.min(1, run.progress)) : 0;
  const alertCount = state.alerts ? (state.alerts.counts ? Object.values(state.alerts.counts).reduce((a, b) => a + b, 0) : 0) : 0;
  const alertBanner = alertCount ? `<div class="banner ${state.alerts.counts.critical || state.alerts.counts.error ? 'bad' : 'warn'}" style="margin-bottom:12px"><span class="icon">!</span><div class="body"><strong>${alertCount} 个活跃告警</strong><div class="small">${esc((state.alerts.active || []).slice(0, 3).map((a) => a.message).join(' · '))}</div></div><a class="small" href="#/run/${run.token}/events">查看</a></div>` : '';
  $('#view').innerHTML = `
    <div class="page-head">
      <div>
        <h1>${esc(run.display_name || run.run_id)}</h1>
        <div class="subtle">${esc(run.dir)} · <span class="mono">${esc(run.token)}</span> · ${esc(run.layout)}</div>
      </div>
      <div class="spacer"></div>
      <div class="page-actions"><span id="run-status">${statusPill(run.state)}</span>${run.stale ? '<span class="chip warn">数据过期</span>' : ''}<span class="chip" id="run-age">${esc(ago(run.age_s))}</span></div>
    </div>
    ${alertBanner}
    <div class="card" style="margin-bottom:14px">
      <div class="bar-row"><div class="progress"><span id="run-progress" style="width:${(progress * 100).toFixed(1)}%"></span></div><strong id="run-step">${int(step)}${total ? ` / ${int(total)}` : ''}</strong><span class="muted small" id="run-progress-label">${progress ? `${(progress * 100).toFixed(1)}%` : ''}</span></div>
      <div class="grid k4" style="margin-top:12px" id="kpis"></div>
    </div>
    <div class="tabs" id="tabs">
      ${tabLink('overview', '概览')}${tabLink('charts', '曲线')}${tabLink('records', '记录')}${tabLink('config', '配置')}${tabLink('topology', '拓扑')}${tabLink('checkpoints', 'Checkpoints')}${tabLink('events', '事件与告警')}${tabLink('system', '系统')}${tabLink('logs', '日志')}
    </div>
    <div id="tab-body"></div>`;
  renderKpis();
  $$('#tabs .tab').forEach((el) => { el.onclick = () => { state.tab = el.dataset.tab; history.replaceState(null, '', `#/run/${state.token}/${state.tab}`); renderRunView(); }; });
}

function tabLink(tab, label) {
  return `<button class="tab ${state.tab === tab ? 'active' : ''}" data-tab="${tab}">${label}</button>`;
}

function renderKpis() {
  const el = $('#kpis');
  if (!el || !state.run) return;
  const last = state.run.last || {};
  const args = (state.manifest && state.manifest.args) || {};
  const micro = args.samples_per_rank != null ? args.samples_per_rank : '—';
  const accum = args.grad_accum != null ? args.grad_accum : '—';
  const step = state.run.step || 0;
  const kpis = [
    ['Iteration', `${int(step)}${state.run.max_steps ? `<small> / ${int(state.run.max_steps)}</small>` : ''}`, `<a href="#" class="step-link" data-step="${step}">查看 step ${int(step)} 阶段详情</a>`],
    ['Epoch', int(state.run.epoch), `iteration ${int(step)}`],
    ['Batch', `${int(last.samples)}<small>samples / iter</small>`, `micro ${esc(micro)} × accum ${esc(accum)}`],
    ['Loss', num(last.loss, 4), last.vision != null ? `v ${num(last.vision, 3)} · a ${num(last.action, 3)}` : ''],
    ['Step time', `${num(last.step_s, 1)}<small>s</small>`, last.skew != null ? `skew ${num(last.skew, 2)}` : ''],
    ['Throughput', `${num(last.samples_per_s, 1)}<small>samples/s</small>`, last.lr != null ? `lr ${num(last.lr, 6)}` : ''],
    ['MFU', last.mfu != null ? `${num(last.mfu * 100, 1)}<small>%</small>` : '—', last.grad_norm != null ? `grad ${num(last.grad_norm, 2)}` : ''],
    ['HBM', last.device_used_gib != null ? `${num(last.device_used_gib, 1)}<small>/${num(last.device_total_gib, 1)} GiB</small>` : '—', last.hbm_fraction != null ? pct(last.hbm_fraction, 0) : ''],
    ['Ranks', `${int(state.run.live_ranks)}<small>/${int(state.run.world)} live</small>`, `${int(state.run.terminal_ranks)} terminal`],
    ['Checkpoints', `${int(state.run.checkpoints && state.run.checkpoints.count)}`, state.run.checkpoints && state.run.checkpoints.last ? `step ${int(state.run.checkpoints.last.step)}` : 'none yet'],
    ['Elapsed / ETA', duration(state.run.eta_s != null ? state.run.eta_s : null), state.run.started_at ? `since ${clock(state.run.started_at)}` : ''],
  ];
  el.innerHTML = kpis.map(([label, value, hint]) => `<div class="kpi"><div class="label">${esc(label)}</div><div class="value">${value}</div><div class="hint">${hint.startsWith('<a ') ? hint : esc(hint)}</div></div>`).join('');
  document.querySelectorAll('.step-link').forEach((link) => {
    link.onclick = (event) => { event.preventDefault(); openStepDetail(Number(link.dataset.step)); };
  });
}

function renderRunView() {
  if (!state.run) return;
  renderRunHeader();
  const body = $('#tab-body');
  if (!body) return;
  if (state.tab === 'overview') renderOverview(body);
  else if (state.tab === 'charts') renderCharts(body);
  else {
    disposeCharts();
    if (state.tab === 'records') renderRecords(body);
    else if (state.tab === 'config') renderConfig(body);
    else if (state.tab === 'topology') renderTopology(body);
    else if (state.tab === 'checkpoints') renderCheckpoints(body);
    else if (state.tab === 'events') renderEvents(body);
    else if (state.tab === 'system') renderSystem(body);
    else if (state.tab === 'logs') renderLogs(body);
  }
}

function renderOverview(body) {
  disposeCharts();
  const run = state.run;
  const last = run.last || {};
  const hosts = (run.hosts || []).map((h) => `<span class="chip">${esc(h.hostname)} · ${esc((h.ranks || []).length)} ranks</span>`).join(' ');
  const recent = (state.events || []).slice(0, 8).map((event) => `<div class="list-row"><div>${severityChip(event.severity)}</div><div class="grow"><div class="ellipsis">${esc(event.message || event.kind)}</div><div class="muted small">step ${esc(event.step)} · ${esc(clock(event.time))}</div></div></div>`).join('');
  const ckpt = state.checkpoints && state.checkpoints.checkpoints ? state.checkpoints.checkpoints.slice(-5).reverse() : [];
  body.innerHTML = `
    <div class="grid k2" style="grid-template-columns: 1.4fr 1fr">
      <div class="card"><div class="chart-head"><h2>Loss</h2><span class="unit">step</span></div><div id="chart-loss-overview" class="chart"></div><div class="chart-note">total / vision / action · 空点表示该 step 未记录或非有限值</div></div>
      <div class="card"><div class="chart-head"><h2>Step time</h2><span class="unit">seconds</span></div><div id="chart-step-overview" class="chart"></div><div class="chart-note">按段堆叠；虚线为端到端 step 时间</div></div>
    </div>
    <div class="grid k2" style="margin-top:12px">
      <div class="card"><div class="card-head"><h2>最近事件</h2><a class="small" href="#/run/${run.token}/events">全部</a></div>${recent || '<div class="muted small">没有事件</div>'}</div>
      <div class="card"><div class="card-head"><h2>最近 Checkpoints</h2><a class="small" href="#/run/${run.token}/checkpoints">全部</a></div>
        ${ckpt.length ? `<table class="table"><thead><tr><th>step</th><th>状态</th><th class="num">大小</th><th class="num">耗时</th></tr></thead><tbody>${ckpt.map((c) => `<tr><td>${int(c.step)}</td><td>${esc(c.status || '—')}</td><td class="num">${bytes(c.bytes)}</td><td class="num">${c.duration_ms != null ? duration(c.duration_ms / 1000) : '—'}</td></tr>`).join('')}</tbody></table>` : '<div class="muted small">还没有 checkpoint 记录</div>'}
      </div>
    </div>
    <div class="card" style="margin-top:12px"><div class="card-head"><h2>节点</h2><span class="muted small">rank → host 来自 manifest；derived 表示按节点/卡数推断</span></div><div class="list">${hosts || '<span class="muted small">manifest 中没有 topology</span>'}</div></div>
    ${renderRunInfoCard()}`;
  renderLossChart('chart-loss-overview');
  renderStepChart('chart-step-overview');
}

function renderRunInfoCard() {
  const manifest = state.manifest || {};
  const args = manifest.args || {};
  const run = manifest.run || {};
  const git = run.git || {};
  const rows = [
    ['model', args.checkpoint],
    ['data', args.data],
    ['world size', manifest.world_size],
    ['effective global batch', manifest.effective_global_batch],
    ['micro batch / accum', `${args.samples_per_rank ?? '—'} × ${args.grad_accum ?? '—'}`],
    ['save every', args.save_every],
    ['git', git.commit ? `${String(git.commit).slice(0, 10)}${git.dirty ? ' (dirty)' : ''}` : '—'],
    ['command', run.command],
  ].filter(([, value]) => value != null && value !== '');
  return `<div class="card" style="margin-top:12px"><div class="card-head"><h2>Run info</h2><span class="muted small">来自 run.json</span></div><div class="kv">${rows.map(([key, value]) => `<div class="k">${esc(key)}</div><div class="v mono">${esc(String(value))}</div>`).join('')}</div></div>`;
}

function renderCharts(body) {
  disposeCharts();
  const rangeOptions = [0, 20, 100, 500, 2000].map((value) => `<option value="${value}" ${state.range === value ? 'selected' : ''}>${value ? `最近 ${value}` : '全部'}</option>`).join('');
  body.innerHTML = `
    <div class="card chart-toolbar" style="margin-bottom:12px">
      <label class="control-group">X 轴
        <select id="chart-xmode">
          <option value="step" ${state.xMode === 'step' ? 'selected' : ''}>iteration (step)</option>
          <option value="time" ${state.xMode === 'time' ? 'selected' : ''}>elapsed time</option>
        </select>
      </label>
      <label class="control-group">范围 <select id="chart-range">${rangeOptions}</select></label>
      <label class="control-group">平滑 <input type="range" id="chart-smooth" min="0" max="20" value="${state.smooth}"><span class="value" id="chart-smooth-value">${state.smooth}</span></label>
      <label class="checkbox"><input type="checkbox" id="chart-annot" ${state.showAnnotations ? 'checked' : ''}> 事件/checkpoint 标注</label>
      <button id="chart-log" class="ghost">${state.chartLog ? '对数 Y：开' : '对数 Y：关'}</button>
      <button id="chart-reset" class="ghost">重置缩放</button>
      <span class="hint-inline">滚轮缩放、拖拽平移；点击任意点打开逐 step 阶段详情</span>
    </div>
    <div class="grid charts">
      <div class="card"><div class="chart-head"><h2>Loss</h2><span class="unit">loss</span></div><div id="chart-loss" class="chart tall"></div></div>
      <div class="card"><div class="chart-head"><h2>Step time</h2><span class="unit">seconds</span></div><div id="chart-step" class="chart tall"></div></div>
      <div class="card"><div class="chart-head"><h2>Stage MFU</h2><span class="unit">% · forward / backward / VAE</span></div><div id="chart-stage-mfu" class="chart"></div></div>
      <div class="card"><div class="chart-head"><h2>Throughput</h2><span class="unit">samples/s</span></div><div id="chart-throughput" class="chart"></div></div>
      <div class="card"><div class="chart-head"><h2>MFU</h2><span class="unit">% of step</span></div><div id="chart-mfu" class="chart"></div></div>
      <div class="card"><div class="chart-head"><h2>Learning rate</h2><span class="unit">lr</span></div><div id="chart-lr" class="chart"></div></div>
      <div class="card"><div class="chart-head"><h2>Gradient norm</h2><span class="unit">norm</span></div><div id="chart-grad" class="chart"></div></div>
      <div class="card"><div class="chart-head"><h2>Memory</h2><span class="unit">GiB</span></div><div id="chart-mem" class="chart"></div></div>
      <div class="card"><div class="chart-head"><h2>Rank skew</h2><span class="unit">max / mean</span></div><div id="chart-skew" class="chart"></div></div>
      <div class="card"><div class="chart-head"><h2>Batch / tokens per step</h2><span class="unit">log count</span></div><div id="chart-data" class="chart"></div></div>
      <div class="card"><div class="chart-head"><h2>Epoch</h2><span class="unit">data.epoch</span></div><div id="chart-epoch" class="chart"></div></div>
    </div>`;
  renderLossChart(); renderStepChart(); renderStageMfuChart(); renderThroughputChart(); renderMfuChart();
  renderLrChart(); renderGradChart(); renderMemoryChart(); renderSkewChart(); renderDataChart(); renderEpochChart();
  for (const chart of state.charts.values()) chart.group = 'cosmos3-run';
  connectCharts();
  const xmode = $('#chart-xmode');
  if (xmode) xmode.onchange = async () => {
    state.xMode = xmode.value;
    state.range = 0;
    await loadRunData();
    renderRunView();
  };
  const range = $('#chart-range');
  if (range) range.onchange = () => { state.range = Number(range.value); renderCharts(body); };
  const smooth = $('#chart-smooth');
  if (smooth) smooth.oninput = () => {
    state.smooth = Number(smooth.value);
    const label = $('#chart-smooth-value');
    if (label) label.textContent = String(state.smooth);
    refreshCharts();
  };
  const annot = $('#chart-annot');
  if (annot) annot.onchange = () => { state.showAnnotations = annot.checked; refreshCharts(); };
  const logButton = $('#chart-log');
  if (logButton) logButton.onclick = () => { state.chartLog = !state.chartLog; renderCharts(body); };
  const reset = $('#chart-reset');
  if (reset) reset.onclick = () => { for (const chart of state.charts.values()) chart.dispatchAction({ type: 'dataZoom', start: 0, end: 100 }); };
}

function refreshCharts() {
  if (!state.series) return;
  renderLossChart(); renderStepChart(); renderStageMfuChart(); renderThroughputChart(); renderMfuChart();
  renderLrChart(); renderGradChart(); renderMemoryChart(); renderSkewChart(); renderDataChart(); renderEpochChart();
}

async function openStepDetail(step) {
  if (!state.token) return;
  try {
    const data = await api(`/api/runs/${state.token}/steps/${Math.round(Number(step))}`);
    state.detailStep = Number(data.iteration != null ? data.iteration : step);
    state.detail = data;
    renderStepDrawer();
    const drawer = document.getElementById('step-drawer');
    const backdrop = document.getElementById('drawer-backdrop');
    if (drawer) { drawer.classList.add('open'); drawer.setAttribute('aria-hidden', 'false'); }
    if (backdrop) backdrop.hidden = false;
  } catch (error) {
    toast(error.message, 'bad');
  }
}

function closeStepDetail() {
  state.detailStep = null;
  state.detail = null;
  const drawer = document.getElementById('step-drawer');
  const backdrop = document.getElementById('drawer-backdrop');
  if (drawer) { drawer.classList.remove('open'); drawer.setAttribute('aria-hidden', 'true'); }
  if (backdrop) backdrop.hidden = true;
}

function renderStepDrawer() {
  const d = state.detail;
  if (!d) return;
  const title = document.getElementById('drawer-title');
  const sub = document.getElementById('drawer-sub');
  const body = document.getElementById('drawer-body');
  if (title) title.textContent = `Iteration ${int(d.iteration)}`;
  if (sub) {
    sub.textContent = `Epoch ${d.epoch == null ? '—' : int(d.epoch)} · Batch ${int(d.batch_samples)} samples · micro ${int(d.micro_batch_samples)} × ${int(d.grad_accum)} · world ${int(d.world)}`;
  }
  const stages = d.stages || [];
  const rows = stages.map((stage) => {
    const share = stage.share == null ? null : Math.max(0, Math.min(1, stage.share));
    const bar = share == null ? '<span class="muted">—</span>' : `<div class="sharebar"><span style="width:${(share * 100).toFixed(1)}%"></span></div>`;
    return `<tr><td>${esc(stage.name)}</td><td class="num">${stage.seconds == null ? '—' : `${num(stage.seconds, 3)}s`}</td><td class="num">${share == null ? '—' : pct(share, 1)}</td><td>${bar}</td><td class="num">${stage.mfu == null ? '—' : pct(stage.mfu, 2)}</td><td class="num">${stage.flops == null ? '—' : `${num(stage.flops / 1e12, 2)} TFLOP`}</td><td>${esc(stage.source || '—')}</td><td class="stage-note">${esc(stage.note || '')}</td></tr>`;
  }).join('');
  const loss = d.loss || {};
  const memory = d.memory || {};
  if (body) {
    body.innerHTML = `
      <div class="drawer-kpis">
        <div class="kpi"><div class="label">Iteration time</div><div class="value">${num(d.step_seconds, 2)}<small>s</small></div><div class="hint">${num(d.samples_per_second, 1)} samples/s</div></div>
        <div class="kpi"><div class="label">Step MFU</div><div class="value">${d.total_mfu == null ? '—' : `${num(d.total_mfu * 100, 1)}<small>%</small>`}</div><div class="hint">${esc(d.activation_checkpointing)} checkpointing</div></div>
        <div class="kpi"><div class="label">Loss</div><div class="value">${num(loss.total, 4)}</div><div class="hint">v ${num(loss.vision, 3)} · a ${num(loss.action, 3)}</div></div>
        <div class="kpi"><div class="label">HBM</div><div class="value">${num(memory.device_used_gib, 1)}<small> GiB</small></div><div class="hint">allocated ${num(memory.allocated_gib, 1)} · peak ${num(memory.peak_reserved_gib, 1)}</div></div>
      </div>
      <div class="drawer-actions">
        <button id="drawer-prev" class="ghost">← 上一个 step</button>
        <button id="drawer-next" class="ghost">下一个 step →</button>
        <span class="sep"></span>
        <span class="hint-inline">stage MFU 把 activation recompute 计入 backward；总 step MFU 用 4× forward FLOPs</span>
      </div>
      <table class="table stage-table"><thead><tr><th>阶段</th><th class="num">时间</th><th class="num">占比</th><th>share</th><th class="num">阶段 MFU</th><th class="num">FLOPs</th><th>source</th><th>说明</th></tr></thead><tbody>${rows}</tbody></table>
      <div class="card" style="margin-top:14px"><div class="card-head"><h2>本 step 关键数据</h2><span class="muted small">${d.complete === false ? 'partial aggregate' : 'complete aggregate'}</span></div>
        <div class="kv">
          <div class="k">iteration</div><div class="v">${int(d.iteration)}</div>
          <div class="k">epoch</div><div class="v">${d.epoch == null ? '—' : int(d.epoch)}</div>
          <div class="k">batch (global samples)</div><div class="v">${int(d.batch_samples)}</div>
          <div class="k">micro batch × accum</div><div class="v">${int(d.micro_batch_samples)} × ${int(d.grad_accum)}</div>
          <div class="k">effective global batch</div><div class="v">${int(d.effective_global_batch)}</div>
          <div class="k">grad norm / lr</div><div class="v">${num((d.optim || {}).grad_norm, 3)} / ${num((d.optim || {}).lr, 8)}</div>
          <div class="k">samples / s</div><div class="v">${num(d.samples_per_second, 2)}</div>
          <div class="k">step seconds</div><div class="v">${num(d.step_seconds, 3)}</div>
        </div>
      </div>`;
  }
  const prev = document.getElementById('drawer-prev');
  const next = document.getElementById('drawer-next');
  if (prev) prev.onclick = () => openStepDetail(Number(d.iteration) - 1);
  if (next) next.onclick = () => openStepDetail(Number(d.iteration) + 1);
}

function renderRecords(body) {
  const rows = (state.records && state.records.rows) || [];
  const total = state.records ? state.records.total : 0;
  const table = rows.map((record) => {
    const data = record.data || {};
    const loss = record.loss || {};
    const perf = record.perf || {};
    const mem = record.mem || {};
    const optim = record.optim || {};
    const stepMs = perf.step_ms;
    const samples = data.samples != null ? data.samples : data.partial_samples;
    const throughput = stepMs && samples ? samples / (stepMs / 1000) : null;
    return `<tr class="clickable" data-step="${int(record.step)}"><td class="num">${int(record.step)}</td><td class="num">${int(data.epoch)}</td><td class="num">${int(samples)}</td><td class="num">${clock(record.time)}</td>
      <td class="num">${num(loss.total, 6)}</td><td class="num">${num(loss.vision, 4)}</td><td class="num">${num(loss.action, 4)}</td>
      <td class="num">${num(optim.lr, 8)}</td><td class="num">${num(optim.grad_norm, 3)}</td>
      <td class="num">${num(stepMs != null ? stepMs / 1000 : null, 1)}</td><td class="num">${num(throughput, 1)}</td>
      <td class="num">${num(mem.device_used_gib, 1)}</td><td class="num">${num(perf.skew, 2)}</td>
      <td>${record.complete === false ? '<span class="chip warn">partial</span>' : '<span class="chip ok">complete</span>'}</td></tr>`;
  }).join('');
  body.innerHTML = `<div class="card"><div class="card-head"><h2>记录 <span class="muted small">${total} steps</span></h2><div class="page-actions"><button id="rec-prev" class="ghost">上一页</button><button id="rec-next" class="ghost">下一页</button><a class="small" href="/api/export/${state.token}.csv?keys=${encodeURIComponent(SERIES_KEYS)}">导出 CSV</a></div></div>
    <div class="table-scroll"><table class="table" id="records-table"><thead><tr><th class="num">iter</th><th class="num">epoch</th><th class="num">batch</th><th class="num">time</th><th class="num">loss</th><th class="num">vision</th><th class="num">action</th><th class="num">lr</th><th class="num">grad</th><th class="num">step s</th><th class="num">samples/s</th><th class="num">HBM GiB</th><th class="num">skew</th><th>complete</th></tr></thead><tbody>${table || '<tr><td colspan="14" class="muted">没有记录</td></tr>'}</tbody></table></div></div>`;
  $$('#records-table tr[data-step]').forEach((row) => { row.onclick = () => openStepDetail(Number(row.dataset.step)); });
  const offset = (state.records && state.records.offset) || 0;
  const limit = (state.records && state.records.limit) || 100;
  const load = async (next) => {
    try { state.records = await api(`/api/runs/${state.token}/records?offset=${next}&limit=${limit}&order=desc`); renderRecords(body); } catch (error) { toast(error.message, 'bad'); }
  };
  $('#rec-prev').onclick = () => load(Math.max(0, offset - limit));
  $('#rec-next').onclick = () => load(offset + limit);
  if (!state.records) load(0);
}

function renderConfig(body) {
  const manifest = state.manifest || {};
  const options = state.runs.filter((r) => r.token !== state.token).map((r) => `<option value="${r.token}">${esc(r.display_name)}</option>`).join('');
  body.innerHTML = `<div class="card" style="margin-bottom:12px"><div class="page-actions">
      <button id="cfg-copy" class="ghost">复制 JSON</button>
      <a class="small" href="/api/runs/${state.token}/manifest" download="run.json">下载 manifest</a>
      <span class="sep"></span><label class="hint-inline">与另一个 run 对比</label>
      <select id="cfg-compare"><option value="">选择 run…</option>${options}</select>
    </div><div id="cfg-diff"></div></div>
    <div class="card" id="cfg-tree">${renderConfigTree(manifest)}</div>`;
  $('#cfg-copy').onclick = async () => {
    try { await navigator.clipboard.writeText(JSON.stringify(manifest, null, 2)); toast('manifest 已复制'); } catch (_) { toast('浏览器拒绝剪贴板访问', 'bad'); }
  };
  $('#cfg-compare').onchange = async (event) => {
    const other = event.target.value;
    if (!other) { $('#cfg-diff').innerHTML = ''; return; }
    try {
      const otherManifest = await api(`/api/runs/${other}/manifest`);
      $('#cfg-diff').innerHTML = renderDiff(manifest, otherManifest);
    } catch (error) { toast(error.message, 'bad'); }
  };
}

function renderConfigTree(value, name = '') {
  if (value == null) return `<span class="muted">null</span>`;
  if (Array.isArray(value)) {
    if (!value.length) return '<span class="muted">[]</span>';
    if (value.length > 12 || value.some((item) => typeof item === 'object')) {
      return `<details class="config"><summary>${esc(name || 'array')} · ${value.length} items</summary><pre class="mono">${esc(JSON.stringify(value, null, 2))}</pre></details>`;
    }
    return `<div class="kv"><div class="k">${esc(name)}</div><div class="v mono">${esc(JSON.stringify(value))}</div></div>`;
  }
  if (typeof value === 'object') {
    const keys = Object.keys(value).sort();
    if (!keys.length) return '<span class="muted">{}</span>';
    return `<details class="config" ${name ? '' : 'open'}><summary>${esc(name || 'manifest')} · ${keys.length}</summary>${keys.map((key) => renderConfigTree(value[key], key)).join('')}</details>`;
  }
  return `<div class="kv"><div class="k">${esc(name)}</div><div class="v mono">${esc(String(value))}</div></div>`;
}

function flattenConfig(value, prefix = '', out = {}) {
  if (value == null) { out[prefix] = value; return out; }
  if (Array.isArray(value)) { out[prefix] = JSON.stringify(value); return out; }
  if (typeof value === 'object') {
    for (const key of Object.keys(value)) flattenConfig(value[key], prefix ? `${prefix}.${key}` : key, out);
    return out;
  }
  out[prefix] = String(value);
  return out;
}

function renderDiff(left, right) {
  const a = flattenConfig(left);
  const b = flattenConfig(right);
  const keys = Array.from(new Set([...Object.keys(a), ...Object.keys(b)])).sort();
  const rows = keys.filter((key) => a[key] !== b[key]).map((key) => `<tr><td class="mono">${esc(key)}</td><td class="mono ${key in a ? 'diff-del' : ''}">${esc(a[key] == null ? '—' : a[key])}</td><td class="mono ${key in b ? 'diff-add' : ''}">${esc(b[key] == null ? '—' : b[key])}</td></tr>`).join('');
  return rows ? `<div class="table-scroll" style="max-height:420px"><table class="table"><thead><tr><th>key</th><th>this run</th><th>other run</th></tr></thead><tbody>${rows}</tbody></table></div>` : '<div class="muted small">两份 manifest 没有差异</div>';
}

function renderTopology(body) {
  const rows = (state.run && state.run.topology) || [];
  const latest = {};
  if (state.system && state.system.hosts) {
    for (const [host, entry] of Object.entries(state.system.hosts)) latest[host] = entry.latest;
  }
  const byHost = {};
  for (const row of rows) {
    const host = row.hostname || 'unknown';
    (byHost[host] = byHost[host] || []).push(row);
  }
  const rankState = {};
  if (state.ranks && state.ranks.ranks) for (const item of state.ranks.ranks) rankState[item.rank] = item;
  const nodeCards = Object.keys(byHost).sort().map((host) => {
    const system = latest[host];
    const devices = (system && system.devices) || [];
    const cards = byHost[host].slice().sort((a, b) => (a.local_rank || 0) - (b.local_rank || 0)).map((row) => {
      const rank = row.rank;
      const device = devices.find((d) => Number(d.index) === Number(row.device_index ?? row.local_rank));
      const used = device && device.hbm_used_mib != null && device.hbm_total_mib ? device.hbm_used_mib / device.hbm_total_mib : null;
      const heartbeat = rankState[rank] && rankState[rank].heartbeat;
      const phase = heartbeat ? heartbeat.phase : '—';
      const color = used == null ? 'transparent' : `rgba(57,135,229,${Math.max(0.08, Math.min(0.7, used * 0.7))})`;
      return `<div class="topo-card" style="background:${color}" title="rank ${rank} · ${esc(host)} · ${device ? `${num(device.hbm_used_mib / 1024, 1)}/${num(device.hbm_total_mib / 1024, 1)} GiB · ${num(device.temperature_c, 0)}°C` : 'no device sample'}">
        <div class="idx">NPU ${esc(row.device_index ?? row.local_rank ?? '?')}</div><div class="rank">rank ${esc(rank)}</div><div class="hbm">${device ? `${num(device.hbm_used_mib / 1024, 1)} GiB · ${num(device.temperature_c, 0)}°C` : '<span class="muted">—</span>'}</div><div class="hbm">${esc(phase)}</div>
      </div>`;
    }).join('');
    const empty = Array.from({ length: Math.max(0, 8 - byHost[host].length) }).map(() => '<div class="topo-card empty"></div>').join('');
    return `<div class="topo-node"><div class="topo-node-head"><span>${esc(host)}${(byHost[host][0] || {}).derived ? ' <span class="chip">derived</span>' : ''}</span><span class="muted small">${(byHost[host] || []).length} ranks</span></div><div class="topo-cards">${cards}${empty}</div></div>`;
  }).join('');
  const groupRows = rows.length && rows[0].groups ? Object.entries(rows[0].groups).map(([name, ranks]) => `<tr><td>${esc(name)}</td><td class="mono">${esc((ranks || []).join(', '))}</td></tr>`).join('') : '';
  body.innerHTML = `<div class="card"><div class="card-head"><h2>节点 × 卡</h2><span class="muted small">颜色 = HBM 使用率；悬停查看温度与容量</span></div><div class="topology">${nodeCards || '<div class="muted small">manifest 中没有 topology</div>'}</div></div>
    ${groupRows ? `<div class="card"><div class="card-head"><h2>并行组</h2><span class="muted small">来自 manifest.topology[0].groups</span></div><table class="table"><thead><tr><th>group</th><th>ranks</th></tr></thead><tbody>${groupRows}</tbody></table></div>` : ''}`;
}

function renderCheckpoints(body) {
  const rows = (state.checkpoints && state.checkpoints.checkpoints) || [];
  const schedule = (state.manifest && (state.manifest.checkpoints || state.manifest.checkpoint_schedule)) || {};
  const table = rows.slice().reverse().map((row) => `<tr>
    <td class="num">${int(row.step)}</td><td>${esc(row.kind || '—')}</td><td>${esc(row.status || '—')}</td>
    <td class="num">${row.started_at ? clock(row.started_at) : '—'}</td><td class="num">${row.ended_at ? clock(row.ended_at) : '—'}</td>
    <td class="num">${row.duration_ms != null ? duration(row.duration_ms / 1000) : '—'}</td><td class="num">${bytes(row.bytes)}</td>
    <td class="num">${int(row.files)}</td><td class="mono ellipsis" title="${esc(row.path || '')}">${esc(row.path || '—')}</td><td>${row.async ? 'async' : 'sync'}</td><td>${esc(row.source || 'ledger')}</td>
  </tr>`).join('');
  body.innerHTML = `<div class="card" style="margin-bottom:12px"><div class="page-actions"><span class="chip">计划：每 ${esc(schedule.save_every || schedule.model_export && schedule.model_export.every || '—')} step</span><span class="chip">${esc(schedule.async_save ? 'async' : schedule.model_export && schedule.model_export.async ? 'async' : 'sync')}</span><span class="chip">${esc(state.run.checkpoints ? bytes(state.run.checkpoints.bytes) : '—')} total</span></div></div>
    <div class="grid k2"><div class="card"><div class="chart-head"><h2>大小</h2><span class="unit">bytes</span></div><div id="chart-ckpt-size" class="chart"></div></div><div class="card"><div class="chart-head"><h2>耗时</h2><span class="unit">seconds</span></div><div id="chart-ckpt-duration" class="chart"></div></div></div>
    <div class="card" style="margin-top:12px"><div class="card-head"><h2>Checkpoint 台账</h2><span class="muted small">begin/end 来自 checkpoints.jsonl；缺失时回退扫描 step-*/checkpoint-* 目录</span></div>
    <div class="table-scroll"><table class="table"><thead><tr><th class="num">step</th><th>kind</th><th>status</th><th class="num">started</th><th class="num">ended</th><th class="num">duration</th><th class="num">size</th><th class="num">files</th><th>path</th><th>mode</th><th>source</th></tr></thead><tbody>${table || '<tr><td colspan="11" class="muted">还没有 checkpoint 记录</td></tr>'}</tbody></table></div></div>`;
  renderCheckpointCharts();
}

function renderEvents(body) {
  const events = state.events || [];
  const alerts = state.alerts || { active: [], counts: {} };
  const active = (alerts.active || []).map((alert) => `<div class="banner ${alert.severity === 'critical' || alert.severity === 'error' ? 'bad' : 'warn'}" style="margin-bottom:8px"><span class="icon">!</span><div class="body"><strong>${esc(alert.rule)}</strong> · ${esc(alert.message)}<div class="muted small">first ${clock(alert.first_seen)} · last ${clock(alert.last_seen)} · ×${alert.count}${alert.acked ? ' · acked' : ''}</div></div><button class="ghost" data-ack="${esc(alert.id)}">ack</button></div>`).join('');
  const rows = events.map((event) => `<tr><td class="num">${int(event.step)}</td><td>${severityChip(event.severity)}</td><td>${esc(event.kind)}</td><td class="num">${esc(event.rank ?? '—')}</td><td>${esc(event.message || '')}</td><td class="num">${clock(event.time)}</td></tr>`).join('');
  const rules = [
    ['nonfinite_loss', 'critical', 'loss 为 NaN/Inf'],
    ['loss_spike / loss_drop', 'warn', 'loss 偏离 EMA 3× 或 1/3'],
    ['step_time_jump', 'warn', 'step 超过滚动中位数 1.5×'],
    ['rank_skew', 'warn', '最慢 rank / 平均 > 1.2'],
    ['data_starvation', 'warn', 'dataloader wait > 30% step'],
    ['stalled_progress', 'critical', '心跳存在但 phase/step 长时间不推进'],
    ['stale_heartbeat', 'critical', '心跳文件超过 15s 未更新'],
    ['rank_process_missing', 'error', '心跳 pid 不在进程表'],
    ['checkpoint_failed', 'critical', 'ledger end status=failed'],
    ['checkpoint_slow', 'warn', '保存耗时 > max(120s, 3× 中位数)'],
    ['disk_low', 'critical', 'checkpoint 文件系统空间不足'],
    ['hbm_high / hbm_critical', 'warn / error', '卡 HBM 使用 > 90% / 95%'],
    ['temperature_high', 'warn', 'NPU 温度 > 85°C'],
  ].map(([kind, severity, condition]) => `<tr><td class="mono">${esc(kind)}</td><td>${esc(severity)}</td><td>${esc(condition)}</td></tr>`).join('');
  body.innerHTML = `<div class="card"><div class="card-head"><h2>活跃告警</h2><span class="muted small">仅看板内；AICore% 不参与告警</span></div>${active || '<div class="muted small">没有活跃告警</div>'}</div>
    <div class="card" style="margin-top:12px"><div class="card-head"><h2>事件</h2><span class="muted small">来自 AnomalyDetector 与 monitor 规则</span></div><div class="table-scroll"><table class="table"><thead><tr><th class="num">step</th><th>级别</th><th>kind</th><th class="num">rank</th><th>message</th><th class="num">time</th></tr></thead><tbody>${rows || '<tr><td colspan="6" class="muted">没有事件</td></tr>'}</tbody></table></div></div>
    <details class="config" style="margin-top:12px"><summary>告警规则（节选）</summary><table class="table rule-table"><thead><tr><th>kind</th><th>级别</th><th>条件</th></tr></thead><tbody>${rules}</tbody></table></details>`;
  $$('[data-ack]').forEach((button) => {
    button.onclick = async () => {
      try { await fetch(`/api/runs/${state.token}/alerts/${encodeURIComponent(button.dataset.ack)}/ack`, { method: 'POST' }); state.alerts = await api(`/api/runs/${state.token}/alerts`); renderEvents(body); } catch (error) { toast(error.message, 'bad'); }
    };
  });
}

function renderSystem(body) {
  const hosts = (state.system && state.system.hosts) || {};
  const names = Object.keys(hosts);
  if (!names.length) {
    body.innerHTML = '<div class="card empty"><h2>还没有系统采集</h2><p>启动每节点 sidecar：</p><pre class="mono">python -m examples.worldmodel.cosmos3_monitor collect --run-dir &lt;out&gt; --interval 5</pre></div>';
    return;
  }
  const selected = state.systemHost && hosts[state.systemHost] ? state.systemHost : names[0];
  state.systemHost = selected;
  const entry = hosts[selected];
  const latest = entry.latest || {};
  const host = latest.host || {};
  const devices = latest.devices || [];
  const deviceCards = devices.map((device) => {
    const used = device.hbm_used_mib != null ? device.hbm_used_mib / 1024 : null;
    const total = device.hbm_total_mib != null ? device.hbm_total_mib / 1024 : null;
    const frac = used != null && total ? used / total : null;
    return `<div class="topo-card"><div class="idx">NPU ${esc(device.index)}</div><div class="rank">${num(used, 1)}<span class="muted small">/${num(total, 1)} GiB</span></div><div class="hbm">${frac != null ? pct(frac, 0) : '—'} · ${num(device.temperature_c, 0)}°C · ${num(device.power_w, 0)}W</div><div class="hbm muted">${(device.processes || []).length} proc</div></div>`;
  }).join('');
  const processes = (latest.processes || []).map((proc) => `<tr><td class="num">${int(proc.pid)}</td><td class="num">${proc.rank == null ? '—' : int(proc.rank)}</td><td>${esc(proc.state || '—')}</td><td>${esc(proc.wchan || '—')}</td><td class="mono ellipsis" title="${esc(proc.command || '')}">${esc(proc.command || '')}</td></tr>`).join('');
  body.innerHTML = `<div class="card" style="margin-bottom:12px"><div class="page-actions"><label class="hint-inline">节点</label><select id="sys-host">${names.map((name) => `<option ${name === selected ? 'selected' : ''}>${esc(name)}</option>`).join('')}</select><span class="chip">采集 ${esc(ago(entry.age_s))}</span><span class="chip ${latest.collector && latest.collector.errors && latest.collector.errors.length ? 'bad' : 'ok'}">${latest.collector && latest.collector.errors && latest.collector.errors.length ? esc(latest.collector.errors[0]) : 'collector ok'}</span></div></div>
    <div class="grid k4">
      <div class="kpi"><div class="label">CPU</div><div class="value">${num(host.cpu_percent, 0)}<small>%</small></div><div class="hint">load ${num(host.load1, 1)} / ${num(host.load5, 1)}</div></div>
      <div class="kpi"><div class="label">Memory</div><div class="value">${num((host.memory || {}).available_gib, 0)}<small>GiB free</small></div><div class="hint">${num((host.memory || {}).total_gib, 0)} GiB total</div></div>
      <div class="kpi"><div class="label">/dev/shm</div><div class="value">${num((host.shm || {}).free_gib, 0)}<small>GiB free</small></div><div class="hint">${num((host.shm || {}).total_gib, 0)} GiB total</div></div>
      <div class="kpi"><div class="label">Training processes</div><div class="value">${int(latest.training_processes)}</div><div class="hint">${devices.length} NPUs visible</div></div>
    </div>
    <div class="card" style="margin-top:12px"><div class="card-head"><h2>NPU</h2><span class="muted small">HBM / 温度 / 功率（AICore% 已排除）</span></div><div class="topo-cards">${deviceCards || '<span class="muted small">没有 NPU 数据</span>'}</div></div>
    <div class="grid k2" style="margin-top:12px"><div class="card"><div class="chart-head"><h2>HBM 使用</h2><span class="unit">GiB</span></div><div id="chart-sys-hbm" class="chart"></div></div><div class="card"><div class="chart-head"><h2>温度</h2><span class="unit">°C</span></div><div id="chart-sys-temp" class="chart"></div></div></div>
    <div class="card" style="margin-top:12px"><div class="card-head"><h2>进程</h2><span class="muted small">来自 /proc；rank 由 RANK 环境变量映射</span></div><div class="table-scroll"><table class="table"><thead><tr><th class="num">pid</th><th class="num">rank</th><th>state</th><th>wchan</th><th>command</th></tr></thead><tbody>${processes || '<tr><td colspan="5" class="muted">没有训练进程</td></tr>'}</tbody></table></div></div>`;
  const selector = $('#sys-host');
  if (selector) selector.onchange = (event) => { state.systemHost = event.target.value; renderSystem(body); };
  renderSystemCharts(selected);
}

async function renderLogs(body) {
  body.innerHTML = '<div class="card"><div class="card-head"><h2>日志</h2><div class="page-actions"><button id="log-refresh" class="ghost">刷新</button></div></div><div id="log-body" class="log">读取中…</div></div>';
  const refresh = $('#log-refresh');
  if (refresh) refresh.onclick = () => renderLogs(body);
  try {
    const data = await api(`/api/runs/${state.token}/log?tail=500`);
    const target = $('#log-body');
    if (!target || state.tab !== 'logs') return;  // the user moved to another tab while the fetch was pending
    target.textContent = (data.lines || []).join('\n') || '（空）';
  } catch (error) {
    const target = $('#log-body');
    if (target) target.innerHTML = `<span class="muted">${esc(error.message)}</span>`;
  }
}

/* --------------------------------------------------------------------------------------------------------- */
/* SSE + polling                                                                                              */
/* --------------------------------------------------------------------------------------------------------- */
function closeStream() {
  if (state.stream) { state.stream.close(); state.stream = null; }
  if (state.detailTimer) { clearInterval(state.detailTimer); state.detailTimer = null; }
}

function openStream(token) {
  if (state.stream) { state.stream.close(); state.stream = null; }
  if (typeof EventSource === 'undefined') return;
  try {
    const stream = new EventSource(`/api/runs/${token}/stream`);
    state.stream = stream;
    stream.addEventListener('run', (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (state.paused) return;
        if (payload && payload.data && payload.data.token === token) {
          state.run = payload.data;
          // Never rebuild the whole view here.  A live frame arriving just after a tab click used to replace
          // the tab body with an empty header, which looked like the tab had closed itself.
          updateHeaderMeta();
          renderKpis();
          const badge = $('#live-badge');
          if (badge) { badge.textContent = 'live'; badge.className = 'chip ok'; }
        }
      } catch (_) { /* a malformed frame is ignored */ }
    });
    stream.onerror = () => {
      const badge = $('#live-badge');
      if (badge) { badge.textContent = 'polling'; badge.className = 'chip warn'; }
    };
  } catch (_) { /* EventSource is optional; the 5 s poll still works */ }
}

/* --------------------------------------------------------------------------------------------------------- */
/* Routing / boot                                                                                             */
/* --------------------------------------------------------------------------------------------------------- */
function route() {
  const hash = location.hash.replace(/^#\/?/, '');
  const parts = hash.split('/');
  if (parts[0] === 'run' && parts[1]) {
    openRun(parts[1], parts[2] || 'overview');
  } else if (parts[0] === 'compare') {
    renderComparePage();
  } else {
    state.token = null;
    state.run = null;
    closeStream();
    renderRunsPage();
  }
}

function renderComparePage() {
  const selected = (location.hash.split('?')[1] || '').split('&').filter((x) => x.startsWith('runs='))[0];
  const tokens = selected ? decodeURIComponent(selected.slice(5)).split(',').filter(Boolean) : [];
  state.compareTokens = new Set(tokens);
  const options = state.runs.map((run) => `<option value="${run.token}" ${tokens.includes(run.token) ? 'selected' : ''}>${esc(run.display_name)}</option>`).join('');
  $('#crumbs').innerHTML = '<a href="#/runs">Runs</a> <span>/</span> <strong>Compare</strong>';
  $('#view').innerHTML = `<div class="page-head"><div><h1>对比</h1><div class="subtle">按 step 对齐多个 run；曲线来自各自的 aggregated 记录</div></div></div>
    <div class="card"><div class="page-actions"><label class="hint-inline">选择 run（Ctrl/Cmd 多选）</label><select id="cmp-runs" multiple size="6" style="min-width:320px">${options}</select><button id="cmp-go" class="primary">对比</button></div></div>
    <div class="grid k2" style="margin-top:12px"><div class="card"><div class="chart-head"><h2>Loss</h2><span class="unit">loss</span></div><div id="cmp-loss" class="chart tall"></div></div><div class="card"><div class="chart-head"><h2>Step time</h2><span class="unit">seconds</span></div><div id="cmp-step" class="chart tall"></div></div></div>
    <div class="card" style="margin-top:12px"><div id="cmp-table" class="muted small">选择至少两个 run 后点击对比</div></div>`;
  $('#cmp-go').onclick = async () => {
    const chosen = Array.from($('#cmp-runs').selectedOptions).map((option) => option.value);
    if (chosen.length < 2) { toast('至少选两个 run', 'bad'); return; }
    const query = `run_ids=${chosen.join(',')}&keys=loss/total,perf/step_s,perf/samples_per_s,mem/device_used_gib&max_points=3000`;
    try {
      const data = await api(`/api/compare?${query}`);
      const p = palette();
      const loss = baseOption(); loss.series = []; const step = baseOption(); step.series = [];
      let index = 0;
      for (const [token, entry] of Object.entries(data.runs || {})) {
        const color = p[index % p.length];
        loss.series.push(line(entry.run_id || token, pointsOf(entry.series, 'loss/total'), color));
        step.series.push(line(entry.run_id || token, pointsOf(entry.series, 'perf/step_s'), color));
        index += 1;
      }
      makeChart('cmp-loss', loss); makeChart('cmp-step', step);
      const rows = Object.entries(data.runs || {}).map(([token, entry]) => `<tr><td><a href="#/run/${token}/overview">${esc(entry.run_id || token)}</a></td><td class="num">${pointsOf(entry.series, 'loss/total').length}</td><td class="num">${num((pointsOf(entry.series, 'loss/total').slice(-1)[0] || [])[1], 4)}</td><td class="num">${num((pointsOf(entry.series, 'perf/step_s').slice(-1)[0] || [])[1], 1)}</td></tr>`).join('');
      $('#cmp-table').innerHTML = `<table class="table"><thead><tr><th>run</th><th class="num">points</th><th class="num">last loss</th><th class="num">last step s</th></tr></thead><tbody>${rows}</tbody></table>`;
    } catch (error) { toast(error.message, 'bad'); }
  };
}

function boot() {
  document.documentElement.dataset.theme = state.theme;
  const search = $('#run-search');
  let searchTimer = null;
  search.addEventListener('input', () => {
    state.search = search.value.trim();
    clearTimeout(searchTimer);
    searchTimer = setTimeout(loadRuns, 200);
  });
  const pause = $('#pause-toggle');
  if (pause) {
    pause.onclick = () => {
      state.paused = !state.paused;
      pause.textContent = state.paused ? '继续' : '暂停';
      const badge = $('#live-badge');
      if (badge) {
        badge.textContent = state.paused ? 'paused' : 'live';
        badge.className = state.paused ? 'chip warn' : 'chip ok';
      }
    };
  }
  const drawerClose = $('#drawer-close');
  if (drawerClose) drawerClose.onclick = closeStepDetail;
  const drawerBackdrop = $('#drawer-backdrop');
  if (drawerBackdrop) drawerBackdrop.onclick = closeStepDetail;
  $('#theme-toggle').onclick = () => {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = state.theme;
    localStorage.setItem('c3-theme', state.theme);
    if (state.run) renderRunView();
    else renderRunsPage();
  };
  document.addEventListener('keydown', (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
    if (event.key === 'Escape') { closeStepDetail(); return; }
    if (state.detail && event.key === 'ArrowLeft') { openStepDetail(Number(state.detail.iteration) - 1); return; }
    if (state.detail && event.key === 'ArrowRight') { openStepDetail(Number(state.detail.iteration) + 1); return; }
    if (event.key === '?') toast('快捷键：/ 搜索 · d 明暗 · Esc 关闭详情 · ←/→ 上/下一步', '');
    if (event.key === '/') { event.preventDefault(); search.focus(); }
    if (event.key === 'd') $('#theme-toggle').click();
  });
  window.addEventListener('hashchange', route);
  loadHealth();
  loadRuns().then(route);
  state.runsTimer = setInterval(() => { loadHealth(); loadRuns(); }, 5000);
}

boot();
