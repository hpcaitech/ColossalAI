/* Cosmos3 training monitor — refined telemetry & cluster operations SPA. */
'use strict';

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const GLOBAL_PREFS_KEY = 'c3-monitor-prefs-v1';
const RUN_PREFS_KEY = 'c3-monitor-run-prefs-v1';

function readJsonStorage(storage, key, fallback) {
  try {
    const raw = storage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch (_) {
    return fallback;
  }
}

function writeJsonStorage(storage, key, value) {
  try {
    storage.setItem(key, JSON.stringify(value));
  } catch (_) { /* storage quota or private mode */ }
}

const savedGlobal = readJsonStorage(localStorage, GLOBAL_PREFS_KEY, {});

const state = {
  runs: [],
  total: 0,
  health: null,
  search: '',
  sidebarFilter: savedGlobal.sidebarFilter || 'all',
  runsFilter: savedGlobal.runsFilter || 'all',
  sidebarCollapsed: Boolean(savedGlobal.sidebarCollapsed),
  token: null,
  run: null,
  manifest: null,
  tab: 'overview',
  series: null,
  records: null,
  events: [],
  checkpoints: null,
  control: null,
  system: null,
  ranks: null,
  throughput: null,
  alerts: null,
  loaded: new Set(),
  charts: new Map(),
  chartMeta: new Map(),
  chartDataIndex: new Map(),
  stream: null,
  detailTimer: null,
  runsTimer: null,
  resizeObserver: null,
  focusStep: null,
  chartLog: false,
  xMode: 'step',
  range: 0,
  smooth: 0,
  showAnnotations: true,
  syncZoom: savedGlobal.syncZoom !== false,
  zoomMode: savedGlobal.zoomMode === 'box' ? 'box' : 'pan',
  chartCategory: savedGlobal.chartCategory || 'all',
  zooms: {},
  legendSelected: {},
  systemHost: null,
  logFilter: '',
  logFollow: savedGlobal.logFollow !== false,
  logLines: [],
  paused: false,
  compareTokens: new Set(),
  detailStep: null,
  detail: null,
  modalChartId: null,
  modalRenderer: null,
  lastRenderedTab: null,
  _syncingZoom: false,
  _syncingPointer: false,
  theme: localStorage.getItem('c3-theme') || 'dark',
};

const SERIES_KEYS = [
  'loss/total', 'loss/vision', 'loss/action', 'loss/objective',
  'loss/vision_supervised_mse', 'loss/action_supervised_mse', 'optim/lr', 'optim/grad_norm',
  'perf/step_s', 'perf/samples_per_s', 'perf/mfu',
  'perf/mfu_forward', 'perf/mfu_backward', 'perf/mfu_vae', 'perf/skew',
  'mem/device_used_gib', 'mem/peak_reserved_gib', 'mem/allocated_gib', 'mem/device_total_gib',
  'data/samples', 'data/ar_tokens', 'data/dm_tokens', 'data/wait_s', 'data/vae_s', 'data/epoch',
  'perf/forward_s', 'perf/backward_s', 'perf/optim_s',
].join(',');

function persistGlobalPrefs() {
  writeJsonStorage(localStorage, GLOBAL_PREFS_KEY, {
    sidebarFilter: state.sidebarFilter,
    runsFilter: state.runsFilter,
    sidebarCollapsed: state.sidebarCollapsed,
    syncZoom: state.syncZoom,
    zoomMode: state.zoomMode,
    chartCategory: state.chartCategory,
    logFollow: state.logFollow,
  });
}

function loadRunPrefs(token) {
  if (!token) return;
  const all = readJsonStorage(sessionStorage, RUN_PREFS_KEY, {});
  const prefs = all[token] || {};
  state.xMode = prefs.xMode === 'time' ? 'time' : 'step';
  state.range = Number.isFinite(Number(prefs.range)) ? Number(prefs.range) : 0;
  state.smooth = Number.isFinite(Number(prefs.smooth)) ? Number(prefs.smooth) : 0;
  state.showAnnotations = prefs.showAnnotations !== false;
  state.chartLog = Boolean(prefs.chartLog);
  state.systemHost = prefs.systemHost || null;
  state.zooms = prefs.zooms && typeof prefs.zooms === 'object' ? prefs.zooms : {};
  state.legendSelected = prefs.legendSelected && typeof prefs.legendSelected === 'object' ? prefs.legendSelected : {};
}

function saveRunPrefs() {
  if (!state.token) return;
  const all = readJsonStorage(sessionStorage, RUN_PREFS_KEY, {});
  all[state.token] = {
    xMode: state.xMode,
    range: state.range,
    smooth: state.smooth,
    showAnnotations: state.showAnnotations,
    chartLog: state.chartLog,
    systemHost: state.systemHost,
    zooms: state.zooms,
    legendSelected: state.legendSelected,
  };
  writeJsonStorage(sessionStorage, RUN_PREFS_KEY, all);
}

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

function withAlpha(color, alpha) {
  if (!color) return `rgba(76, 154, 255, ${alpha})`;
  if (color.startsWith('#')) {
    const hex = color.slice(1);
    const full = hex.length === 3 ? hex.split('').map((c) => c + c).join('') : hex;
    const n = parseInt(full, 16);
    if (Number.isFinite(n)) {
      return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
    }
  }
  return color;
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
    stopped: '已停止', initializing: '初始化', unknown: '未知', stale: '过期',
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

async function apiSend(path, method, payload) {
  const response = await fetch(path, {
    method,
    cache: 'no-store',
    headers: payload ? { 'Content-Type': 'application/json' } : undefined,
    body: payload ? JSON.stringify(payload) : undefined,
  });
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
  setTimeout(() => el.remove(), 4800);
}

/* --------------------------------------------------------------------------------------------------------- */
/* Sparkline / small SVG                                                                                      */
/* --------------------------------------------------------------------------------------------------------- */
function sparkline(values, color, width = 96, height = 26) {
  if (!values || values.length < 2) return '<span class="muted small">—</span>';
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const step = width / (values.length - 1);
  const coords = values.map((v, i) => [
    (i * step).toFixed(1),
    (height - 3 - ((v - min) / span) * (height - 6)).toFixed(1),
  ]);
  const linePts = coords.map(([x, y]) => `${x},${y}`).join(' ');
  const fillPts = `0,${height} ${linePts} ${width},${height}`;
  return `<svg class="sparkline" viewBox="0 0 ${width} ${height}" aria-hidden="true">
    <polygon fill="${withAlpha(color, 0.14)}" points="${fillPts}"/>
    <polyline fill="none" stroke="${color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" points="${linePts}"/>
  </svg>`;
}

/* --------------------------------------------------------------------------------------------------------- */
/* Zoom State & ECharts Lifecycle                                                                             */
/* --------------------------------------------------------------------------------------------------------- */
function zoomScopeKey(scope, chartId) {
  const base = scope === 'charts' ? `charts:${state.xMode}`
    : scope === 'overview' ? `overview:${state.xMode}`
      : scope === 'system' ? `system:${state.systemHost || 'default'}`
        : scope || 'charts';
  if ((scope === 'charts' || scope === 'overview') && !state.syncZoom && chartId) {
    return `${base}:${chartId}`;
  }
  return base;
}

function getSavedZoom(scope, chartId) {
  const key = zoomScopeKey(scope, chartId);
  const entry = state.zooms && state.zooms[key];
  if (!entry || !Number.isFinite(entry.startValue) || !Number.isFinite(entry.endValue)) return null;
  if (entry.endValue <= entry.startValue) return null;
  return entry;
}

function setSavedZoom(scope, chartId, rangeObj) {
  const key = zoomScopeKey(scope, chartId);
  if (!state.zooms) state.zooms = {};
  if (!rangeObj) delete state.zooms[key];
  else state.zooms[key] = { startValue: Number(rangeObj.startValue), endValue: Number(rangeObj.endValue) };
  saveRunPrefs();
  updateZoomPill();
}

function formatZoomValue(v, scope) {
  if (!Number.isFinite(v)) return '—';
  if (scope === 'system') return new Date(v * 1000).toLocaleTimeString();
  if (state.xMode === 'time' && (scope === 'charts' || scope === 'overview')) return elapsed(v);
  return int(Math.round(v));
}

function updateZoomPill() {
  const pill = $('#zoom-pill');
  if (!pill) return;
  const z = getSavedZoom('charts', null);
  if (!z || !state.syncZoom) {
    if (!state.syncZoom) {
      const anyCustom = Object.keys(state.zooms || {}).some((k) => k.startsWith(`charts:${state.xMode}:`));
      pill.hidden = !anyCustom;
      if (anyCustom) pill.innerHTML = '独立放缩中 <button class="mini ghost" id="zoom-pill-reset" title="重置全部缩放">重置</button>';
    } else {
      pill.hidden = true;
    }
  } else {
    pill.hidden = false;
    const prefix = state.xMode === 'step' ? 'step ' : '';
    pill.innerHTML = `${prefix}${esc(formatZoomValue(z.startValue, 'charts'))} – ${esc(formatZoomValue(z.endValue, 'charts'))} <button class="mini ghost" id="zoom-pill-reset" title="重置缩放">✕</button>`;
  }
  const btn = $('#zoom-pill-reset');
  if (btn) btn.onclick = () => resetZoomForScope('charts');
}

function resetZoomForScope(scope, onlyChartId = null) {
  if (onlyChartId && !state.syncZoom) {
    setSavedZoom(scope, onlyChartId, null);
  } else if (onlyChartId && state.syncZoom) {
    setSavedZoom(scope, null, null);
  } else {
    const prefix = zoomScopeKey(scope, '');
    for (const k of Object.keys(state.zooms || {})) {
      if (k === prefix || k.startsWith(`${prefix}:`)) delete state.zooms[k];
    }
    saveRunPrefs();
    updateZoomPill();
  }
  state._syncingZoom = true;
  try {
    for (const [id, chart] of state.charts.entries()) {
      const meta = state.chartMeta.get(id);
      if (!meta || meta.scope !== scope) continue;
      if (onlyChartId && !state.syncZoom && meta.id !== onlyChartId) continue;
      chart.dispatchAction({ type: 'dataZoom', start: 0, end: 100 });
    }
  } finally {
    state._syncingZoom = false;
  }
}

function seriesXBounds(option) {
  let min = Infinity;
  let max = -Infinity;
  for (const s of (option.series || [])) {
    for (const pt of (s.data || [])) {
      const x = Array.isArray(pt) ? Number(pt[0]) : NaN;
      if (Number.isFinite(x)) {
        if (x < min) min = x;
        if (x > max) max = x;
      }
    }
  }
  return Number.isFinite(min) && Number.isFinite(max) ? { min, max } : null;
}

function hasSeriesData(option) {
  for (const s of (option.series || [])) {
    if (Array.isArray(s.data) && s.data.length > 0) return true;
  }
  return false;
}

function ensureChartResizeObserver() {
  if (state.resizeObserver || typeof ResizeObserver === 'undefined') return state.resizeObserver;
  state.resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const chart = typeof echarts !== 'undefined' ? echarts.getInstanceByDom(entry.target) : null;
      if (chart && !chart.isDisposed()) chart.resize();
    }
  });
  return state.resizeObserver;
}

function setChartReadout(id, text) {
  const el = document.querySelector(`[data-readout-for="${id}"]`);
  if (el) el.textContent = text || '—';
}

function setChartEmpty(id, empty, message = '暂无可用数据') {
  const el = document.querySelector(`[data-empty-for="${id}"]`);
  if (!el) return;
  el.hidden = !empty;
  if (empty) el.textContent = message;
}

function baseOption() {
  const p = palette();
  return {
    backgroundColor: 'transparent',
    animation: false,
    color: p,
    textStyle: { color: css('--text-2'), fontFamily: '-apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif', fontSize: 11 },
    grid: { left: 56, right: 20, top: 34, bottom: 32, containLabel: false },
    tooltip: {
      trigger: 'axis',
      confine: true,
      backgroundColor: css('--surface-elevated'),
      borderColor: css('--border-strong'),
      borderWidth: 1,
      padding: [8, 10],
      textStyle: { color: css('--text'), fontSize: 11.5 },
      axisPointer: {
        type: 'cross',
        lineStyle: { color: css('--muted'), width: 1, type: 'dashed' },
        crossStyle: { color: css('--muted'), width: 1, type: 'dashed' },
        label: { backgroundColor: css('--surface-3'), color: css('--text'), fontSize: 10 },
      },
      valueFormatter: (value) => (typeof value === 'number' ? value.toLocaleString('en-US', { maximumFractionDigits: 4 }) : value),
    },
    toolbox: {
      show: false,
      feature: { dataZoom: { yAxisIndex: false } },
    },
    xAxis: {
      type: 'value',
      // Integer ticks are mandatory when the axis is an optimizer step.  In elapsed-time mode the axis is
      // seconds from the first record and gets a clock formatter instead.
      minInterval: state.xMode === 'step' ? 1 : null,
      axisLine: { lineStyle: { color: css('--grid') } },
      axisTick: { lineStyle: { color: css('--grid') } },
      axisLabel: {
        color: css('--muted'),
        fontSize: 10,
        fontFamily: css('--mono'),
        formatter: state.xMode === 'step' ? (v) => (v >= 1000 ? `${+(v / 1000).toFixed(2)}k` : v) : (v) => elapsed(v),
      },
      splitLine: { show: false },
      scale: true,
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: css('--muted'), fontSize: 10, fontFamily: css('--mono') },
      splitLine: { lineStyle: { color: css('--grid'), width: 1, type: 'dashed' } },
      scale: true,
    },
    legend: {
      // The unit lives in the card header.  Keeping it out of the ECharts axis name is what stops the
      // "device used" / "GiB" and "skew" / "/ mean" collisions in the top-left corner.
      type: 'scroll',
      top: 0,
      left: 0,
      right: 0,
      icon: 'roundRect',
      itemWidth: 10,
      itemHeight: 6,
      textStyle: { color: css('--text-2'), fontSize: 11 },
      itemGap: 12,
      pageIconSize: 8,
      pageTextStyle: { color: css('--muted'), fontSize: 10 },
    },
    dataZoom: [{
      id: 'c3-dz-x',
      type: 'inside',
      filterMode: 'none',
      rangeMode: ['value', 'value'],
      zoomOnMouseWheel: true,
      moveOnMouseMove: state.zoomMode !== 'box',
      moveOnMouseWheel: false,
      preventDefaultMouseMove: true,
    }],
  };
}

function areaFill(color, topAlpha = 0.22) {
  return {
    color: {
      type: 'linear',
      x: 0, y: 0, x2: 0, y2: 1,
      colorStops: [
        { offset: 0, color: withAlpha(color, topAlpha) },
        { offset: 1, color: withAlpha(color, 0.01) },
      ],
    },
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
    emphasis: { focus: 'series', lineStyle: { width: 2.6 } },
    data: points,
    connectNulls: false,
  }, extra);
}

function pointsFromItem(item) {
  if (!item || !Array.isArray(item.x)) return [];
  let pairs = item.x.map((x, i) => [x, item.y[i]]).filter((pair) => pair[1] != null);
  if (state.range && state.xMode === 'step' && pairs.length) {
    const last = Math.max(...pairs.map((pair) => pair[0]));
    pairs = pairs.filter((pair) => pair[0] >= last - state.range + 1);
  } else if (state.range && state.xMode === 'time' && pairs.length) {
    pairs = pairs.slice(-state.range);
  }
  return pairs;
}

function pointsOf(series, key, ignoreRange = false) {
  const item = series && series[key];
  if (!item) return [];
  let pairs = item.x.map((x, i) => [x, item.y[i]]).filter((pair) => pair[1] != null);
  if (!ignoreRange && state.range && state.xMode === 'step' && pairs.length) {
    const last = Math.max(...pairs.map((pair) => pair[0]));
    const first = last - state.range + 1;
    pairs = pairs.filter((pair) => pair[0] >= first);
  } else if (!ignoreRange && state.range && state.xMode === 'time' && pairs.length) {
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

function annotationSummary() {
  if (!state.showAnnotations || state.xMode !== 'step') return '';
  const eventSteps = new Set();
  for (const event of (state.events || []).filter((item) => ['warn', 'error', 'critical'].includes(item.severity))) {
    const step = Number(event.step);
    if (Number.isFinite(step)) eventSteps.add(step);
  }
  const checkpointSteps = new Set();
  for (const checkpoint of (state.checkpoints && state.checkpoints.checkpoints) || []) {
    const step = Number(checkpoint.step);
    if (Number.isFinite(step)) checkpointSteps.add(step);
  }
  const parts = [];
  if (eventSteps.size) {
    parts.push(`${eventSteps.size} 个告警 step${eventSteps.size > 40 ? '（图中最近 40 个）' : ''}`);
  }
  if (checkpointSteps.size) {
    parts.push(`${checkpointSteps.size} 个 checkpoint${checkpointSteps.size > 40 ? '（图中最近 40 个）' : ''}`);
  }
  if (!parts.length) return '暂无告警/ckpt标注';
  const dense = eventSteps.size > 8 || checkpointSteps.size > 8;
  return `${parts.join(' · ')}${dense ? '；过密用 ! / ◆' : ''}`;
}

function updateAnnotationNote() {
  const note = $('#annot-note');
  if (note) note.textContent = annotationSummary();
}

function annotationLines() {
  if (!state.showAnnotations || state.xMode !== 'step') return [];
  const rows = [];
  if (state.focusStep != null) {
    rows.push({
      xAxis: state.focusStep,
      lineStyle: { color: css('--warning'), type: 'dashed', width: 1 },
      label: { formatter: `step ${state.focusStep}`, color: css('--warning'), fontSize: 10, position: 'insideEndTop' },
    });
  }
  const seen = new Set();
  const checkpoints = [];
  for (const checkpoint of (state.checkpoints && state.checkpoints.checkpoints) || []) {
    const step = Number(checkpoint.step);
    if (!Number.isFinite(step) || seen.has(`c${step}`)) continue;
    seen.add(`c${step}`);
    checkpoints.push(step);
  }
  const eventByStep = new Map();
  for (const event of (state.events || []).filter((item) => ['warn', 'error', 'critical'].includes(item.severity))) {
    const step = Number(event.step);
    if (!Number.isFinite(step)) continue;
    const entry = eventByStep.get(step) || { kinds: new Set(), count: 0 };
    entry.kinds.add(String(event.kind || 'event'));
    entry.count += 1;
    eventByStep.set(step, entry);
  }
  // One line per step, not one per host/device event.  A long-running run can have thousands of events; the
  // chart only annotates the most recent window, and dense windows use a one-character label so the labels
  // cannot run into each other along the top edge.
  const eventSteps = [...eventByStep.keys()].sort((a, b) => a - b).slice(-40);
  const denseEvents = eventSteps.length > 8;
  for (const step of eventSteps) {
    const entry = eventByStep.get(step);
    const kinds = [...entry.kinds];
    const text = denseEvents ? '!' : (kinds.length === 1 ? kinds[0] : `${kinds[0]} +${kinds.length - 1}`);
    rows.push({
      xAxis: step,
      lineStyle: { color: css('--warning'), type: 'dashed', width: 1, opacity: denseEvents ? 0.5 : 1 },
      label: { formatter: text, color: css('--warning'), fontSize: 9, position: 'insideEndTop' },
    });
  }
  const recentCheckpoints = checkpoints.slice(-40);
  const denseCheckpoints = recentCheckpoints.length > 8;
  for (const step of recentCheckpoints) {
    rows.push({
      xAxis: step,
      lineStyle: { color: css('--good'), type: 'dotted', width: 1, opacity: denseCheckpoints ? 0.65 : 1 },
      label: { formatter: denseCheckpoints ? '◆' : 'ckpt', color: css('--good'), fontSize: 9, position: 'insideEndTop' },
    });
  }
  return rows;
}

function markLines() {
  return annotationLines();
}

function findNearestDataIndex(points, targetX) {
  if (!points || !points.length) return -1;
  let lo = 0;
  let hi = points.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (points[mid][0] < targetX) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(points[lo - 1][0] - targetX) < Math.abs(points[lo][0] - targetX)) {
    return lo - 1;
  }
  return lo;
}

function syncSiblingZoom(sourceId, scope, saved) {
  state._syncingZoom = true;
  try {
    for (const [id, chart] of state.charts.entries()) {
      if (id === sourceId || chart.isDisposed()) continue;
      const meta = state.chartMeta.get(id);
      if (!meta || meta.scope !== scope) continue;
      if (saved) {
        chart.dispatchAction({ type: 'dataZoom', startValue: saved.startValue, endValue: saved.endValue });
      } else {
        chart.dispatchAction({ type: 'dataZoom', start: 0, end: 100 });
      }
    }
  } finally {
    state._syncingZoom = false;
  }
}

function makeChart(id, option, scope = 'charts') {
  const el = document.getElementById(id);
  if (!el || typeof echarts === 'undefined') return null;
  const existing = echarts.getInstanceByDom(el);
  const chart = existing || echarts.init(el, null, { renderer: 'canvas' });
  const sourceChartId = id === 'chart-modal-canvas' && state.modalChartId ? state.modalChartId : id;

  // Apply persisted zoom by exact X data values (startValue/endValue) so refreshes never drift or reset.
  const bounds = seriesXBounds(option);
  const savedZoom = getSavedZoom(scope, sourceChartId);
  const dzList = Array.isArray(option.dataZoom) && option.dataZoom.length
    ? option.dataZoom
    : [{ id: 'c3-dz-x', type: 'inside', filterMode: 'none' }];
  const dz0 = Object.assign({}, dzList[0], {
    id: 'c3-dz-x',
    type: 'inside',
    filterMode: 'none',
    rangeMode: ['value', 'value'],
    zoomOnMouseWheel: true,
    moveOnMouseMove: state.zoomMode !== 'box',
    moveOnMouseWheel: false,
    preventDefaultMouseMove: true,
  });
  const zoomOverlaps = savedZoom && (!bounds || (savedZoom.endValue >= bounds.min && savedZoom.startValue <= bounds.max));
  if (zoomOverlaps) {
    delete dz0.start;
    delete dz0.end;
    dz0.startValue = savedZoom.startValue;
    dz0.endValue = savedZoom.endValue;
  } else {
    delete dz0.startValue;
    delete dz0.endValue;
    dz0.start = 0;
    dz0.end = 100;
  }
  option.dataZoom = [dz0];

  if (!option.toolbox) {
    option.toolbox = { show: false, feature: { dataZoom: { yAxisIndex: false } } };
  }
  if (option.legend && state.legendSelected[sourceChartId]) {
    option.legend = Object.assign({}, option.legend, { selected: state.legendSelected[sourceChartId] });
  }

  // Track first series' sorted [x, y] pairs for fast O(log N) crosshair hover sync.
  const firstLine = (option.series || []).find((s) => s.type !== 'heatmap' && Array.isArray(s.data) && s.data.length);
  if (firstLine) state.chartDataIndex.set(id, firstLine.data);
  else state.chartDataIndex.delete(id);

  setChartEmpty(id, !hasSeriesData(option));

  // Replacing every option resets the user's zoom on each 5 s refresh.  Keep the viewport and replace only the
  // data series while explicitly pinning startValue/endValue in dataZoom.
  chart.setOption(option, existing ? { notMerge: false, replaceMerge: ['series', 'visualMap'] } : true);

  if (state.zoomMode === 'box' && (scope === 'charts' || id === 'chart-modal-canvas')) {
    chart.dispatchAction({ type: 'takeGlobalCursor', key: 'dataZoomSelect', dataZoomSelectActive: true });
  } else {
    chart.dispatchAction({ type: 'takeGlobalCursor', key: 'dataZoomSelect', dataZoomSelectActive: false });
  }

  state.chartMeta.set(id, { id: sourceChartId, domId: id, scope });
  state.charts.set(id, chart);

  const ro = ensureChartResizeObserver();
  if (ro) ro.observe(el);

  if (!chart.__c3Bound) {
    chart.on('datazoom', () => {
      if (state._syncingZoom) return;
      const meta = state.chartMeta.get(id);
      if (!meta) return;
      const dz = (chart.getOption().dataZoom || [])[0];
      if (!dz) return;
      if (dz.start <= 0.05 && dz.end >= 99.95) {
        setSavedZoom(meta.scope, meta.id, null);
        if (state.syncZoom || meta.domId === 'chart-modal-canvas') syncSiblingZoom(id, meta.scope, null);
      } else if (Number.isFinite(dz.startValue) && Number.isFinite(dz.endValue) && dz.endValue > dz.startValue) {
        const next = { startValue: dz.startValue, endValue: dz.endValue };
        setSavedZoom(meta.scope, meta.id, next);
        if (state.syncZoom || meta.domId === 'chart-modal-canvas') syncSiblingZoom(id, meta.scope, next);
      }
    });
    chart.on('dblclick', () => {
      const meta = state.chartMeta.get(id);
      if (meta) resetZoomForScope(meta.scope, state.syncZoom ? null : meta.id);
    });
    chart.on('legendselectchanged', (params) => {
      const meta = state.chartMeta.get(id);
      if (!meta || !params || !params.selected) return;
      state.legendSelected[meta.id] = params.selected;
      saveRunPrefs();
    });
    chart.on('updateAxisPointer', (event) => {
      if (state._syncingPointer || !state.syncZoom) return;
      const meta = state.chartMeta.get(id);
      if (!meta || (meta.scope !== 'charts' && meta.scope !== 'overview' && meta.scope !== 'compare')) return;
      const axisInfo = event && event.axesInfo && event.axesInfo[0];
      if (!axisInfo || !Number.isFinite(Number(axisInfo.value))) return;
      const targetX = Number(axisInfo.value);
      state._syncingPointer = true;
      try {
        for (const [otherId, otherChart] of state.charts.entries()) {
          if (otherId === id || otherChart.isDisposed()) continue;
          const otherMeta = state.chartMeta.get(otherId);
          if (!otherMeta || otherMeta.scope !== meta.scope) continue;
          const pts = state.chartDataIndex.get(otherId);
          if (!pts || !pts.length) continue;
          const idx = findNearestDataIndex(pts, targetX);
          if (idx >= 0) otherChart.dispatchAction({ type: 'showTip', seriesIndex: 0, dataIndex: idx });
        }
      } finally {
        state._syncingPointer = false;
      }
    });
    chart.on('globalout', () => {
      if (state._syncingPointer || !state.syncZoom) return;
      const meta = state.chartMeta.get(id);
      if (!meta) return;
      state._syncingPointer = true;
      try {
        for (const [otherId, otherChart] of state.charts.entries()) {
          if (otherId === id || otherChart.isDisposed()) continue;
          const otherMeta = state.chartMeta.get(otherId);
          if (otherMeta && otherMeta.scope === meta.scope) otherChart.dispatchAction({ type: 'hideTip' });
        }
      } finally {
        state._syncingPointer = false;
      }
    });
    chart.on('click', (params) => {
      const meta = state.chartMeta.get(id);
      if (!meta || (meta.scope !== 'charts' && meta.scope !== 'overview' && meta.scope !== 'checkpoints')) return;
      if (state.xMode !== 'step' && meta.scope !== 'checkpoints') return;
      const value = params && params.value;
      const step = Array.isArray(value) ? value[0] : value;
      if (Number.isFinite(Number(step))) openStepDetail(Math.round(Number(step)));
    });
    chart.__c3Bound = true;
  }
  return chart;
}

function disposeCharts() {
  if (state.resizeObserver) {
    state.resizeObserver.disconnect();
  }
  for (const chart of state.charts.values()) {
    try { chart.dispose(); } catch (_) { /* the DOM may already be gone */ }
  }
  state.charts.clear();
  state.chartMeta.clear();
  state.chartDataIndex.clear();
}

function applyZoomModeToCharts() {
  for (const [id, chart] of state.charts.entries()) {
    if (chart.isDisposed()) continue;
    const meta = state.chartMeta.get(id);
    if (!meta || (meta.scope !== 'charts' && id !== 'chart-modal-canvas')) continue;
    chart.setOption({ dataZoom: [{ id: 'c3-dz-x', moveOnMouseMove: state.zoomMode !== 'box' }] }, false);
    chart.dispatchAction({ type: 'takeGlobalCursor', key: 'dataZoomSelect', dataZoomSelectActive: state.zoomMode === 'box' });
  }
}

/* --------------------------------------------------------------------------------------------------------- */
/* Individual chart renderers                                                                                 */
/* --------------------------------------------------------------------------------------------------------- */
function lastY(points) {
  return points && points.length ? points[points.length - 1][1] : null;
}

function renderLossChart(id = 'chart-loss', tall = false, scope = 'charts') {
  if (!state.series) return;
  const p = palette();
  const option = baseOption();
  option.yAxis.type = state.chartLog ? 'log' : 'value';
  const objective = pointsOf(state.series, 'loss/objective');
  const hasObjective = objective.some((point) => point[1] != null);
  const total = hasObjective ? objective : pointsOf(state.series, 'loss/total');
  const vision = pointsOf(state.series, 'loss/vision');
  const action = pointsOf(state.series, 'loss/action');
  const visionSupervised = pointsOf(state.series, 'loss/vision_supervised_mse');
  const actionSupervised = pointsOf(state.series, 'loss/action_supervised_mse');
  const hasSupervised = visionSupervised.some((point) => point[1] != null)
    || actionSupervised.some((point) => point[1] != null);
  option.series = [
    line(hasObjective ? 'training objective' : 'loss', total, p[0], {
      lineStyle: { width: 2.25, color: p[0] },
      areaStyle: state.chartLog ? undefined : areaFill(p[0], 0.16),
      markLine: { symbol: 'none', data: markLines(), silent: true },
    }),
    line(hasObjective ? 'vision (upstream mean)' : 'vision', vision, p[1], { lineStyle: { width: 1.6, color: p[1] } }),
    line(hasObjective ? 'action (upstream mean)' : 'action', action, p[2], { lineStyle: { width: 1.6, color: p[2] } }),
  ];
  if (hasSupervised) {
    option.series.push(
      line('vision (supervised-element MSE)', visionSupervised, p[4], {
        lineStyle: { width: 1.3, color: p[4], type: 'dashed' },
      }),
      line('action (supervised-element MSE)', actionSupervised, p[5], {
        lineStyle: { width: 1.3, color: p[5], type: 'dashed' },
      }),
    );
  }
  if (state.smooth) {
    option.series.push(line(`total (EMA ${state.smooth})`, smoothPoints(total, state.smooth), css('--text-2'), {
      lineStyle: { width: 2.2, color: css('--text-2'), type: 'dashed' }, itemStyle: { color: css('--text-2') },
    }));
  }
  const latest = lastY(total);
  setChartReadout(id, latest != null ? num(latest, 4) : '—');
  makeChart(id, option, scope);
}

function renderStepChart(id = 'chart-step', scope = 'charts') {
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
    stack: 'step',
    areaStyle: { opacity: 0.28, color },
    lineStyle: { width: 1.2, color },
  }));
  const totalPts = pointsOf(state.series, 'perf/step_s');
  option.series.push(line('total', totalPts, css('--text-2'), {
    stack: undefined,
    lineStyle: { width: 1.75, type: 'dashed', color: css('--text-2') },
    markLine: { symbol: 'none', data: markLines(state.focusStep), silent: true },
  }));
  const latest = lastY(totalPts);
  setChartReadout(id, latest != null ? `${num(latest, 2)}s` : '—');
  makeChart(id, option, scope);
}

function renderThroughputChart(id = 'chart-throughput', scope = 'charts') {
  if (!state.series) return;
  const p = palette();
  const option = baseOption();
  const total = pointsOf(state.series, 'perf/samples_per_s');
  option.series = [
    line('total', total, p[0], {
      lineStyle: { width: 2.4, color: p[0] },
      areaStyle: areaFill(p[0], 0.14),
    }),
  ];
  const nodes = (state.throughput && state.throughput.nodes) || {};
  Object.keys(nodes).sort().forEach((host, index) => {
    option.series.push(line(host, pointsFromItem(nodes[host]), p[(index + 1) % p.length], {
      lineStyle: { width: 1.25, color: p[(index + 1) % p.length], opacity: 0.85 },
    }));
  });
  const latest = lastY(total);
  setChartReadout(id, latest != null ? `${num(latest, 1)} /s` : '—');
  makeChart(id, option, scope);
}

function renderRankHeatmap(id = 'chart-rank-heat', scope = 'charts') {
  const ranks = (state.throughput && state.throughput.ranks) || {};
  const keys = Object.keys(ranks).map(Number).sort((a, b) => a - b);
  const rankIndex = new Map(keys.map((rank, index) => [rank, index]));
  const data = [];
  let maximum = 0;
  for (const rank of keys) {
    for (const [x, y] of pointsFromItem(ranks[rank])) {
      data.push([x, rankIndex.get(rank), y]);
      if (y > maximum) maximum = y;
    }
  }
  const light = state.theme === 'light';
  const option = {
    backgroundColor: 'transparent',
    animation: false,
    textStyle: { color: css('--text-2'), fontSize: 11 },
    grid: { left: 52, right: 74, top: 18, bottom: 32 },
    tooltip: {
      trigger: 'item',
      confine: true,
      backgroundColor: css('--surface-elevated'),
      borderColor: css('--border-strong'),
      textStyle: { color: css('--text'), fontSize: 11.5 },
      formatter: (params) => `rank <strong>${keys[params.value[1]]}</strong> · ${state.xMode === 'step' ? 'step' : 't'} ${params.value[0]}<br>${num(params.value[2], 2)} samples/s`,
    },
    toolbox: { show: false, feature: { dataZoom: { yAxisIndex: false } } },
    xAxis: {
      type: 'value',
      scale: true,
      minInterval: state.xMode === 'step' ? 1 : null,
      axisLine: { lineStyle: { color: css('--grid') } },
      axisLabel: {
        color: css('--muted'),
        fontSize: 10,
        fontFamily: css('--mono'),
        formatter: state.xMode === 'step' ? (v) => (v >= 1000 ? `${+(v / 1000).toFixed(2)}k` : v) : (v) => elapsed(v),
      },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'category',
      data: keys.map(String),
      axisLine: { lineStyle: { color: css('--grid') } },
      axisLabel: { color: css('--muted'), fontSize: 10, fontFamily: css('--mono'), interval: 'auto' },
      splitLine: { show: false },
    },
    visualMap: {
      min: 0,
      max: maximum || 1,
      calculable: true,
      orient: 'vertical',
      right: 4,
      top: 18,
      bottom: 20,
      itemWidth: 10,
      inRange: {
        color: light
          ? ['#e6effd', '#60a5fa', '#1d4ed8']
          : ['#16263d', '#2f6ec4', '#4c9aff', '#22c58b'],
      },
      textStyle: { color: css('--muted'), fontSize: 10, fontFamily: css('--mono') },
    },
    dataZoom: [{ id: 'c3-dz-x', type: 'inside', filterMode: 'none', rangeMode: ['value', 'value'] }],
    series: [{
      type: 'heatmap',
      data,
      progressive: 2000,
      emphasis: { itemStyle: { borderColor: css('--text'), borderWidth: 1 } },
    }],
  };
  setChartReadout(id, keys.length ? `${keys.length} ranks` : '—');
  makeChart(id, option, scope);
}

function renderMfuChart(id = 'chart-mfu', scope = 'charts') {
  if (!state.series) return;
  const p = palette();
  const option = baseOption();
  option.yAxis.axisLabel.formatter = (v) => `${Number(v).toFixed(0)}%`;
  const points = pointsOf(state.series, 'perf/mfu').map(([x, y]) => [x, y * 100]);
  option.series = [line('MFU', points, p[2], { areaStyle: areaFill(p[2], 0.18) })];
  const latest = lastY(points);
  setChartReadout(id, latest != null ? `${num(latest, 1)}%` : '—');
  makeChart(id, option, scope);
}

function renderLrChart(id = 'chart-lr', scope = 'charts') {
  if (!state.series) return;
  const p = palette();
  const option = baseOption();
  const points = pointsOf(state.series, 'optim/lr');
  option.series = [line('learning rate', points, p[0], { areaStyle: areaFill(p[0], 0.12) })];
  const latest = lastY(points);
  setChartReadout(id, latest != null ? Number(latest).toExponential(2) : '—');
  makeChart(id, option, scope);
}

function renderGradChart(id = 'chart-grad', scope = 'charts') {
  if (!state.series) return;
  const p = palette();
  const option = baseOption();
  const points = pointsOf(state.series, 'optim/grad_norm');
  option.series = [line('grad norm', points, p[1], { areaStyle: areaFill(p[1], 0.14) })];
  const latest = lastY(points);
  setChartReadout(id, latest != null ? num(latest, 2) : '—');
  makeChart(id, option, scope);
}

function renderMemoryChart(id = 'chart-mem', scope = 'charts') {
  if (!state.series) return;
  const p = palette();
  const option = baseOption();
  const used = pointsOf(state.series, 'mem/device_used_gib');
  option.series = [
    line('device used', used, p[0], { areaStyle: areaFill(p[0], 0.12) }),
    line('peak reserved', pointsOf(state.series, 'mem/peak_reserved_gib'), p[1]),
    line('allocated', pointsOf(state.series, 'mem/allocated_gib'), p[2]),
    line('device total', pointsOf(state.series, 'mem/device_total_gib'), css('--muted'), { lineStyle: { width: 1.5, type: 'dashed', color: css('--muted') } }),
  ];
  const latest = lastY(used);
  setChartReadout(id, latest != null ? `${num(latest, 1)} GiB` : '—');
  makeChart(id, option, scope);
}

function renderSkewChart(id = 'chart-skew', scope = 'charts') {
  if (!state.series) return;
  const p = palette();
  const option = baseOption();
  const points = pointsOf(state.series, 'perf/skew');
  option.series = [line('skew', points, p[4], {
    areaStyle: areaFill(p[4], 0.12),
    markLine: {
      symbol: 'none', silent: true,
      data: [{ yAxis: 1.2, lineStyle: { color: css('--warning'), type: 'dashed' }, label: { formatter: '1.2×', color: css('--warning'), fontSize: 10 } }],
    },
  })];
  const latest = lastY(points);
  setChartReadout(id, latest != null ? `${num(latest, 2)}×` : '—');
  makeChart(id, option, scope);
}

function renderDataChart(id = 'chart-data', scope = 'charts') {
  if (!state.series) return;
  const p = palette();
  const option = baseOption();
  option.yAxis.type = 'log';
  const samples = pointsOf(state.series, 'data/samples');
  option.series = [
    line('samples', samples, p[0]),
    line('AR tokens', pointsOf(state.series, 'data/ar_tokens'), p[1]),
    line('DM tokens', pointsOf(state.series, 'data/dm_tokens'), p[2]),
  ];
  const latest = lastY(samples);
  setChartReadout(id, latest != null ? int(latest) : '—');
  makeChart(id, option, scope);
}

function renderStageMfuChart(id = 'chart-stage-mfu', scope = 'charts') {
  if (!state.series) return;
  const p = palette();
  const option = baseOption();
  option.yAxis.name = '';
  option.yAxis.axisLabel.formatter = (value) => `${value.toFixed(0)}%`;
  const series = [];
  const stage = (name, key, color) => {
    const values = pointsOf(state.series, key).map(([x, y]) => [x, y * 100]);
    if (values.length) series.push(line(name, values, color));
  };
  // All three values are recorded by the trainer with the model geometry and checkpointing mode it ran.
  stage('forward', 'perf/mfu_forward', p[0]);
  stage('backward', 'perf/mfu_backward', p[1]);
  stage('VAE', 'perf/mfu_vae', p[2]);
  option.series = series;
  const fwdLast = series.length ? lastY(series[0].data) : null;
  setChartReadout(id, fwdLast != null ? `fwd ${num(fwdLast, 1)}%` : '—');
  makeChart(id, option, scope);
}

function renderEpochChart(id = 'chart-epoch', scope = 'charts') {
  if (!state.series) return;
  const p = palette();
  const option = baseOption();
  option.yAxis.name = '';
  const points = pointsOf(state.series, 'data/epoch');
  option.series = [line('epoch', points, p[3], { step: 'end', areaStyle: areaFill(p[3], 0.14) })];
  const latest = lastY(points);
  setChartReadout(id, latest != null ? `ep ${int(latest)}` : '—');
  makeChart(id, option, scope);
}

function renderCheckpointCharts() {
  const rows = (state.checkpoints && state.checkpoints.checkpoints) || [];
  const p = palette();
  const sizeOption = baseOption();
  sizeOption.xAxis.minInterval = 1;
  sizeOption.xAxis.axisLabel.formatter = (v) => (v >= 1000 ? `${+(v / 1000).toFixed(2)}k` : v);
  sizeOption.yAxis.axisLabel.formatter = (v) => bytes(v);
  sizeOption.series = [{
    name: 'checkpoint size',
    type: 'bar',
    barMaxWidth: 20,
    itemStyle: { color: p[0], borderRadius: [4, 4, 0, 0] },
    data: rows.filter((r) => r.bytes != null).map((r) => [Number(r.step) || 0, Number(r.bytes)]),
  }];
  makeChart('chart-ckpt-size', sizeOption, 'checkpoints');

  const durationOption = baseOption();
  durationOption.xAxis.minInterval = 1;
  durationOption.xAxis.axisLabel.formatter = (v) => (v >= 1000 ? `${+(v / 1000).toFixed(2)}k` : v);
  durationOption.yAxis.axisLabel.formatter = (v) => `${Number(v).toFixed(1)}s`;
  durationOption.series = [{
    name: 'save duration',
    type: 'bar',
    barMaxWidth: 20,
    itemStyle: { color: p[2], borderRadius: [4, 4, 0, 0] },
    data: rows.filter((r) => r.duration_ms != null).map((r) => [Number(r.step) || 0, Number(r.duration_ms) / 1000]),
  }];
  makeChart('chart-ckpt-duration', durationOption, 'checkpoints');
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
  hbmOption.xAxis.minInterval = null;
  hbmOption.xAxis.axisLabel.formatter = (v) => new Date(v * 1000).toLocaleTimeString();
  makeChart('chart-sys-hbm', hbmOption, 'system');

  const tempOption = baseOption();
  tempOption.series = devices.map((index, i) => line(`NPU ${index}`, (entry.devices[index].temperature_c || []).map(([t, v]) => [t, v]), p[i % p.length]));
  tempOption.xAxis.minInterval = null;
  tempOption.xAxis.axisLabel.formatter = (v) => new Date(v * 1000).toLocaleTimeString();
  makeChart('chart-sys-temp', tempOption, 'system');
}

const CHART_RENDERERS = {
  'chart-loss': { title: 'Loss', unit: 'loss', fn: (targetId, scope) => renderLossChart(targetId, true, scope) },
  'chart-loss-overview': { title: 'Loss', unit: 'step', fn: (targetId, scope) => renderLossChart(targetId, false, scope) },
  'chart-step': { title: 'Step time', unit: 'seconds', fn: (targetId, scope) => renderStepChart(targetId, scope) },
  'chart-step-overview': { title: 'Step time', unit: 'seconds', fn: (targetId, scope) => renderStepChart(targetId, scope) },
  'chart-stage-mfu': { title: 'Stage MFU', unit: '% · forward / backward / VAE', fn: (targetId, scope) => renderStageMfuChart(targetId, scope) },
  'chart-throughput': { title: 'Throughput', unit: 'samples/s · total + per node', fn: (targetId, scope) => renderThroughputChart(targetId, scope) },
  'chart-rank-heat': { title: 'Rank throughput', unit: 'samples/s · rank × step', fn: (targetId, scope) => renderRankHeatmap(targetId, scope) },
  'chart-mfu': { title: 'MFU', unit: '% of step', fn: (targetId, scope) => renderMfuChart(targetId, scope) },
  'chart-lr': { title: 'Learning rate', unit: 'lr', fn: (targetId, scope) => renderLrChart(targetId, scope) },
  'chart-grad': { title: 'Gradient norm', unit: 'norm', fn: (targetId, scope) => renderGradChart(targetId, scope) },
  'chart-mem': { title: 'Memory', unit: 'GiB', fn: (targetId, scope) => renderMemoryChart(targetId, scope) },
  'chart-skew': { title: 'Rank skew', unit: 'max / mean', fn: (targetId, scope) => renderSkewChart(targetId, scope) },
  'chart-data': { title: 'Batch / tokens per step', unit: 'log count', fn: (targetId, scope) => renderDataChart(targetId, scope) },
  'chart-epoch': { title: 'Epoch', unit: 'data.epoch', fn: (targetId, scope) => renderEpochChart(targetId, scope) },
};

function openChartModal(chartId) {
  const spec = CHART_RENDERERS[chartId];
  if (!spec) return;
  const scope = chartId.endsWith('-overview') ? 'overview' : 'charts';
  state.modalChartId = chartId;
  state.modalRenderer = () => spec.fn('chart-modal-canvas', scope);
  const title = $('#chart-modal-title');
  const unit = $('#chart-modal-unit');
  if (title) title.textContent = spec.title;
  if (unit) unit.textContent = spec.unit;
  const backdrop = $('#chart-modal-backdrop');
  if (backdrop) backdrop.hidden = false;
  requestAnimationFrame(() => {
    if (state.modalRenderer) state.modalRenderer();
    const chart = state.charts.get('chart-modal-canvas');
    if (chart && !chart.isDisposed()) chart.resize();
  });
}

function closeChartModal() {
  const backdrop = $('#chart-modal-backdrop');
  if (backdrop) backdrop.hidden = true;
  const chart = state.charts.get('chart-modal-canvas');
  if (chart) {
    try { chart.dispose(); } catch (_) { /* ignore */ }
    state.charts.delete('chart-modal-canvas');
    state.chartMeta.delete('chart-modal-canvas');
    state.chartDataIndex.delete('chart-modal-canvas');
  }
  state.modalChartId = null;
  state.modalRenderer = null;
}

/* --------------------------------------------------------------------------------------------------------- */
/* Runs list & sidebar                                                                                        */
/* --------------------------------------------------------------------------------------------------------- */
async function loadHealth() {
  try {
    state.health = await api('/api/health');
    const h = state.health;
    $('#health').innerHTML = `<span class="dot" style="width:7px;height:7px;border-radius:50%;background:${h.last_error ? 'var(--critical)' : 'var(--good)'};display:inline-block"></span><span>${h.runs} runs · scan ${(h.last_scan_duration_ms || 0).toFixed(0)}ms${h.last_error ? ` · <span class="bad">${esc(h.last_error)}</span>` : ' · ok'}</span>`;
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
    if (!state.token && !location.hash.startsWith('#/compare')) renderRunsPage();
  } catch (error) {
    $('#run-list').innerHTML = `<div class="bad small" style="padding:12px">${esc(error.message)}</div>`;
  }
}

function matchesFilter(run, filter) {
  if (filter === 'running') return run.state === 'running' || run.state === 'initializing';
  if (filter === 'issues') return run.state === 'failed' || run.state === 'attention' || (run.alert_total || 0) > 0;
  if (filter === 'completed') return run.state === 'completed' || run.state === 'stopped';
  return true;
}

function renderSidebar() {
  const active = state.token;
  const filtered = state.runs.filter((run) => matchesFilter(run, state.sidebarFilter));
  const countEl = $('#sidebar-count');
  if (countEl) countEl.textContent = filtered.length === state.runs.length ? `${state.runs.length}` : `${filtered.length}/${state.runs.length}`;
  $$('#sidebar-filters .filter-pill').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.filter === state.sidebarFilter);
  });
  const rows = filtered.map((run) => {
    const cls = run.token === active ? 'active' : '';
    const step = run.step != null ? `step ${int(run.step)}` : 'no steps';
    const loss = run.last && run.last.loss != null ? `loss ${num(run.last.loss, 4)}` : '';
    const alert = run.alert_total ? `<span class="bad">${run.alert_total} alert</span>` : '';
    const progress = run.progress != null ? Math.max(0, Math.min(1, run.progress)) : null;
    const bar = progress != null ? `<div class="ri-progress"><span style="width:${(progress * 100).toFixed(1)}%"></span></div>` : '';
    return `<a class="run-item ${cls}" href="#/run/${run.token}/overview" data-token="${run.token}">
      <div class="ri-top">${statusPill(run.state)}<span class="ri-name" title="${esc(run.display_name)}">${esc(run.display_name)}</span></div>
      <div class="ri-meta"><span>${esc(step)}</span>${loss ? `<span>${esc(loss)}</span>` : ''}${alert ? `<span>${alert}</span>` : ''}<span style="margin-left:auto">${esc(ago(run.age_s))}</span></div>
      ${bar}
    </a>`;
  }).join('');
  $('#run-list').innerHTML = rows || '<div class="empty small">没有匹配的 run<br><span class="muted">尝试调整过滤或 <code>--roots</code></span></div>';
}

function renderRunsPage() {
  const p = palette();
  const runningCount = state.runs.filter((r) => r.state === 'running' || r.state === 'initializing').length;
  const completedCount = state.runs.filter((r) => r.state === 'completed' || r.state === 'stopped').length;
  const issueCount = state.runs.filter((r) => r.state === 'failed' || r.state === 'attention' || (r.alert_total || 0) > 0).length;
  const visibleRuns = state.runs.filter((r) => matchesFilter(r, state.runsFilter));

  const fleet = `<div class="grid k4 fleet-kpis">
    <div class="kpi primary-kpi" style="--kpi-accent:var(--accent)"><div class="label">Total runs</div><div class="value">${int(state.total)}</div><div class="hint">已索引运行目录</div></div>
    <div class="kpi primary-kpi" style="--kpi-accent:var(--series-1)"><div class="label">Running</div><div class="value">${int(runningCount)}</div><div class="hint">活跃训练进程</div></div>
    <div class="kpi primary-kpi" style="--kpi-accent:var(--good)"><div class="label">Completed / Stopped</div><div class="value">${int(completedCount)}</div><div class="hint">已收尾归档</div></div>
    <div class="kpi primary-kpi" style="--kpi-accent:${issueCount ? 'var(--critical)' : 'var(--good)'}"><div class="label">Alerts / Failed</div><div class="value">${int(issueCount)}</div><div class="hint">${issueCount ? '需要查看诊断' : '全部健康'}</div></div>
  </div>`;

  const rows = visibleRuns.map((run) => {
    const last = run.last || {};
    const hbm = last.hbm_fraction != null ? `${pct(last.hbm_fraction, 0)}` : '—';
    const progress = run.progress != null ? Math.max(0, Math.min(1, run.progress)) : 0;
    const alerts = run.alert_total ? `<span class="chip bad">${run.alert_total}</span>` : '<span class="muted">—</span>';
    return `<tr class="clickable" data-href="#/run/${run.token}/overview">
      <td data-no-nav="1"><input class="cmp-select" type="checkbox" data-token="${run.token}" ${state.compareTokens.has(run.token) ? 'checked' : ''} title="加入对比"></td>
      <td><a href="#/run/${run.token}/overview"><strong>${esc(run.display_name)}</strong></a><div class="muted small ellipsis" style="max-width:340px">${esc(run.dir)}</div></td>
      <td>${statusPill(run.state)}${run.stale ? ' <span class="chip warn">stale</span>' : ''}</td>
      <td class="num">${int(run.step)}${run.max_steps ? `<span class="muted"> / ${int(run.max_steps)}</span>` : ''}<div class="progress" style="margin-top:5px;height:4px"><span style="width:${(progress * 100).toFixed(1)}%"></span></div></td>
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

  const filterBar = `<div class="segmented" id="runs-filter-seg">
    <button class="seg-btn ${state.runsFilter === 'all' ? 'active' : ''}" data-rf="all">全部 (${state.runs.length})</button>
    <button class="seg-btn ${state.runsFilter === 'running' ? 'active' : ''}" data-rf="running">运行中 (${runningCount})</button>
    <button class="seg-btn ${state.runsFilter === 'completed' ? 'active' : ''}" data-rf="completed">已完成/停止 (${completedCount})</button>
    <button class="seg-btn ${state.runsFilter === 'issues' ? 'active' : ''}" data-rf="issues">需关注/失败 (${issueCount})</button>
  </div>`;

  const table = state.runs.length ? `<div class="card"><div class="card-head"><div class="card-head-left"><h2>Runs</h2>${filterBar}</div><div class="hint-inline">点击任意行进入详情 · 勾选 ≥2 个可对比</div></div>
    <div class="table-scroll"><table class="table"><thead><tr>
      <th></th><th>run</th><th>状态</th><th class="num">step</th><th class="num">loss</th><th class="num">step time</th>
      <th class="num">samples/s</th><th class="num">MFU</th><th class="num">HBM</th><th class="num">loss 趋势</th><th class="num">alerts</th><th class="num">更新</th>
    </tr></thead><tbody>${rows || '<tr><td colspan="12" class="muted">当前过滤条件下没有 run</td></tr>'}</tbody></table></div></div>` :
    `<div class="card empty"><h2>还没有发现 run</h2><p>用 <code>--roots</code> 指向包含 <code>metrics/run.json</code> 或 <code>metrics/aggregated.jsonl</code> 的目录。例如：</p><pre class="mono">python -m examples.worldmodel.cosmos3_monitor serve --roots /mnt/disk0/jiangzs/nzf/bench_online/runs</pre></div>`;

  $('#crumbs').innerHTML = '<strong>Runs</strong>';
  $('#view').innerHTML = `<div class="page-head"><div><h1>训练运行</h1><div class="subtle">只读索引 · 数据来自各 run 目录 JSONL 遥测</div></div><div class="spacer"></div><div class="page-actions"><span class="chip">${state.total} runs</span><button id="compare-selected" class="primary" ${state.compareTokens.size < 2 ? 'disabled' : ''}>对比选中 (${state.compareTokens.size})</button><button id="refresh-runs" class="ghost">刷新</button></div></div>${fleet}${table}`;

  const button = $('#refresh-runs');
  if (button) button.onclick = () => { loadRuns(); loadHealth(); };
  $$('#runs-filter-seg [data-rf]').forEach((btn) => {
    btn.onclick = () => { state.runsFilter = btn.dataset.rf; persistGlobalPrefs(); renderRunsPage(); };
  });
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
  $$('#view tr[data-href]').forEach((tr) => {
    tr.onclick = (event) => {
      if (event.target.closest('input, a, button, [data-no-nav]')) return;
      location.hash = tr.dataset.href;
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
    state.control = null;
    state.events = [];
    state.alerts = null;
    state.system = null;
    state.ranks = null;
    state.throughput = null;
    state.records = null;
    state.loaded = new Set();
    state.focusStep = null;
    state.lastRenderedTab = null;
    closeChartModal();
    disposeCharts();
    closeStream();
    loadRunPrefs(token);
  }
  state.token = token;
  state.tab = tab || 'overview';
  history.replaceState(null, '', `#/run/${token}/${state.tab}`);
  renderSidebar();
  await loadRunData({ manifest: true });
  renderRunView();
  openStream(token);
  if (!state.detailTimer) state.detailTimer = setInterval(refreshDetailQuiet, 5000);
}

// What each tab draws, beyond the run snapshot, alerts and control view that every tab shares.  The rest is
// fetched when its tab is opened, not on every poll: /system was 1.3 MB (process tables plus 500-sample
// device series) and /throughput?source=ranks another megabyte, and neither is on screen while the overview
// is.  The old load-everything path made the first paint wait for the slowest of ten requests.
const TAB_NEEDS = {
  overview: ['series', 'events', 'checkpoints'],
  charts: ['series', 'throughput'],
  checkpoints: ['checkpoints'],
  events: ['events'],
  system: ['system'],
  topology: ['ranks', 'system'],
  records: [],
  config: [],
  logs: [],
};

async function loadRunData({ manifest = false, tab = state.tab } = {}) {
  const token = state.token;
  if (!token) return;
  const needs = new Set(TAB_NEEDS[tab] || []);
  const jobs = [
    api(`/api/runs/${token}${manifest || tab === 'config' ? '?include=manifest' : ''}`).then((detail) => {
      if (state.token !== token || !detail) return;
      state.run = detail;
      if (detail.manifest) state.manifest = detail.manifest;
    }),
    api(`/api/runs/${token}/control`).then((view) => { if (state.token === token && view) state.control = view; }).catch(() => null),
    api(`/api/runs/${token}/alerts`).then((view) => { if (state.token === token && view) state.alerts = view; }).catch(() => null),
  ];
  if (needs.has('series')) {
    jobs.push(api(`/api/runs/${token}/series?keys=${encodeURIComponent(SERIES_KEYS)}&max_points=5000&x=${state.xMode}`)
      .then((view) => { if (state.token === token && view) state.series = view.series; }).catch(() => null));
  }
  if (needs.has('events')) {
    jobs.push(api(`/api/runs/${token}/events?limit=400`)
      .then((view) => { if (state.token === token) state.events = (view && view.events) || []; }).catch(() => null));
  }
  if (needs.has('checkpoints')) {
    jobs.push(api(`/api/runs/${token}/checkpoints`)
      .then((view) => { if (state.token === token && view) state.checkpoints = view; }).catch(() => null));
  }
  if (needs.has('throughput')) {
    jobs.push(api(`/api/runs/${token}/throughput?source=nodes&x=${state.xMode}&max_points=5000`)
      .then((view) => { if (state.token === token) state.throughput = { ...(state.throughput || {}), nodes: (view && view.series) || {} }; }).catch(() => null));
    // 600 columns is already more than the rank heatmap can show; 2000 was a megabyte per poll.
    jobs.push(api(`/api/runs/${token}/throughput?source=ranks&x=${state.xMode}&max_points=600`)
      .then((view) => { if (state.token === token) state.throughput = { ...(state.throughput || {}), ranks: (view && view.series) || {} }; }).catch(() => null));
  }
  if (needs.has('system')) {
    // The process table and the device series are only drawn on the System tab; the topology view needs
    // just the latest device values, which the compact form carries.
    const query = tab === 'system' ? '?processes=1&points=180' : '?points=0';
    jobs.push(api(`/api/runs/${token}/system${query}`)
      .then((view) => { if (state.token === token && view) state.system = view; }).catch(() => null));
  }
  if (needs.has('ranks')) {
    jobs.push(api(`/api/runs/${token}/ranks`)
      .then((view) => { if (state.token === token && view) state.ranks = view; }).catch(() => null));
  }
  await Promise.all(jobs);
  if (state.token !== token) return;
  for (const need of needs) state.loaded.add(need);
}

async function refreshDetailQuiet() {
  if (!state.token || document.hidden || state.paused) return;
  try {
    await loadRunData();
    updateHeaderMeta();
    renderKpis();
    updateTabBadges();
    if (state.tab === 'charts') {
      refreshCharts();
    } else if (state.tab === 'overview') {
      refreshOverviewInPlace();
    } else if (state.tab === 'checkpoints') {
      refreshCheckpointsInPlace();
    } else if (state.tab === 'system') {
      refreshSystemInPlace();
    } else if (state.tab === 'topology') {
      refreshTopologyInPlace();
    } else if (state.tab === 'events') {
      refreshEventsInPlace();
    } else if (state.tab === 'records') {
      refreshRecordsInPlace();
    }
  } catch (_) { /* keep the last good view; the stale badge explains it */ }
}

function updateTabBadges() {
  const alertCount = state.alerts && state.alerts.counts ? Object.values(state.alerts.counts).reduce((a, b) => a + b, 0) : 0;
  const evBadge = $('#tab-badge-events');
  if (evBadge) {
    evBadge.textContent = alertCount ? String(alertCount) : '';
    evBadge.hidden = !alertCount;
  }
  const ckptCount = state.run && state.run.checkpoints ? state.run.checkpoints.count : 0;
  const ckBadge = $('#tab-badge-checkpoints');
  if (ckBadge) {
    ckBadge.textContent = ckptCount ? String(ckptCount) : '';
    ckBadge.hidden = !ckptCount;
  }
}

function updateHeaderMeta() {
  const run = state.run;
  if (!run) return;
  const status = $('#run-status');
  if (status) status.innerHTML = statusPill(run.state);
  const age = $('#run-age');
  if (age) age.textContent = ago(run.age_s);
  const stale = $('#run-stale');
  if (stale) stale.hidden = !run.stale;
  const progress = run.progress != null ? Math.max(0, Math.min(1, run.progress)) : 0;
  const bar = $('#run-progress');
  if (bar) bar.style.width = `${(progress * 100).toFixed(1)}%`;
  const step = $('#run-step');
  if (step) step.textContent = `${int(run.step)}${run.max_steps ? ` / ${int(run.max_steps)}` : ''}`;
  const label = $('#run-progress-label');
  if (label) label.textContent = run.progress != null ? `${(progress * 100).toFixed(1)}%` : '';
  const etaLine = $('#run-eta');
  if (etaLine) {
    const eta = run.eta_s != null ? duration(run.eta_s) : '—';
    const elapsedVal = run.elapsed_s != null ? duration(run.elapsed_s) : '—';
    const total = run.total_s != null ? duration(run.total_s) : '—';
    const finish = run.completion_at ? clock(run.completion_at) : '—';
    etaLine.innerHTML = `<span>已用时 <strong>${esc(elapsedVal)}</strong></span><span>·</span><span>ETA <strong>${esc(eta)}</strong>${run.eta_source ? ` <span class="muted">(${esc(run.eta_source)})</span>` : ''}</span><span>·</span><span>预计总长 <strong>${esc(total)}</strong></span><span>·</span><span>预计完成 <strong>${esc(finish)}</strong></span>`;
  }
  const controlLine = $('#run-control');
  if (controlLine) {
    const note = controlNote();
    controlLine.textContent = note;
  }
  // The buttons depend on the loop's answer, which arrives through the same 5 s refresh as the KPIs;
  // rebuild them only when the available actions actually changed.
  const actions = $('#run-control-actions');
  if (actions) {
    const markup = controlButtons();
    if (actions.innerHTML !== markup) {
      actions.innerHTML = markup;
      wireControlButtons();
    }
  }
}

function runCrumbs() {
  const run = state.run || {};
  const tabLabels = {
    overview: '概览', charts: '曲线', records: '记录', config: '配置',
    topology: '拓扑', checkpoints: 'Checkpoints', events: '事件与告警', system: '系统', logs: '日志',
  };
  $('#crumbs').innerHTML = `<a href="#/runs">Runs</a> <span>/</span> <strong>${esc(run.display_name || run.run_id || state.token)}</strong> <span>/</span> <span>${esc(tabLabels[state.tab] || state.tab)}</span>`;
}

/* --------------------------------------------------------------------------------------------------------- */
/* Stop / resume                                                                                              */
/* --------------------------------------------------------------------------------------------------------- */
const CONTROL_REASONS = {
  control_disabled: '本服务用 --no-control 关闭了控制',
  read_only: 'run 目录对本服务不可写',
  mirrored: '这是远端主机的 SSH 镜像，控制只能在训练主机上使用',
  no_channel: '训练进程未启用 graceful stop（需要新版 train_sft.py / train.py）',
  not_responding: '训练进程超过一分钟没有更新控制状态',
  not_running: 'run 当前没有在运行',
  already_stopping: '停止请求已在处理中',
  no_checkpoint: '没有找到已发布的 checkpoint 代',
  no_launcher: '本服务未配置 --launch-template，无法直接拉起训练',
  launch_running: '续训进程正在启动',
  still_running: '请先停止当前 run 再续训',
  bad_checkpoint: '该 checkpoint 不属于这个 run',
};

function controlReason(code) {
  return CONTROL_REASONS[code] || code || '';
}

// The note under the progress bar: what was asked for and what the loop has confirmed.
function controlNote() {
  const control = state.control;
  if (!control) return '';
  const request = control.request;
  const status = control.status || {};
  const acknowledged = request && status.request && status.request.request_id === request.request_id;
  if (acknowledged) {
    if (request.action === 'stop') {
      const at = status.stop_at_step != null ? status.stop_at_step : status.step;
      return `停止请求已确认：第 ${int(at)} 步结束后停止，并写一个可续训 generation`;
    }
    return '续训请求已确认';
  }
  if (request && request.action === 'stop') return '停止请求已发送，等待训练进程确认…';
  const launch = control.launch;
  if (launch) {
    if (launch.state === 'running') return `续训进程已启动（pid ${launch.pid}），等待 run.json 与心跳…`;
    if (launch.state === 'failed') return `续训进程已退出（exit ${launch.exit_code}），日志：${launch.log_path}`;
  }
  if (control.status && control.status.phase === 'stopping') return '正在收尾：写最后一个 generation 并退出';
  return '';
}

function controlButtons() {
  const control = state.control;
  if (!control || !control.can) return '';
  const buttons = [];
  if (control.can.cancel) buttons.push('<button id="ctl-cancel" class="ghost">撤销请求</button>');
  if (control.can.stop) buttons.push('<button id="ctl-stop" class="ghost danger">停止（本 step 后）</button>');
  if (control.can.resume) buttons.push('<button id="ctl-resume" class="primary">续训</button>');
  return buttons.join('');
}

function wireControlButtons() {
  const actions = { 'ctl-stop': 'stop', 'ctl-resume': 'resume', 'ctl-cancel': 'cancel' };
  for (const [id, action] of Object.entries(actions)) {
    const button = document.getElementById(id);
    if (button) button.onclick = () => controlAction(action);
  }
}

async function controlAction(action) {
  const question = {
    stop: '在当前 step 结束后停止训练，并写一个可续训的 checkpoint？',
    resume: '从最新 checkpoint 启动续训？',
    cancel: '撤销尚未被训练进程看到的请求？',
  }[action];
  if (question && !window.confirm(question)) return;
  try {
    const data = action === 'cancel'
      ? await apiSend(`/api/runs/${state.token}/control`, 'DELETE')
      : await apiSend(`/api/runs/${state.token}/control`, 'POST', { action });
    state.control = data;
    toast({ stop: '停止请求已发送', resume: '续训已启动', cancel: '请求已撤销' }[action] || '已提交');
    await loadRunData();
    renderRunView();
  } catch (error) {
    toast(error.message, 'bad');
  }
}

function renderRunHeader() {
  const run = state.run;
  if (!run) return;
  runCrumbs();
  const step = run.step || 0;
  const total = run.max_steps;
  const progress = run.progress != null ? Math.max(0, Math.min(1, run.progress)) : 0;
  const alertCount = state.alerts ? (state.alerts.counts ? Object.values(state.alerts.counts).reduce((a, b) => a + b, 0) : 0) : 0;
  const ckptCount = run.checkpoints ? run.checkpoints.count : 0;
  const alertBanner = alertCount ? `<div class="banner ${state.alerts.counts.critical || state.alerts.counts.error ? 'bad' : 'warn'}" style="margin-bottom:12px"><span class="icon">!</span><div class="body"><strong>${alertCount} 个活跃告警</strong><div class="small">${esc((state.alerts.active || []).slice(0, 3).map((a) => a.message).join(' · '))}</div></div><a class="small" href="#/run/${run.token}/events">查看</a></div>` : '';
  $('#view').innerHTML = `
    <div class="page-head">
      <div style="min-width:0">
        <h1>${esc(run.display_name || run.run_id)}</h1>
        <div class="subtle">${esc(run.dir)} · <span class="mono">${esc(run.token)}</span> · <span class="chip">${esc(run.layout)}</span></div>
      </div>
      <div class="spacer"></div>
      <div class="page-actions"><span id="run-control-actions">${controlButtons()}</span><span id="run-status">${statusPill(run.state)}</span><span id="run-stale" class="chip warn" ${run.stale ? '' : 'hidden'}>数据过期</span><span class="chip" id="run-age">${esc(ago(run.age_s))}</span></div>
    </div>
    ${alertBanner}
    <div class="card hero-card">
      <div class="bar-row"><div class="progress"><span id="run-progress" style="width:${(progress * 100).toFixed(1)}%"></span></div><span class="step-readout" id="run-step">${int(step)}${total ? ` / ${int(total)}` : ''}</span><span class="chip accent" id="run-progress-label">${progress ? `${(progress * 100).toFixed(1)}%` : '—'}</span></div>
      <div class="eta-strip" id="run-eta"></div>
      <div class="control-note-bar" id="run-control"></div>
      <div class="grid k4" style="margin-top:12px" id="kpis"></div>
    </div>
    <div class="tabs" id="tabs">
      ${tabLink('overview', '概览')}${tabLink('charts', '曲线')}${tabLink('records', '记录')}${tabLink('config', '配置')}${tabLink('topology', '拓扑')}${tabLink('checkpoints', 'Checkpoints', ckptCount ? `<span id="tab-badge-checkpoints" class="tab-badge">${ckptCount}</span>` : '<span id="tab-badge-checkpoints" class="tab-badge" hidden></span>')}${tabLink('events', '事件与告警', alertCount ? `<span id="tab-badge-events" class="tab-badge bad">${alertCount}</span>` : '<span id="tab-badge-events" class="tab-badge bad" hidden></span>')}${tabLink('system', '系统')}${tabLink('logs', '日志')}
    </div>
    <div id="tab-body"></div>`;
  renderKpis();
  // renderRunHeader rebuilds the ETA node empty; fill it immediately.  Waiting for the 5 s timer left the
  // line blank on every tab switch, while the SSE frame only filled it when a new frame happened to arrive.
  updateHeaderMeta();
  wireControlButtons();
  $$('#tabs .tab').forEach((el) => {
    el.onclick = () => {
      const token = state.token;
      const target = el.dataset.tab;
      state.tab = target;
      history.replaceState(null, '', `#/run/${token}/${target}`);
      const missing = (TAB_NEEDS[target] || []).filter((need) => !state.loaded.has(need));
      renderRunView();
      if (!missing.length) return;
      // This tab's data was never fetched (the poll only refreshes the active tab).  Paint the tab at once
      // so the click feels immediate, then paint again when the data lands.
      loadRunData({ tab: target }).then(() => {
        if (state.token === token && state.tab === target) renderRunView();
      }).catch(() => { /* the stale badge explains a failed load */ });
    };
  });
}

function tabLink(tab, label, badge = '') {
  return `<button class="tab ${state.tab === tab ? 'active' : ''}" data-tab="${tab}">${label}${badge ? ` ${badge}` : ''}</button>`;
}

function renderKpis() {
  const el = $('#kpis');
  if (!el || !state.run) return;
  const last = state.run.last || {};
  const args = (state.manifest && state.manifest.args) || {};
  const micro = args.samples_per_rank != null ? args.samples_per_rank : '—';
  const accum = args.grad_accum != null ? args.grad_accum : '—';
  const step = state.run.step || 0;
  const cov = state.run.coverage || {};
  const kpis = [
    ['Iteration', `${int(step)}${state.run.max_steps ? `<small> / ${int(state.run.max_steps)}</small>` : ''}`, `<a href="#" class="step-link" data-step="${step}">查看 step ${int(step)} 阶段详情 →</a>`, 'var(--series-1)'],
    ['Training objective', num(last.loss, 4), last.vision != null ? `upstream mean: vision ${num(last.vision, 3)} · action ${num(last.action, 3)}` : '优化器实际最小化的标量', 'var(--series-2)'],
    ['Throughput', `${num(last.samples_per_s, 1)}<small>samples/s</small>`, `step ${num(last.step_s, 1)}s · skew ${num(last.skew, 2)}`, 'var(--series-3)'],
    ['MFU', last.mfu != null ? `${num(last.mfu * 100, 1)}<small>%</small>` : '—', '计算利用率 (cube peak)', 'var(--series-4)'],
    ['Step time', `${num(last.step_s, 2)}<small>s</small>`, last.skew != null ? `rank skew ${num(last.skew, 2)}×` : '端到端耗时', null],
    ['HBM', last.device_used_gib != null ? `${num(last.device_used_gib, 1)}<small>/${num(last.device_total_gib, 1)} GiB</small>` : '—', last.hbm_fraction != null ? `使用率 ${pct(last.hbm_fraction, 0)}` : '设备显存', null],
    ['LR / Grad', last.lr != null ? Number(last.lr).toExponential(2) : '—', last.grad_norm != null ? `grad norm ${num(last.grad_norm, 2)}` : '优化器状态', null],
    ['Batch', `${int(last.samples)}<small>samples</small>`, `micro ${esc(micro)} × accum ${esc(accum)}`, null],
    ['Epoch', int(state.run.epoch), `iteration ${int(step)}`, null],
    ['Ranks', `${int(state.run.live_ranks)}<small>/${int(state.run.world)} live</small>`, `${int(cov.nodes_live ?? 0)}/${int(cov.nodes_expected ?? 0)} nodes · ${int(state.run.terminal_ranks)} terminal`, null],
    ['Checkpoints', `${int(state.run.checkpoints && state.run.checkpoints.count)}`, state.run.checkpoints && state.run.checkpoints.last ? `最新 step ${int(state.run.checkpoints.last.step)}` : '暂无记录', null],
    ['Elapsed / ETA', state.run.eta_s != null ? duration(state.run.eta_s) : '—', `elapsed ${duration(state.run.elapsed_s)} · total ${duration(state.run.total_s)}`, null],
  ];
  el.innerHTML = kpis.map(([label, value, hint, accent]) => `
    <div class="kpi ${accent ? 'primary-kpi' : ''}" ${accent ? `style="--kpi-accent:${accent}"` : ''}>
      <div class="label">${esc(label)}</div>
      <div class="value">${value}</div>
      <div class="hint">${hint.startsWith('<a ') ? hint : esc(hint)}</div>
    </div>`).join('');
  document.querySelectorAll('.step-link').forEach((link) => {
    link.onclick = (event) => { event.preventDefault(); openStepDetail(Number(link.dataset.step)); };
  });
}

function captureOpenDetails() {
  const open = new Set();
  document.querySelectorAll('#view details[data-detail-key]').forEach((element) => {
    if (element.open) open.add(element.dataset.detailKey);
  });
  return open;
}

function restoreOpenDetails(open) {
  if (!open || !open.size) return;
  document.querySelectorAll('#view details[data-detail-key]').forEach((element) => {
    if (open.has(element.dataset.detailKey)) element.open = true;
  });
}

function renderRunView() {
  if (!state.run) return;
  const sameTab = state.lastRenderedTab === state.tab;
  const openDetails = captureOpenDetails();
  const scrollY = window.scrollY;
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
  restoreOpenDetails(openDetails);
  if (sameTab) window.scrollTo(0, scrollY);
  state.lastRenderedTab = state.tab;
}

/* --------------------------------------------------------------------------------------------------------- */
/* Overview tab                                                                                               */
/* --------------------------------------------------------------------------------------------------------- */
function overviewEventsMarkup() {
  const recent = (state.events || []).slice(0, 8).map((event) => `
    <div class="list-row">
      <div>${severityChip(event.severity)}</div>
      <div class="grow">
        <div class="ellipsis">${esc(event.message || event.kind)}</div>
        <div class="muted small">step ${esc(event.step)} · ${esc(clock(event.time))}</div>
      </div>
    </div>`).join('');
  return recent || '<div class="muted small">暂无异常事件</div>';
}

function overviewCheckpointsMarkup() {
  const ckpt = state.checkpoints && state.checkpoints.checkpoints ? state.checkpoints.checkpoints.slice(-5).reverse() : [];
  if (!ckpt.length) return '<div class="muted small">还没有 checkpoint 记录</div>';
  return `<table class="table"><thead><tr><th>step</th><th>状态</th><th class="num">大小</th><th class="num">耗时</th></tr></thead><tbody>${ckpt.map((c) => `<tr><td class="num">${int(c.step)}</td><td>${esc(c.status || '—')}</td><td class="num">${bytes(c.bytes)}</td><td class="num">${c.duration_ms != null ? duration(c.duration_ms / 1000) : '—'}</td></tr>`).join('')}</tbody></table>`;
}

function overviewNodesMarkup() {
  const run = state.run;
  const cov = (run && run.coverage) || {};
  const hosts = (cov.hosts || (run && run.hosts) || []).map((h) => {
    const total = (h.ranks || []).length;
    const live = h.live ?? 0;
    const cls = total && live >= total ? 'ok' : (live > 0 ? 'warn' : 'bad');
    const system = h.system_age_s != null ? ` · sys ${ago(h.system_age_s)}` : '';
    const detail = `${h.reporting ?? 0}/${total} reporting${h.stale ? ` · ${h.stale} stale` : ''}${h.missing ? ` · ${h.missing} missing` : ''}`;
    return `<span class="chip ${cls}" title="${esc(detail)}">${esc(h.hostname)} · ${int(live)}/${int(total)} live${esc(system)}</span>`;
  }).join(' ');
  const coverageLine = cov.nodes_expected
    ? `<span class="chip">${int(cov.nodes_live)}/${int(cov.nodes_expected)} nodes live</span><span class="chip">${int(cov.reporting_ranks)}/${int(cov.world)} ranks reporting</span>${cov.nodes_system != null ? `<span class="chip">${int(cov.nodes_system)}/${int(cov.nodes_expected)} system</span>` : ''}`
    : '';
  return `<div class="page-actions" style="margin-bottom:8px">${coverageLine}</div><div class="node-strip" style="margin-top:0">${hosts || '<span class="muted small">manifest 中没有 topology</span>'}</div>`;
}

function chartCardHead(id, title, unit) {
  return `<div class="chart-head">
    <div class="chart-title-group"><h2>${esc(title)}</h2><span class="unit-badge">${esc(unit)}</span></div>
    <div class="chart-head-right">
      <span class="chart-readout" data-readout-for="${id}">—</span>
      <div class="chart-actions">
        <button class="mini ghost" data-chart-expand="${id}" title="放大查看">⤢</button>
        <button class="mini ghost" data-chart-reset-one="${id}" title="重置此图缩放">↺</button>
      </div>
    </div>
  </div>`;
}

function wireChartCardActions(container, scope) {
  container.querySelectorAll('[data-chart-expand]').forEach((btn) => {
    btn.onclick = () => openChartModal(btn.dataset.chartExpand);
  });
  container.querySelectorAll('[data-chart-reset-one]').forEach((btn) => {
    btn.onclick = () => resetZoomForScope(scope, state.syncZoom ? null : btn.dataset.chartResetOne);
  });
}

function renderOverview(body) {
  disposeCharts();
  const run = state.run;
  body.innerHTML = `
    <div class="grid k2" style="grid-template-columns: 1.38fr 1fr">
      <div class="card chart-card">
        ${chartCardHead('chart-loss-overview', 'Loss', 'step')}
        <div class="chart-wrap"><div id="chart-loss-overview" class="chart"></div><div class="chart-empty" data-empty-for="chart-loss-overview" hidden></div></div>
        <div class="chart-note">objective / upstream means / supervised MSE · 滚轮放缩（双击或点 ↺ 重置）；点击数据点打开 step 详情</div>
      </div>
      <div class="card chart-card">
        ${chartCardHead('chart-step-overview', 'Step time', 'seconds')}
        <div class="chart-wrap"><div id="chart-step-overview" class="chart"></div><div class="chart-empty" data-empty-for="chart-step-overview" hidden></div></div>
        <div class="chart-note">按阶段堆叠面积；虚线为端到端 step 耗时</div>
      </div>
    </div>
    <div class="grid k2" style="margin-top:12px">
      <div class="card"><div class="card-head"><h2>最近事件</h2><a class="small" href="#/run/${run.token}/events">全部事件 →</a></div><div id="overview-events">${overviewEventsMarkup()}</div></div>
      <div class="card"><div class="card-head"><h2>最近 Checkpoints</h2><a class="small" href="#/run/${run.token}/checkpoints">全部台账 →</a></div><div id="overview-checkpoints">${overviewCheckpointsMarkup()}</div></div>
    </div>
    <div class="card" style="margin-top:12px"><div class="card-head"><h2>节点在线覆盖</h2><span class="muted small">rank → host 来自 manifest；live 来自每 rank 心跳</span></div><div id="overview-nodes">${overviewNodesMarkup()}</div></div>
    <div class="grid k2" style="margin-top:12px">
      <div id="overview-control">${renderControlCard()}</div>
      ${renderRunInfoCard()}
    </div>`;
  wireChartCardActions(body, 'overview');
  renderLossChart('chart-loss-overview', false, 'overview');
  renderStepChart('chart-step-overview', 'overview');
}

function refreshOverviewInPlace() {
  const ev = $('#overview-events');
  if (ev) ev.innerHTML = overviewEventsMarkup();
  const ck = $('#overview-checkpoints');
  if (ck) ck.innerHTML = overviewCheckpointsMarkup();
  const nd = $('#overview-nodes');
  if (nd) nd.innerHTML = overviewNodesMarkup();
  const ctl = $('#overview-control');
  if (ctl) {
    const open = captureOpenDetails();
    ctl.innerHTML = renderControlCard();
    restoreOpenDetails(open);
  }
  renderLossChart('chart-loss-overview', false, 'overview');
  renderStepChart('chart-step-overview', 'overview');
  if (state.modalRenderer && state.modalChartId && state.modalChartId.endsWith('-overview')) state.modalRenderer();
}

function renderControlCard() {
  const control = state.control;
  if (!control) return '<div class="card"><div class="card-head"><h2>续训与停止</h2></div><div class="muted small">控制通道信息不可用</div></div>';
  const status = control.status || {};
  const channel = control.channel || {};
  const checkpoint = control.checkpoint;
  const reasons = control.reasons || {};
  const channelText = channel.fresh
    ? `<span class="good">已连接</span> · ${esc(status.phase || '—')} · step ${int(status.step)}`
    : channel.present
      ? '<span class="warn">状态已过期（训练进程可能已退出）</span>'
      : '未启用（训练进程未写 status.json）';
  const rows = [
    ['控制通道', channelText],
    ['最新 checkpoint', checkpoint ? `step ${int(checkpoint.step)} · <span class="mono">${esc(checkpoint.generation)}</span>` : '没有已发布的代'],
  ];
  const launch = control.launch;
  if (launch) {
    rows.push(['最近一次续训', `${esc(launch.state)} · pid ${int(launch.pid)} · <span class="mono">${esc(launch.command)}</span>`]);
  }
  const blocked = [];
  for (const [action, label] of [['stop', '停止'], ['resume', '续训']]) {
    if ((control.can || {})[action]) continue;
    const code = reasons[action];
    if (code) blocked.push(`${label}不可用：${esc(controlReason(code))}`);
  }
  const manual = control.manual_command
    ? `<div class="hint-inline" style="margin-top:8px">手动续训：在训练节点上给原命令加 <code class="mono">${esc(control.manual_command)}</code></div>`
    : '';
  const command = control.resume_command
    ? `<details class="config" data-detail-key="control:resume-command" style="margin-top:8px"><summary>续训命令（--launch-template 渲染）</summary><pre class="mono">${esc(control.resume_command)}</pre></details>`
    : '';
  return `<div class="card"><div class="card-head"><h2>续训与停止</h2><span class="muted small">&lt;run&gt;/control · 当前 step 结束后生效</span></div>
    <div class="kv">${rows.map(([key, value]) => `<div class="k">${esc(key)}</div><div class="v">${value}</div>`).join('')}</div>
    ${blocked.length ? `<div class="hint-inline" style="margin-top:8px">${blocked.join(' · ')}</div>` : ''}
    ${manual}${command}</div>`;
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
  return `<div class="card"><div class="card-head"><h2>Run info</h2><span class="muted small">来自 run.json</span></div><div class="kv">${rows.map(([key, value]) => `<div class="k">${esc(key)}</div><div class="v mono">${esc(String(value))}</div>`).join('')}</div></div>`;
}

/* --------------------------------------------------------------------------------------------------------- */
/* Charts tab                                                                                                 */
/* --------------------------------------------------------------------------------------------------------- */
const CHART_CARDS = [
  ['chart-loss', 'Loss', 'loss', 'train', true],
  ['chart-step', 'Step time', 'seconds', 'perf', true],
  ['chart-stage-mfu', 'Stage MFU', '% · forward / backward / VAE', 'perf', false],
  ['chart-throughput', 'Throughput', 'samples/s · total + per node', 'perf', false],
  ['chart-rank-heat', 'Rank throughput', 'samples/s · rank × step', 'perf', false],
  ['chart-mfu', 'MFU', '% of step', 'perf', false],
  ['chart-lr', 'Learning rate', 'lr', 'train', false],
  ['chart-grad', 'Gradient norm', 'norm', 'train', false],
  ['chart-mem', 'Memory', 'GiB', 'resource', false],
  ['chart-skew', 'Rank skew', 'max / mean', 'resource', false],
  ['chart-data', 'Batch / tokens per step', 'log count', 'resource', false],
  ['chart-epoch', 'Epoch', 'data.epoch', 'resource', false],
];

function applyChartCategoryFilter() {
  $$('#charts-grid .chart-card').forEach((card) => {
    const show = state.chartCategory === 'all' || card.dataset.category === state.chartCategory;
    card.classList.toggle('is-hidden', !show);
    if (show) {
      const id = card.dataset.chartId;
      const chart = state.charts.get(id);
      if (chart && !chart.isDisposed()) requestAnimationFrame(() => chart.resize());
    }
  });
  $$('#chart-cat-seg [data-cat]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.cat === state.chartCategory);
  });
}

function renderCharts(body) {
  disposeCharts();
  const rangeOptions = [0, 20, 100, 500, 2000].map((value) => `<option value="${value}" ${state.range === value ? 'selected' : ''}>${value ? `最近 ${value}` : '全部'}</option>`).join('');
  const cardsHtml = CHART_CARDS.map(([id, title, unit, category, tall]) => `
    <div class="card chart-card" data-chart-id="${id}" data-category="${category}">
      ${chartCardHead(id, title, unit)}
      <div class="chart-wrap"><div id="${id}" class="chart${tall ? ' tall' : ''}"></div><div class="chart-empty" data-empty-for="${id}" hidden></div></div>
    </div>`).join('');

  body.innerHTML = `
    <div class="card chart-toolbar">
      <div class="toolbar-row">
        <div class="toolbar-group">
          <div class="segmented" id="chart-cat-seg" title="按指标类别筛选视图">
            <button class="seg-btn ${state.chartCategory === 'all' ? 'active' : ''}" data-cat="all">全部 12</button>
            <button class="seg-btn ${state.chartCategory === 'train' ? 'active' : ''}" data-cat="train">训练收敛</button>
            <button class="seg-btn ${state.chartCategory === 'perf' ? 'active' : ''}" data-cat="perf">算力与吞吐</button>
            <button class="seg-btn ${state.chartCategory === 'resource' ? 'active' : ''}" data-cat="resource">显存与数据</button>
          </div>
          <label class="control-group">X 轴
            <select id="chart-xmode">
              <option value="step" ${state.xMode === 'step' ? 'selected' : ''}>iteration (step)</option>
              <option value="time" ${state.xMode === 'time' ? 'selected' : ''}>elapsed time</option>
            </select>
          </label>
          <label class="control-group">范围 <select id="chart-range">${rangeOptions}</select></label>
        </div>
        <div class="toolbar-group">
          <div class="segmented" title="鼠标拖拽交互模式">
            <button class="seg-btn ${state.zoomMode === 'pan' ? 'active' : ''}" data-zmode="pan">平移拖拽</button>
            <button class="seg-btn ${state.zoomMode === 'box' ? 'active' : ''}" data-zmode="box">框选缩放</button>
          </div>
          <button id="chart-sync-toggle" class="ghost ${state.syncZoom ? 'active' : ''}" title="切换多图联动缩放与十字准星">${state.syncZoom ? '联动缩放：开' : '联动缩放：关'}</button>
          <span id="zoom-pill" class="zoom-pill" hidden></span>
          <button id="chart-reset" class="ghost">重置缩放</button>
        </div>
      </div>
      <div class="toolbar-row">
        <div class="toolbar-group">
          <label class="control-group">EMA 平滑 <input type="range" id="chart-smooth" min="0" max="20" value="${state.smooth}"><span class="value" id="chart-smooth-value">${state.smooth}</span></label>
          <button id="chart-log" class="ghost ${state.chartLog ? 'active' : ''}">${state.chartLog ? '对数 Y：开' : '对数 Y：关'}</button>
          <label class="checkbox"><input type="checkbox" id="chart-annot" ${state.showAnnotations ? 'checked' : ''}> 事件/checkpoint 标注</label>
          <span id="annot-note" class="hint-inline"></span>
        </div>
        <div class="hint-inline">滚轮缩放 · 双击恢复 · 点击数据点看 step 详情</div>
      </div>
    </div>
    <div class="grid charts" id="charts-grid">${cardsHtml}</div>`;

  wireChartCardActions(body, 'charts');
  refreshCharts();
  applyChartCategoryFilter();
  updateZoomPill();

  $$('#chart-cat-seg [data-cat]').forEach((btn) => {
    btn.onclick = () => {
      state.chartCategory = btn.dataset.cat;
      persistGlobalPrefs();
      applyChartCategoryFilter();
    };
  });
  $$('[data-zmode]').forEach((btn) => {
    btn.onclick = () => {
      state.zoomMode = btn.dataset.zmode;
      persistGlobalPrefs();
      $$('[data-zmode]').forEach((b) => b.classList.toggle('active', b.dataset.zmode === state.zoomMode));
      applyZoomModeToCharts();
    };
  });
  const syncToggle = $('#chart-sync-toggle');
  if (syncToggle) syncToggle.onclick = () => {
    state.syncZoom = !state.syncZoom;
    persistGlobalPrefs();
    syncToggle.textContent = state.syncZoom ? '联动缩放：开' : '联动缩放：关';
    syncToggle.classList.toggle('active', state.syncZoom);
    refreshCharts();
    updateZoomPill();
  };
  const xmode = $('#chart-xmode');
  if (xmode) xmode.onchange = async () => {
    state.xMode = xmode.value;
    state.range = 0;
    saveRunPrefs();
    await loadRunData();
    renderRunView();
  };
  const range = $('#chart-range');
  if (range) range.onchange = () => {
    state.range = Number(range.value);
    saveRunPrefs();
    refreshCharts();
  };
  const smooth = $('#chart-smooth');
  if (smooth) smooth.oninput = () => {
    state.smooth = Number(smooth.value);
    const label = $('#chart-smooth-value');
    if (label) label.textContent = String(state.smooth);
    saveRunPrefs();
    refreshCharts();
  };
  const annot = $('#chart-annot');
  if (annot) annot.onchange = () => {
    state.showAnnotations = annot.checked;
    saveRunPrefs();
    refreshCharts();
  };
  const logButton = $('#chart-log');
  if (logButton) logButton.onclick = () => {
    state.chartLog = !state.chartLog;
    saveRunPrefs();
    logButton.textContent = state.chartLog ? '对数 Y：开' : '对数 Y：关';
    logButton.classList.toggle('active', state.chartLog);
    renderLossChart('chart-loss', true, 'charts');
    if (state.modalRenderer && state.modalChartId === 'chart-loss') state.modalRenderer();
  };
  const reset = $('#chart-reset');
  if (reset) reset.onclick = () => resetZoomForScope('charts');
}

function refreshCharts() {
  if (!state.series) return;
  renderLossChart('chart-loss', true, 'charts');
  renderStepChart('chart-step', 'charts');
  renderStageMfuChart('chart-stage-mfu', 'charts');
  renderThroughputChart('chart-throughput', 'charts');
  renderRankHeatmap('chart-rank-heat', 'charts');
  renderMfuChart('chart-mfu', 'charts');
  renderLrChart('chart-lr', 'charts');
  renderGradChart('chart-grad', 'charts');
  renderMemoryChart('chart-mem', 'charts');
  renderSkewChart('chart-skew', 'charts');
  renderDataChart('chart-data', 'charts');
  renderEpochChart('chart-epoch', 'charts');
  updateAnnotationNote();
  updateZoomPill();
  if (state.modalRenderer && state.modalChartId && !state.modalChartId.endsWith('-overview')) state.modalRenderer();
}

/* --------------------------------------------------------------------------------------------------------- */
/* Step-detail Drawer                                                                                         */
/* --------------------------------------------------------------------------------------------------------- */
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

function renderStepThroughput(throughput, world = 0) {
  if (!throughput) return '';
  const nodes = throughput.nodes || [];
  const ranks = throughput.ranks || [];
  const nodeRows = nodes.map((node) => `<tr><td>${esc(node.host)}</td><td class="num">${node.ranks.length}</td><td class="num">${int(node.samples)}</td><td class="num">${num(node.step_s, 3)}</td><td class="num">${num(node.throughput, 2)}</td><td>${node.complete ? '<span class="chip ok">complete</span>' : '<span class="chip warn">partial</span>'}</td></tr>`).join('');
  const rankRows = ranks.map((rank) => `<tr><td class="num">${int(rank.rank)}</td><td>${esc(rank.host)}</td><td class="num">${int(rank.samples)}</td><td class="num">${num(rank.step_s, 3)}</td><td class="num">${num(rank.throughput, 2)}</td></tr>`).join('');
  // The per-rank table can hold fewer ranks than the run had: the monitor keeps a bounded history per rank
  // file, so a step older than that window has no per-rank rows left. Say so instead of implying the run ran
  // on `ranks.length` cards.
  const coverage = world && ranks.length < world ? ` · <span class="warn">${ranks.length}/${int(world)} ranks 已保留</span>` : '';
  return `<div class="card" style="margin-top:14px"><div class="card-head"><h2>Throughput by node</h2><span class="muted small">sum(samples) / slowest rank step${coverage}</span></div>
    <table class="table"><thead><tr><th>node</th><th class="num">ranks</th><th class="num">samples</th><th class="num">step s</th><th class="num">samples/s</th><th>状态</th></tr></thead><tbody>${nodeRows || '<tr><td colspan="6" class="muted">没有 rank 数据</td></tr>'}</tbody></table></div>
    <details class="config" style="margin-top:8px"><summary>Per-rank throughput · ${ranks.length} ranks${world ? ` / ${int(world)}` : ''}</summary><table class="table"><thead><tr><th class="num">rank</th><th>node</th><th class="num">samples</th><th class="num">step s</th><th class="num">samples/s</th></tr></thead><tbody>${rankRows || '<tr><td colspan="5" class="muted">没有 rank 数据</td></tr>'}</tbody></table></details>`;
}

function renderStepDrawer() {
  const d = state.detail;
  if (!d) return;
  const title = document.getElementById('drawer-title');
  const sub = document.getElementById('drawer-sub');
  const body = document.getElementById('drawer-body');
  // How many ranks the monitor still holds for this step.  A step older than the per-rank history window has
  // fewer, and then `batch_samples` is the sum over those ranks, not the global batch: label it rather than
  // letting "Batch 480" look like a 15-card run.
  const worldSize = Number(d.world) || 0;
  const heldRanks = ((d.throughput || {}).ranks || []).length;
  const shortRanks = Boolean(worldSize) && heldRanks < worldSize;
  if (title) title.textContent = `Iteration ${int(d.iteration)}`;
  if (sub) {
    sub.textContent = `Epoch ${d.epoch == null ? '—' : int(d.epoch)} · Batch ${int(d.batch_samples)} samples`
      + (shortRanks ? ` (${heldRanks}/${worldSize} ranks)` : '')
      + ` · micro ${int(d.micro_batch_samples)} × ${int(d.grad_accum)} · world ${int(d.world)}`;
  }
  const p = palette();
  const stageColor = { wait: p[4], VAE: p[3], forward: p[0], backward: p[1], optim: p[2] };
  const stages = d.stages || [];
  const rows = stages.map((stage) => {
    const share = stage.share == null ? null : Math.max(0, Math.min(1, stage.share));
    const color = stageColor[stage.name] || p[0];
    const bar = share == null ? '<span class="muted">—</span>' : `<div class="sharebar"><span style="width:${(share * 100).toFixed(1)}%;background:${color}"></span></div>`;
    return `<tr><td>${esc(stage.name)}</td><td class="num">${stage.seconds == null ? '—' : `${num(stage.seconds, 3)}s`}</td><td class="num">${share == null ? '—' : pct(share, 1)}</td><td>${bar}</td><td class="num">${stage.mfu == null ? '—' : pct(stage.mfu, 2)}</td><td class="num">${stage.flops == null ? '—' : `${num(stage.flops / 1e12, 2)} TFLOP`}</td><td>${esc(stage.source || '—')}</td><td class="stage-note">${esc(stage.note || '')}</td></tr>`;
  }).join('');
  const loss = d.loss || {};
  const memory = d.memory || {};
  if (body) {
    body.innerHTML = `
      <div class="drawer-kpis">
        <div class="kpi"><div class="label">Iteration time</div><div class="value">${num(d.step_seconds, 2)}<small>s</small></div><div class="hint">${num(d.samples_per_second, 1)} samples/s</div></div>
        <div class="kpi"><div class="label">Step MFU</div><div class="value">${d.total_mfu == null ? '—' : `${num(d.total_mfu * 100, 1)}<small>%</small>`}</div><div class="hint">${esc(d.activation_checkpointing)} ckpt</div></div>
        <div class="kpi"><div class="label">Training objective</div><div class="value">${num(loss.objective ?? loss.total, 6)}</div><div class="hint">upstream mean v ${num(loss.vision, 6)} · a ${num(loss.action, 6)} · supervised MSE v ${num(loss.vision_supervised_mse, 6)} · a ${num(loss.action_supervised_mse, 6)}</div></div>
        <div class="kpi"><div class="label">HBM</div><div class="value">${num(memory.device_used_gib, 1)}<small> GiB</small></div><div class="hint">alloc ${num(memory.allocated_gib, 1)} · peak ${num(memory.peak_reserved_gib, 1)}</div></div>
      </div>
      <div class="drawer-actions">
        <button id="drawer-prev" class="ghost">← 上一个 step</button>
        <button id="drawer-next" class="ghost">下一个 step →</button>
        <span class="sep"></span>
        <span class="hint-inline">MFU/FLOPs 来自训练侧记录；backward 含 activation recompute</span>
        ${shortRanks
          ? `<span class="chip warn">batch/MFU 只由 ${heldRanks}/${worldSize} 个 rank 校正（其余超出监控的 per-rank 保留窗口）</span>`
          : (d.aggregate_from_ranks ? '<span class="chip warn">batch/MFU 由镜像的全部 rank 校正</span>' : (d.samples_estimated ? '<span class="chip warn">samples 由单卡 batch × world 估算</span>' : ''))}
      </div>
      <table class="table stage-table"><thead><tr><th>阶段</th><th class="num">时间</th><th class="num">占比</th><th>share</th><th class="num">阶段 MFU</th><th class="num">FLOPs</th><th>source</th><th>说明</th></tr></thead><tbody>${rows}</tbody></table>
      <div class="card" style="margin-top:14px"><div class="card-head"><h2>本 step 关键数据</h2><span class="muted small">${shortRanks ? `阶段时间来自 rank 0 聚合；batch/MFU 只用 ${heldRanks}/${worldSize} 个 rank` : (d.aggregate_from_ranks ? '阶段时间来自 rank 0 聚合；batch/MFU 用全部 rank' : (d.samples_estimated ? 'samples 由配置的单卡 batch × world 估算；阶段时间来自 rank 0 聚合' : (d.complete === false ? 'partial aggregate（rank 0 只聚合到本节点的 rank）' : 'complete aggregate')))}</span></div>
        <div class="kv">
          <div class="k">iteration</div><div class="v mono">${int(d.iteration)}</div>
          <div class="k">epoch</div><div class="v mono">${d.epoch == null ? '—' : int(d.epoch)}</div>
          <div class="k">batch (global samples)</div><div class="v mono">${int(d.batch_samples)}</div>
          <div class="k">micro batch × accum</div><div class="v mono">${int(d.micro_batch_samples)} × ${int(d.grad_accum)}</div>
          <div class="k">effective global batch</div><div class="v mono">${int(d.effective_global_batch)}</div>
          <div class="k">grad norm / lr</div><div class="v mono">${num((d.optim || {}).grad_norm, 3)} / ${num((d.optim || {}).lr, 8)}</div>
          <div class="k">samples / s</div><div class="v mono">${num(d.samples_per_second, 2)}</div>
          <div class="k">step seconds</div><div class="v mono">${num(d.step_seconds, 3)}</div>
        </div>
      </div>
      ${renderStepThroughput(d.throughput, worldSize)}`;
  }
  const prev = document.getElementById('drawer-prev');
  const next = document.getElementById('drawer-next');
  if (prev) prev.onclick = () => openStepDetail(Number(d.iteration) - 1);
  if (next) next.onclick = () => openStepDetail(Number(d.iteration) + 1);
}

/* --------------------------------------------------------------------------------------------------------- */
/* Records, Config, Topology, Checkpoints, Events, System, Logs                                               */
/* --------------------------------------------------------------------------------------------------------- */
function recordRowsMarkup() {
  const rows = (state.records && state.records.rows) || [];
  return rows.map((record) => {
    const data = record.data || {};
    const loss = record.loss || {};
    const perf = record.perf || {};
    const mem = record.mem || {};
    const optim = record.optim || {};
    const stepMs = perf.step_ms;
    const samples = data.samples != null ? data.samples : data.partial_samples;
    const throughput = stepMs && samples ? samples / (stepMs / 1000) : null;
    return `<tr class="clickable" data-step="${int(record.step)}"><td class="num">${int(record.step)}</td><td class="num">${int(data.epoch)}</td><td class="num">${int(samples)}</td><td class="num">${clock(record.time)}</td>
      <td class="num">${num(loss.objective ?? loss.total, 6)}</td><td class="num">${num(loss.vision, 6)}</td><td class="num">${num(loss.action, 6)}</td><td class="num">${num(loss.vision_supervised_mse, 6)}</td><td class="num">${num(loss.action_supervised_mse, 6)}</td>
      <td class="num">${num(optim.lr, 8)}</td><td class="num">${num(optim.grad_norm, 3)}</td>
      <td class="num">${num(stepMs != null ? stepMs / 1000 : null, 1)}</td><td class="num">${num(throughput, 1)}</td>
      <td class="num">${num(mem.device_used_gib, 1)}</td><td class="num">${num(perf.skew, 2)}</td>
      <td>${record.complete === false ? `<span class="chip warn">${record.samples_estimated ? 'estimated' : 'partial'}</span>` : '<span class="chip ok">complete</span>'}</td></tr>`;
  }).join('');
}

function renderRecords(body) {
  const total = state.records ? state.records.total : 0;
  const table = recordRowsMarkup();
  body.innerHTML = `<div class="card"><div class="card-head"><h2>记录 <span class="muted small" id="rec-total">${total} steps</span></h2><div class="page-actions"><button id="rec-prev" class="ghost">上一页</button><button id="rec-next" class="ghost">下一页</button><a class="small" href="/api/export/${state.token}.csv?keys=${encodeURIComponent(SERIES_KEYS)}">导出 CSV</a></div></div>
    <div class="table-scroll"><table class="table" id="records-table"><thead><tr><th class="num">iter</th><th class="num">epoch</th><th class="num">batch</th><th class="num">time</th><th class="num">training objective</th><th class="num">vision upstream mean</th><th class="num">action upstream mean</th><th class="num">vision supervised MSE</th><th class="num">action supervised MSE</th><th class="num">lr</th><th class="num">grad</th><th class="num">step s</th><th class="num">samples/s</th><th class="num">HBM GiB</th><th class="num">skew</th><th>complete</th></tr></thead><tbody id="records-tbody">${table || '<tr><td colspan="16" class="muted">没有记录</td></tr>'}</tbody></table></div></div>`;
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

async function refreshRecordsInPlace() {
  if (!state.records || (state.records.offset || 0) !== 0) return;
  const limit = state.records.limit || 100;
  try {
    state.records = await api(`/api/runs/${state.token}/records?offset=0&limit=${limit}&order=desc`);
    const tbody = $('#records-tbody');
    if (tbody) {
      tbody.innerHTML = recordRowsMarkup() || '<tr><td colspan="14" class="muted">没有记录</td></tr>';
      $$('#records-table tr[data-step]').forEach((row) => { row.onclick = () => openStepDetail(Number(row.dataset.step)); });
    }
    const totalEl = $('#rec-total');
    if (totalEl) totalEl.textContent = `${state.records.total} steps`;
  } catch (_) { /* keep current table */ }
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

function renderConfigTree(value, name = '', path = '') {
  const currentPath = path || name || 'manifest';
  if (value == null) return `<span class="muted">null</span>`;
  if (Array.isArray(value)) {
    if (!value.length) return '<span class="muted">[]</span>';
    if (value.length > 12 || value.some((item) => typeof item === 'object')) {
      return `<details class="config" data-detail-key="config:${esc(currentPath)}"><summary>${esc(name || 'array')} · ${value.length} items</summary><pre class="mono">${esc(JSON.stringify(value, null, 2))}</pre></details>`;
    }
    return `<div class="kv"><div class="k">${esc(name)}</div><div class="v mono">${esc(JSON.stringify(value))}</div></div>`;
  }
  if (typeof value === 'object') {
    const keys = Object.keys(value).sort();
    if (!keys.length) return '<span class="muted">{}</span>';
    return `<details class="config" data-detail-key="config:${esc(currentPath)}" ${name ? '' : 'open'}><summary>${esc(name || 'manifest')} · ${keys.length}</summary>${keys.map((key) => renderConfigTree(value[key], key, `${currentPath}.${key}`)).join('')}</details>`;
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
  return rows ? `<div class="table-scroll" style="max-height:420px;margin-top:10px"><table class="table"><thead><tr><th>key</th><th>this run</th><th>other run</th></tr></thead><tbody>${rows}</tbody></table></div>` : '<div class="muted small" style="margin-top:8px">两份 manifest 没有差异</div>';
}

function topologyContentMarkup() {
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
  const covHosts = {};
  for (const item of ((state.run && state.run.coverage && state.run.coverage.hosts) || [])) covHosts[item.hostname] = item;
  const nodeCards = Object.keys(byHost).sort().map((host) => {
    const system = latest[host];
    const devices = (system && system.devices) || [];
    const coverage = covHosts[host];
    const cards = byHost[host].slice().sort((a, b) => (a.local_rank || 0) - (b.local_rank || 0)).map((row) => {
      const rank = row.rank;
      const device = devices.find((d) => Number(d.index) === Number(row.device_index ?? row.local_rank));
      const used = device && device.hbm_used_mib != null && device.hbm_total_mib ? device.hbm_used_mib / device.hbm_total_mib : null;
      const item = rankState[rank];
      const heartbeat = item && item.heartbeat;
      const status = item ? item.status : 'missing';
      const phase = heartbeat ? heartbeat.phase : '—';
      const stepText = item && item.step != null ? `step ${int(item.step)}` : '';
      const ageText = item && item.age_s != null ? `hb ${ago(item.age_s)}` : '';
      const statusText = status === 'live' ? 'live' : (status === 'stale' ? 'stale' : (status === 'completed' ? 'done' : 'missing'));
      const offline = status === 'live' || status === 'completed' ? '' : ' offline';
      const loadCls = used != null && used >= 0.92 ? ' crit' : (used != null && used >= 0.8 ? ' hot' : '');
      const color = used == null ? 'var(--surface)' : `color-mix(in srgb, var(--accent) ${Math.round(Math.max(8, Math.min(28, used * 30)))}%, var(--surface))`;
      const detail = [stepText, ageText, statusText].filter(Boolean).join(' · ');
      const bar = used != null ? `<div class="hbm-bar"><span style="width:${(Math.min(1, used) * 100).toFixed(0)}%"></span></div>` : '';
      return `<div class="topo-card${offline}${loadCls}" style="background:${color}" title="rank ${rank} · ${esc(host)} · ${device ? `${num(device.hbm_used_mib / 1024, 1)}/${num(device.hbm_total_mib / 1024, 1)} GiB · ${num(device.temperature_c, 0)}°C` : 'no device sample'} · ${esc(detail)}">
        <div class="idx"><span>NPU ${esc(row.device_index ?? row.local_rank ?? '?')}</span><span>${esc(statusText)}</span></div>
        <div class="rank">rank ${esc(rank)}</div>
        <div class="hbm">${device ? `${num(device.hbm_used_mib / 1024, 1)}G · ${num(device.temperature_c, 0)}°C` : '<span class="muted">—</span>'}</div>
        <div class="hbm muted">${esc(phase)}</div>
        ${bar}
      </div>`;
    }).join('');
    const empty = Array.from({ length: Math.max(0, 8 - byHost[host].length) }).map(() => '<div class="topo-card empty"></div>').join('');
    const liveText = coverage ? `${int(coverage.live)}/${int((coverage.ranks || []).length)} live` : `${(byHost[host] || []).length} ranks`;
    const sysText = coverage && coverage.system_age_s != null ? ` · sys ${ago(coverage.system_age_s)}` : '';
    return `<div class="topo-node"><div class="topo-node-head"><span>${esc(host)}${(byHost[host][0] || {}).derived ? ' <span class="chip">derived</span>' : ''}</span><span class="muted small">${liveText}${esc(sysText)}</span></div><div class="topo-cards">${cards}${empty}</div></div>`;
  }).join('');
  const groupRows = rows.length && rows[0].groups ? Object.entries(rows[0].groups).map(([name, ranks]) => `<tr><td>${esc(name)}</td><td class="mono">${esc((ranks || []).join(', '))}</td></tr>`).join('') : '';
  return `<div class="card"><div class="card-head"><h2>节点 × 卡</h2><span class="muted small">颜色与进度条 = HBM 使用率；悬停查看详情</span></div><div class="topology">${nodeCards || '<div class="muted small">manifest 中没有 topology</div>'}</div></div>
    ${groupRows ? `<div class="card"><div class="card-head"><h2>并行组</h2><span class="muted small">来自 manifest.topology[0].groups</span></div><table class="table"><thead><tr><th>group</th><th>ranks</th></tr></thead><tbody>${groupRows}</tbody></table></div>` : ''}`;
}

function renderTopology(body) {
  if (state.ranks == null || state.system == null) {
    body.innerHTML = '<div class="card"><div class="muted small">读取拓扑数据中…</div></div>';
    return;
  }
  body.innerHTML = `<div id="topology-root">${topologyContentMarkup()}</div>`;
}

function refreshTopologyInPlace() {
  const root = $('#topology-root');
  if (!root) {
    const body = $('#tab-body');
    if (body) renderTopology(body);
    return;
  }
  root.innerHTML = topologyContentMarkup();
}

function checkpointsChipsMarkup() {
  const schedule = (state.manifest && (state.manifest.checkpoints || state.manifest.checkpoint_schedule)) || {};
  return `<span class="chip">计划：每 ${esc(schedule.save_every || (schedule.model_export && schedule.model_export.every) || '—')} step</span><span class="chip">${esc(schedule.async_save || (schedule.model_export && schedule.model_export.async) ? 'async' : 'sync')}</span><span class="chip">${esc(state.run.checkpoints ? bytes(state.run.checkpoints.bytes) : '—')} total</span>`;
}

function checkpointsRowsMarkup() {
  const rows = (state.checkpoints && state.checkpoints.checkpoints) || [];
  return rows.slice().reverse().map((row) => {
    const hasStep = row.step != null && Number.isFinite(Number(row.step));
    return `<tr${hasStep ? ` class="clickable" data-step="${Number(row.step)}"` : ''}>
    <td class="num">${int(row.step)}</td><td>${esc(row.kind || '—')}</td><td>${esc(row.status || '—')}</td>
    <td class="num">${row.started_at ? clock(row.started_at) : '—'}</td><td class="num">${row.ended_at ? clock(row.ended_at) : '—'}</td>
    <td class="num">${row.duration_ms != null ? duration(row.duration_ms / 1000) : '—'}</td><td class="num">${bytes(row.bytes)}</td>
    <td class="num">${int(row.files)}</td><td class="mono ellipsis" style="max-width:260px" title="${esc(row.path || '')}">${esc(row.path || '—')}</td><td>${row.async ? 'async' : 'sync'}</td><td>${esc(row.source || 'ledger')}</td>
  </tr>`;
  }).join('');
}

function renderCheckpoints(body) {
  disposeCharts();
  body.innerHTML = `<div class="card" style="margin-bottom:12px"><div class="page-actions" id="ckpt-summary-chips">${checkpointsChipsMarkup()}</div></div>
    <div class="grid k2">
      <div class="card chart-card"><div class="chart-head"><div class="chart-title-group"><h2>大小</h2><span class="unit-badge">bytes</span></div></div><div class="chart-wrap"><div id="chart-ckpt-size" class="chart"></div><div class="chart-empty" data-empty-for="chart-ckpt-size" hidden></div></div></div>
      <div class="card chart-card"><div class="chart-head"><div class="chart-title-group"><h2>耗时</h2><span class="unit-badge">seconds</span></div></div><div class="chart-wrap"><div id="chart-ckpt-duration" class="chart"></div><div class="chart-empty" data-empty-for="chart-ckpt-duration" hidden></div></div></div>
    </div>
    <div class="card" style="margin-top:12px"><div class="card-head"><h2>Checkpoint 台账</h2><span class="muted small">begin/end 来自 checkpoints.jsonl；缺失时回退扫描目录</span></div>
    <div class="table-scroll"><table class="table"><thead><tr><th class="num">step</th><th>kind</th><th>status</th><th class="num">started</th><th class="num">ended</th><th class="num">duration</th><th class="num">size</th><th class="num">files</th><th>path</th><th>mode</th><th>source</th></tr></thead><tbody id="ckpt-tbody">${checkpointsRowsMarkup() || '<tr><td colspan="11" class="muted">还没有 checkpoint 记录</td></tr>'}</tbody></table></div></div>`;
  const tbody = $('#ckpt-tbody');
  if (tbody) {
    tbody.onclick = (event) => {
      const tr = event.target && event.target.closest ? event.target.closest('tr[data-step]') : null;
      if (tr && tr.dataset.step) openStepDetail(Number(tr.dataset.step));
    };
  }
  renderCheckpointCharts();
}

function refreshCheckpointsInPlace() {
  const chips = $('#ckpt-summary-chips');
  if (chips) chips.innerHTML = checkpointsChipsMarkup();
  const tbody = $('#ckpt-tbody');
  if (tbody) tbody.innerHTML = checkpointsRowsMarkup() || '<tr><td colspan="11" class="muted">还没有 checkpoint 记录</td></tr>';
  // Update bars in-place without disposing charts -> zoom stays intact!
  renderCheckpointCharts();
}

function eventsContentMarkup() {
  const events = state.events || [];
  const alerts = state.alerts || { active: [], counts: {} };
  const active = (alerts.active || []).map((alert) => `<div class="banner ${alert.severity === 'critical' || alert.severity === 'error' ? 'bad' : 'warn'}" style="margin-bottom:8px"><span class="icon">!</span><div class="body"><strong>${esc(alert.rule)}</strong> · ${esc(alert.message)}<div class="muted small">first ${clock(alert.first_seen)} · last ${clock(alert.last_seen)} · ×${alert.count}${alert.acked ? ' · acked' : ''}</div></div><button class="ghost" data-ack="${esc(alert.id)}">ack</button></div>`).join('');
  const rows = events.map((event) => `<tr><td class="num">${int(event.step)}</td><td>${severityChip(event.severity)}</td><td>${esc(event.kind)}</td><td class="num">${esc(event.rank ?? '—')}</td><td>${esc(event.message || '')}</td><td class="num">${clock(event.time)}</td></tr>`).join('');
  return { active, rows };
}

function wireAckButtons(container) {
  container.querySelectorAll('[data-ack]').forEach((button) => {
    button.onclick = async () => {
      try {
        await fetch(`/api/runs/${state.token}/alerts/${encodeURIComponent(button.dataset.ack)}/ack`, { method: 'POST' });
        state.alerts = await api(`/api/runs/${state.token}/alerts`);
        refreshEventsInPlace();
      } catch (error) { toast(error.message, 'bad'); }
    };
  });
}

function renderEvents(body) {
  const { active, rows } = eventsContentMarkup();
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
  body.innerHTML = `<div class="card"><div class="card-head"><h2>活跃告警</h2><span class="muted small">仅看板内；AICore% 不参与告警</span></div><div id="events-active">${active || '<div class="muted small">没有活跃告警</div>'}</div></div>
    <div class="card" style="margin-top:12px"><div class="card-head"><h2>事件</h2><span class="muted small">来自 AnomalyDetector 与 monitor 规则</span></div><div class="table-scroll"><table class="table"><thead><tr><th class="num">step</th><th>级别</th><th>kind</th><th class="num">rank</th><th>message</th><th class="num">time</th></tr></thead><tbody id="events-tbody">${rows || '<tr><td colspan="6" class="muted">没有事件</td></tr>'}</tbody></table></div></div>
    <details class="config" data-detail-key="events-rules" style="margin-top:12px"><summary>告警规则（节选）</summary><table class="table rule-table"><thead><tr><th>kind</th><th>级别</th><th>条件</th></tr></thead><tbody>${rules}</tbody></table></details>`;
  wireAckButtons(body);
}

function refreshEventsInPlace() {
  const { active, rows } = eventsContentMarkup();
  const activeEl = $('#events-active');
  if (activeEl) {
    activeEl.innerHTML = active || '<div class="muted small">没有活跃告警</div>';
    wireAckButtons(activeEl);
  }
  const tbody = $('#events-tbody');
  if (tbody) tbody.innerHTML = rows || '<tr><td colspan="6" class="muted">没有事件</td></tr>';
}

function systemPartsMarkup(selected) {
  const hosts = (state.system && state.system.hosts) || {};
  const entry = hosts[selected];
  const latest = (entry && entry.latest) || {};
  const host = latest.host || {};
  const devices = latest.devices || [];
  const cov = (state.run && state.run.coverage) || {};
  const stripHosts = (cov.hosts || []).map((item) => {
    const total = (item.ranks || []).length;
    const cls = item.live > 0 ? 'ok' : (item.reporting > 0 ? 'warn' : 'bad');
    const system = item.system_age_s != null ? `sys ${ago(item.system_age_s)}` : 'no sys';
    const selCls = item.hostname === selected ? ' selected' : '';
    return `<button class="chip ${cls}${selCls}" data-sys-host="${esc(item.hostname)}">${esc(item.hostname)} · ${int(item.live)}/${int(total)} ranks · ${esc(system)}</button>`;
  }).join('');
  const stripHtml = cov.nodes_expected ? `<div class="page-actions"><span class="chip">${int(cov.nodes_live)}/${int(cov.nodes_expected)} nodes live</span><span class="chip">${int(cov.nodes_system)}/${int(cov.nodes_expected)} system samples</span><span class="chip">${int(cov.reporting_ranks)}/${int(cov.world)} ranks reporting</span></div><div class="node-strip">${stripHosts}</div>` : '';
  const kpisHtml = `
    <div class="kpi"><div class="label">CPU</div><div class="value">${num(host.cpu_percent, 0)}<small>%</small></div><div class="hint">load ${num(host.load1, 1)} / ${num(host.load5, 1)}</div></div>
    <div class="kpi"><div class="label">Memory</div><div class="value">${num((host.memory || {}).available_gib, 0)}<small>GiB free</small></div><div class="hint">${num((host.memory || {}).total_gib, 0)} GiB total</div></div>
    <div class="kpi"><div class="label">/dev/shm</div><div class="value">${num((host.shm || {}).free_gib, 0)}<small>GiB free</small></div><div class="hint">${num((host.shm || {}).total_gib, 0)} GiB total</div></div>
    <div class="kpi"><div class="label">Training processes</div><div class="value">${int(latest.training_processes)}</div><div class="hint">${devices.length} NPUs visible</div></div>`;
  const devicesHtml = devices.map((device) => {
    const used = device.hbm_used_mib != null ? device.hbm_used_mib / 1024 : null;
    const total = device.hbm_total_mib != null ? device.hbm_total_mib / 1024 : null;
    const frac = used != null && total ? used / total : null;
    const loadCls = frac != null && frac >= 0.92 ? ' crit' : (frac != null && frac >= 0.8 ? ' hot' : '');
    const bar = frac != null ? `<div class="hbm-bar"><span style="width:${(Math.min(1, frac) * 100).toFixed(0)}%"></span></div>` : '';
    return `<div class="topo-card${loadCls}">
      <div class="idx"><span>NPU ${esc(device.index)}</span><span>${device.process_count != null ? int(device.process_count) : (device.processes || []).length} proc</span></div>
      <div class="rank">${num(used, 1)}<span class="muted small">/${num(total, 1)}G</span></div>
      <div class="hbm">${frac != null ? pct(frac, 0) : '—'} · ${num(device.temperature_c, 0)}°C · ${num(device.power_w, 0)}W</div>
      ${bar}
    </div>`;
  }).join('');
  const processesHtml = (latest.processes || []).map((proc) => `<tr><td class="num">${int(proc.pid)}</td><td class="num">${proc.rank == null ? '—' : int(proc.rank)}</td><td>${esc(proc.state || '—')}</td><td>${esc(proc.wchan || '—')}</td><td class="mono ellipsis" title="${esc(proc.command || '')}">${esc(proc.command || '')}</td></tr>`).join('');
  const collectorOk = !(latest.collector && latest.collector.errors && latest.collector.errors.length);
  const hostMetaHtml = `<span class="chip">采集 ${esc(ago(entry && entry.age_s))}</span><span class="chip ${collectorOk ? 'ok' : 'bad'}">${collectorOk ? 'collector ok' : esc(latest.collector.errors[0])}</span>`;
  return { stripHtml, kpisHtml, devicesHtml, processesHtml, hostMetaHtml };
}

function renderSystem(body) {
  disposeCharts();
  if (state.system == null) {
    body.innerHTML = '<div class="card"><div class="muted small">读取系统采集中…</div></div>';
    return;
  }
  const hosts = (state.system && state.system.hosts) || {};
  const names = Object.keys(hosts);
  if (!names.length) {
    body.innerHTML = '<div class="card empty"><h2>还没有系统采集</h2><p>启动每节点 sidecar：</p><pre class="mono">python -m examples.worldmodel.cosmos3_monitor collect --run-dir &lt;out&gt; --interval 5</pre></div>';
    return;
  }
  const selected = state.systemHost && hosts[state.systemHost] ? state.systemHost : names[0];
  state.systemHost = selected;
  const parts = systemPartsMarkup(selected);
  body.innerHTML = `${parts.stripHtml ? `<div class="card" style="margin-bottom:12px" id="sys-strip">${parts.stripHtml}</div>` : ''}
    <div class="card" style="margin-bottom:12px"><div class="page-actions"><label class="hint-inline">节点</label><select id="sys-host">${names.map((name) => `<option ${name === selected ? 'selected' : ''}>${esc(name)}</option>`).join('')}</select><span id="sys-host-meta" class="page-actions">${parts.hostMetaHtml}</span></div></div>
    <div class="grid k4" id="sys-kpis">${parts.kpisHtml}</div>
    <div class="card" style="margin-top:12px"><div class="card-head"><h2>NPU</h2><span class="muted small">HBM / 温度 / 功率（AICore% 已排除）</span></div><div class="topo-cards" id="sys-devices">${parts.devicesHtml || '<span class="muted small">没有 NPU 数据</span>'}</div></div>
    <div class="grid k2" style="margin-top:12px">
      <div class="card chart-card"><div class="chart-head"><div class="chart-title-group"><h2>HBM 使用</h2><span class="unit-badge">GiB</span></div></div><div class="chart-wrap"><div id="chart-sys-hbm" class="chart"></div><div class="chart-empty" data-empty-for="chart-sys-hbm" hidden></div></div></div>
      <div class="card chart-card"><div class="chart-head"><div class="chart-title-group"><h2>温度</h2><span class="unit-badge">°C</span></div></div><div class="chart-wrap"><div id="chart-sys-temp" class="chart"></div><div class="chart-empty" data-empty-for="chart-sys-temp" hidden></div></div></div>
    </div>
    <div class="card" style="margin-top:12px"><div class="card-head"><h2>进程</h2><span class="muted small">来自 /proc；rank 由 RANK 环境变量映射</span></div><div class="table-scroll"><table class="table"><thead><tr><th class="num">pid</th><th class="num">rank</th><th>state</th><th>wchan</th><th>command</th></tr></thead><tbody id="sys-procs">${parts.processesHtml || '<tr><td colspan="5" class="muted">没有训练进程</td></tr>'}</tbody></table></div></div>`;
  const selector = $('#sys-host');
  if (selector) selector.onchange = (event) => {
    state.systemHost = event.target.value;
    saveRunPrefs();
    renderSystem(body);
  };
  $$('[data-sys-host]').forEach((button) => {
    button.onclick = () => {
      state.systemHost = button.dataset.sysHost;
      saveRunPrefs();
      renderSystem(body);
    };
  });
  renderSystemCharts(selected);
}

function refreshSystemInPlace() {
  const hosts = (state.system && state.system.hosts) || {};
  const names = Object.keys(hosts);
  if (!names.length) return;
  const body = $('#tab-body');
  if (!body || !$('#chart-sys-hbm')) {
    if (body) renderSystem(body);
    return;
  }
  const selected = state.systemHost && hosts[state.systemHost] ? state.systemHost : names[0];
  state.systemHost = selected;
  const parts = systemPartsMarkup(selected);
  const strip = $('#sys-strip');
  if (strip && parts.stripHtml) {
    strip.innerHTML = parts.stripHtml;
    $$('[data-sys-host]').forEach((button) => {
      button.onclick = () => { state.systemHost = button.dataset.sysHost; saveRunPrefs(); renderSystem(body); };
    });
  }
  const meta = $('#sys-host-meta');
  if (meta) meta.innerHTML = parts.hostMetaHtml;
  const kpis = $('#sys-kpis');
  if (kpis) kpis.innerHTML = parts.kpisHtml;
  const devs = $('#sys-devices');
  if (devs) devs.innerHTML = parts.devicesHtml || '<span class="muted small">没有 NPU 数据</span>';
  const procs = $('#sys-procs');
  if (procs) procs.innerHTML = parts.processesHtml || '<tr><td colspan="5" class="muted">没有训练进程</td></tr>';
  // Refresh charts in-place without disposing -> zoom stays intact!
  renderSystemCharts(selected);
}

function updateLogView() {
  const target = $('#log-body');
  if (!target) return;
  const q = state.logFilter.trim().toLowerCase();
  const filtered = q ? state.logLines.filter((line) => line.toLowerCase().includes(q)) : state.logLines;
  const countEl = $('#log-count');
  if (countEl) countEl.textContent = q ? `${filtered.length}/${state.logLines.length} 行` : `${state.logLines.length} 行`;
  target.textContent = filtered.join('\n') || '（空）';
  if (state.logFollow) target.scrollTop = target.scrollHeight;
}

async function renderLogs(body) {
  body.innerHTML = `<div class="card"><div class="card-head">
      <div class="card-head-left"><h2>日志</h2><span id="log-count" class="chip">loading…</span></div>
      <div class="page-actions">
        <input id="log-filter" placeholder="过滤关键字…" value="${esc(state.logFilter)}" style="width:180px">
        <label class="checkbox"><input type="checkbox" id="log-follow" ${state.logFollow ? 'checked' : ''}> 自动滚到底部</label>
        <button id="log-refresh" class="ghost">刷新</button>
      </div>
    </div><div id="log-body" class="log">读取中…</div></div>`;
  const refresh = $('#log-refresh');
  if (refresh) refresh.onclick = () => renderLogs(body);
  const filter = $('#log-filter');
  if (filter) filter.oninput = () => { state.logFilter = filter.value; updateLogView(); };
  const follow = $('#log-follow');
  if (follow) follow.onchange = () => { state.logFollow = follow.checked; persistGlobalPrefs(); updateLogView(); };
  try {
    const data = await api(`/api/runs/${state.token}/log?tail=500`);
    const target = $('#log-body');
    if (!target || state.tab !== 'logs') return;  // the user moved to another tab while the fetch was pending
    state.logLines = data.lines || [];
    updateLogView();
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

function setLiveBadge(mode) {
  const badge = $('#live-badge');
  if (!badge) return;
  if (mode === 'paused') {
    badge.innerHTML = '<span class="live-dot"></span>paused';
    badge.className = 'chip warn';
  } else if (mode === 'polling') {
    badge.innerHTML = '<span class="live-dot"></span>polling';
    badge.className = 'chip warn';
  } else {
    badge.innerHTML = '<span class="live-dot"></span>live';
    badge.className = 'chip ok';
  }
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
          // the tab body with an empty header, which looked like the selected tab had closed itself.
          updateHeaderMeta();
          renderKpis();
          setLiveBadge('live');
        }
      } catch (_) { /* a malformed frame is ignored */ }
    });
    stream.onerror = () => {
      if (!state.paused) setLiveBadge('polling');
    };
  } catch (_) { /* EventSource is optional; the 5 s poll still works */ }
}

/* --------------------------------------------------------------------------------------------------------- */
/* Routing / boot                                                                                             */
/* --------------------------------------------------------------------------------------------------------- */
function route() {
  const [pathPart] = location.hash.replace(/^#\/?/, '').split('?');
  const parts = pathPart.split('/');
  if (parts[0] === 'run' && parts[1]) {
    openRun(parts[1], parts[2] || 'overview');
  } else if (parts[0] === 'compare') {
    renderComparePage();
  } else {
    state.token = null;
    state.run = null;
    closeChartModal();
    disposeCharts();
    closeStream();
    renderRunsPage();
  }
}

function renderComparePage() {
  disposeCharts();
  const selected = (location.hash.split('?')[1] || '').split('&').filter((x) => x.startsWith('runs='))[0];
  const tokens = selected ? decodeURIComponent(selected.slice(5)).split(',').filter(Boolean) : [];
  state.compareTokens = new Set(tokens);
  const options = state.runs.map((run) => `<option value="${run.token}" ${tokens.includes(run.token) ? 'selected' : ''}>${esc(run.display_name)}</option>`).join('');
  const chips = state.runs.map((run) => `<button class="chip cmp-chip ${state.compareTokens.has(run.token) ? 'active' : ''}" data-cmp-token="${run.token}">${esc(run.display_name)}</button>`).join('');
  $('#crumbs').innerHTML = '<a href="#/runs">Runs</a> <span>/</span> <strong>Compare</strong>';
  $('#view').innerHTML = `<div class="page-head"><div><h1>多运行对比</h1><div class="subtle">按 step 对齐多个 run；滚轮联动放缩，双击重置</div></div></div>
    <div class="card">
      <div class="cmp-chip-grid" id="cmp-chips">${chips || '<span class="muted small">没有可对比的 run</span>'}</div>
      <div class="page-actions">
        <select id="cmp-runs" multiple size="4" style="min-width:320px;display:none">${options}</select>
        <button id="cmp-go" class="primary">开始对比 (${state.compareTokens.size})</button>
        <button id="cmp-reset" class="ghost">重置缩放</button>
      </div>
    </div>
    <div class="grid k2" style="margin-top:12px">
      <div class="card chart-card"><div class="chart-head"><div class="chart-title-group"><h2>Loss</h2><span class="unit-badge">loss</span></div></div><div id="cmp-loss" class="chart tall"></div></div>
      <div class="card chart-card"><div class="chart-head"><div class="chart-title-group"><h2>Step time</h2><span class="unit-badge">seconds</span></div></div><div id="cmp-step" class="chart tall"></div></div>
      <div class="card chart-card"><div class="chart-head"><div class="chart-title-group"><h2>Throughput</h2><span class="unit-badge">samples/s</span></div></div><div id="cmp-tp" class="chart"></div></div>
      <div class="card chart-card"><div class="chart-head"><div class="chart-title-group"><h2>HBM Used</h2><span class="unit-badge">GiB</span></div></div><div id="cmp-mem" class="chart"></div></div>
    </div>
    <div class="card" style="margin-top:12px"><div id="cmp-table" class="muted small">选择至少两个 run 后点击开始对比</div></div>`;

  const syncCompareSelect = () => {
    const sel = $('#cmp-runs');
    if (sel) Array.from(sel.options).forEach((opt) => { opt.selected = state.compareTokens.has(opt.value); });
    const go = $('#cmp-go');
    if (go) { go.textContent = `开始对比 (${state.compareTokens.size})`; go.disabled = state.compareTokens.size < 2; }
  };
  $$('#cmp-chips .cmp-chip').forEach((chip) => {
    chip.onclick = () => {
      const tok = chip.dataset.cmpToken;
      if (state.compareTokens.has(tok)) state.compareTokens.delete(tok);
      else state.compareTokens.add(tok);
      chip.classList.toggle('active', state.compareTokens.has(tok));
      syncCompareSelect();
    };
  });
  syncCompareSelect();
  const cmpReset = $('#cmp-reset');
  if (cmpReset) cmpReset.onclick = () => resetZoomForScope('compare');

  const runCompare = async () => {
    const sel = $('#cmp-runs');
    const chosen = state.compareTokens.size ? [...state.compareTokens] : (sel ? Array.from(sel.selectedOptions).map((option) => option.value) : []);
    if (chosen.length < 2) { toast('至少选两个 run', 'bad'); return; }
    const query = `run_ids=${chosen.join(',')}&keys=loss/total,perf/step_s,perf/samples_per_s,mem/device_used_gib&max_points=3000`;
    try {
      const data = await api(`/api/compare?${query}`);
      const p = palette();
      const cmpBase = () => {
        const opt = baseOption();
        opt.xAxis.minInterval = 1;
        opt.xAxis.axisLabel.formatter = (v) => (v >= 1000 ? `${+(v / 1000).toFixed(2)}k` : v);
        opt.series = [];
        return opt;
      };
      const loss = cmpBase();
      const step = cmpBase();
      const tp = cmpBase();
      const mem = cmpBase();
      let index = 0;
      for (const [token, entry] of Object.entries(data.runs || {})) {
        const color = p[index % p.length];
        const name = entry.run_id || token;
        loss.series.push(line(name, pointsOf(entry.series, 'loss/total', true), color));
        step.series.push(line(name, pointsOf(entry.series, 'perf/step_s', true), color));
        tp.series.push(line(name, pointsOf(entry.series, 'perf/samples_per_s', true), color));
        mem.series.push(line(name, pointsOf(entry.series, 'mem/device_used_gib', true), color));
        index += 1;
      }
      makeChart('cmp-loss', loss, 'compare');
      makeChart('cmp-step', step, 'compare');
      makeChart('cmp-tp', tp, 'compare');
      makeChart('cmp-mem', mem, 'compare');
      const rows = Object.entries(data.runs || {}).map(([token, entry]) => {
        const lastLoss = (pointsOf(entry.series, 'loss/total', true).slice(-1)[0] || [])[1];
        const lastStep = (pointsOf(entry.series, 'perf/step_s', true).slice(-1)[0] || [])[1];
        const lastTp = (pointsOf(entry.series, 'perf/samples_per_s', true).slice(-1)[0] || [])[1];
        const lastMem = (pointsOf(entry.series, 'mem/device_used_gib', true).slice(-1)[0] || [])[1];
        return `<tr><td><a href="#/run/${token}/overview">${esc(entry.run_id || token)}</a></td><td class="num">${pointsOf(entry.series, 'loss/total', true).length}</td><td class="num">${num(lastLoss, 4)}</td><td class="num">${num(lastStep, 2)}s</td><td class="num">${num(lastTp, 1)}</td><td class="num">${num(lastMem, 1)} GiB</td></tr>`;
      }).join('');
      $('#cmp-table').innerHTML = `<table class="table"><thead><tr><th>run</th><th class="num">points</th><th class="num">last loss</th><th class="num">last step</th><th class="num">samples/s</th><th class="num">HBM</th></tr></thead><tbody>${rows}</tbody></table>`;
    } catch (error) { toast(error.message, 'bad'); }
  };
  $('#cmp-go').onclick = runCompare;
  if (tokens.length >= 2) runCompare();
}

function applySidebarCollapsed() {
  const appEl = $('#app');
  if (appEl) appEl.classList.toggle('sidebar-collapsed', state.sidebarCollapsed);
}

function boot() {
  document.documentElement.dataset.theme = state.theme;
  applySidebarCollapsed();

  const sidebarToggle = $('#sidebar-toggle');
  if (sidebarToggle) {
    sidebarToggle.onclick = () => {
      state.sidebarCollapsed = !state.sidebarCollapsed;
      persistGlobalPrefs();
      applySidebarCollapsed();
      setTimeout(() => {
        for (const chart of state.charts.values()) {
          if (!chart.isDisposed()) chart.resize();
        }
      }, 200);
    };
  }

  $$('#sidebar-filters .filter-pill').forEach((btn) => {
    btn.onclick = () => {
      state.sidebarFilter = btn.dataset.filter;
      persistGlobalPrefs();
      renderSidebar();
    };
  });

  const search = $('#run-search');
  const clearBtn = $('#run-search-clear');
  let searchTimer = null;
  search.addEventListener('input', () => {
    state.search = search.value.trim();
    if (clearBtn) clearBtn.hidden = !search.value;
    clearTimeout(searchTimer);
    searchTimer = setTimeout(loadRuns, 180);
  });
  if (clearBtn) {
    clearBtn.onclick = () => {
      search.value = '';
      state.search = '';
      clearBtn.hidden = true;
      loadRuns();
    };
  }

  const refreshBtn = $('#refresh-now');
  if (refreshBtn) {
    refreshBtn.onclick = async () => {
      await Promise.all([loadHealth(), loadRuns()]);
      if (state.token) {
        const wasPaused = state.paused;
        state.paused = false;
        await refreshDetailQuiet();
        state.paused = wasPaused;
      }
      toast('已刷新最新数据');
    };
  }

  const pause = $('#pause-toggle');
  if (pause) {
    pause.onclick = () => {
      state.paused = !state.paused;
      pause.textContent = state.paused ? '继续' : '暂停';
      pause.classList.toggle('active', state.paused);
      setLiveBadge(state.paused ? 'paused' : 'live');
    };
  }

  const help = $('#shortcut-help');
  if (help) {
    help.onclick = () => toast('快捷键：/ 搜索 · b 侧边栏 · p 暂停 · r 刷新 · d 明暗 · Esc 关闭弹窗 · ←/→ 上/下一步');
  }

  const drawerClose = $('#drawer-close');
  if (drawerClose) drawerClose.onclick = closeStepDetail;
  const drawerBackdrop = $('#drawer-backdrop');
  if (drawerBackdrop) drawerBackdrop.onclick = closeStepDetail;

  const modalClose = $('#chart-modal-close');
  if (modalClose) modalClose.onclick = closeChartModal;
  const modalBackdrop = $('#chart-modal-backdrop');
  if (modalBackdrop) modalBackdrop.onclick = (event) => { if (event.target === modalBackdrop) closeChartModal(); };
  const modalReset = $('#chart-modal-reset');
  if (modalReset) modalReset.onclick = () => {
    if (!state.modalChartId) return;
    const scope = state.modalChartId.endsWith('-overview') ? 'overview' : 'charts';
    resetZoomForScope(scope, state.syncZoom ? null : state.modalChartId);
  };

  $('#theme-toggle').onclick = () => {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = state.theme;
    localStorage.setItem('c3-theme', state.theme);
    if (state.run) renderRunView();
    else if (location.hash.startsWith('#/compare')) renderComparePage();
    else renderRunsPage();
    if (state.modalRenderer) state.modalRenderer();
  };

  document.addEventListener('keydown', (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
      if (event.key === 'Escape') event.target.blur();
      return;
    }
    if (event.key === 'Escape') {
      if (state.modalChartId) closeChartModal();
      else closeStepDetail();
      return;
    }
    if (state.detail && event.key === 'ArrowLeft') { openStepDetail(Number(state.detail.iteration) - 1); return; }
    if (state.detail && event.key === 'ArrowRight') { openStepDetail(Number(state.detail.iteration) + 1); return; }
    if (event.key === '?') toast('快捷键：/ 搜索 · b 侧边栏 · p 暂停 · r 刷新 · d 明暗 · Esc 关闭弹窗 · ←/→ 上/下一步', '');
    if (event.key === '/') { event.preventDefault(); search.focus(); }
    if (event.key === 'd') $('#theme-toggle').click();
    if (event.key === 'b' && sidebarToggle) sidebarToggle.click();
    if (event.key === 'p' && pause) pause.click();
    if (event.key === 'r' && refreshBtn) refreshBtn.click();
  });

  window.addEventListener('hashchange', route);
  loadHealth();
  loadRuns().then(route);
  state.runsTimer = setInterval(() => { loadHealth(); loadRuns(); }, 5000);
}

boot();
