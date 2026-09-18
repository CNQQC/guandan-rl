'use strict';
const $ = id => document.getElementById(id);
let game = null, selected = new Set(), overview = null, busy = false;
let toastTimer;
const names = ['你', '右侧对手', '你的搭档', '左侧对手'];
const escapeHTML = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = x => typeof x === 'number' ? x.toLocaleString('zh-CN') : '—';
const percent = x => typeof x === 'number' ? (x * 100).toFixed(1) + '%' : '—';
function toast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 4500); }
async function api(path, payload, method = 'POST') {
  const r = await fetch(path, payload === undefined ? {}
    : {method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  const data = await r.json();
  if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail
    : Array.isArray(data.detail) ? data.detail.map(d => `${(d.loc || []).at(-1)} ${d.msg}`).join('；')
    : `请求失败 (${r.status})`);
  return data;
}
async function gameAction(fn) {
  if (busy) return;
  busy = true; updateButtons(); $('new-game').disabled = true;
  try { await fn(); } catch (error) { toast(error.message); }
  finally { busy = false; $('new-game').disabled = false; updateButtons(); }
}
/* ---------- 浮层：下拉框与菜单共用一套开合与定位 ---------- */
let layer = null;
function placeLayer(panel, anchor) {
  const box = anchor.getBoundingClientRect();
  panel.style.minWidth = box.width + 'px';
  panel.style.maxHeight = '';
  panel.style.left = panel.style.top = '0px';
  const below = innerHeight - box.bottom - 12, above = box.top - 12;
  // Open upwards only when the panel genuinely does not fit below.
  const flip = below < Math.min(panel.offsetHeight, 200) && above > below;
  panel.style.maxHeight = Math.max(140, flip ? above : below) + 'px';
  panel.style.top = (flip ? Math.max(8, box.top - panel.offsetHeight - 6) : box.bottom + 6) + 'px';
  panel.style.left = Math.max(8, Math.min(box.left, innerWidth - panel.offsetWidth - 8)) + 'px';
}
function closeLayer() { const open = layer; layer = null; if (open) open.close(); }
function showLayer(anchor, panel, close) {
  closeLayer();
  panel.hidden = false;
  placeLayer(panel, anchor);
  layer = {anchor, panel, close};
}
addEventListener('pointerdown', event => {
  if (layer && !layer.panel.contains(event.target) && !layer.anchor.contains(event.target)) closeLayer();
}, true);
addEventListener('keydown', event => {
  if (event.key === 'Escape' && layer) { const anchor = layer.anchor; closeLayer(); anchor.focus(); }
});
addEventListener('resize', closeLayer);
addEventListener('scroll', () => { if (layer) placeLayer(layer.panel, layer.anchor); }, true);

/* ---------- 下拉框：原生 select 留作数据源，外观与弹层自绘 ---------- */
// Call sites keep using select.value / .options / onchange; only the operating
// system's popup is replaced, so the control matches the rest of the page.
function enhanceSelect(select) {
  if (select.dataset.custom) return;
  select.dataset.custom = '1';
  const root = document.createElement('span');
  root.className = 'select';
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'select-button';
  button.setAttribute('aria-haspopup', 'listbox');
  button.setAttribute('aria-expanded', 'false');
  // A long option must shorten inside the control, never wrap it onto two lines.
  const text = document.createElement('span');
  text.className = 'select-text';
  button.append(text);
  const panel = document.createElement('div');
  panel.className = 'select-panel';
  panel.setAttribute('role', 'listbox');
  panel.hidden = true;
  select.replaceWith(root);
  // The button goes first so a wrapping <label> points at it, not the hidden select.
  root.append(button, select, panel);
  select.tabIndex = -1;
  select.setAttribute('aria-hidden', 'true');
  const items = () => [...panel.querySelectorAll('.select-option:not(.disabled)')];
  const sync = () => {
    const option = select.selectedOptions[0];
    text.textContent = option ? option.textContent : '—';
    button.disabled = select.disabled;
    root.hidden = select.hidden;
    for (const item of panel.querySelectorAll('.select-option')) {
      const on = item.dataset.value === select.value;
      item.classList.toggle('on', on);
      item.setAttribute('aria-selected', String(on));
    }
  };
  const build = () => {
    const append = option => {
      const item = document.createElement('div');
      item.className = 'select-option' + (option.disabled ? ' disabled' : '');
      item.setAttribute('role', 'option');
      item.tabIndex = -1;
      item.dataset.value = option.value;
      item.textContent = option.textContent;
      panel.append(item);
    };
    panel.innerHTML = '';
    for (const node of select.children) {
      if (node.tagName !== 'OPTGROUP') { append(node); continue; }
      const head = document.createElement('div');
      head.className = 'select-group';
      head.textContent = node.label;
      panel.append(head);
      [...node.children].forEach(append);
    }
    sync();
    // A background refresh may rebuild the list while it is open and focused.
    if (!panel.hidden) (panel.querySelector('.select-option.on') || items()[0])?.focus();
  };
  const close = () => { panel.hidden = true; button.setAttribute('aria-expanded', 'false'); };
  const open = () => {
    if (select.disabled || !select.options.length) return;
    showLayer(button, panel, close);
    button.setAttribute('aria-expanded', 'true');
    (panel.querySelector('.select-option.on') || items()[0])?.focus();
  };
  const commit = item => {
    select.value = item.dataset.value;
    select.dispatchEvent(new Event('change', {bubbles: true}));
    closeLayer();
    button.focus();
  };
  button.addEventListener('click', () => panel.hidden ? open() : closeLayer());
  button.addEventListener('keydown', event => {
    if (['ArrowDown', 'ArrowUp', 'Enter', ' '].includes(event.key)) { event.preventDefault(); open(); }
  });
  panel.addEventListener('click', event => {
    const item = event.target.closest('.select-option:not(.disabled)');
    if (item) commit(item);
  });
  panel.addEventListener('keydown', event => {
    const all = items(), at = all.indexOf(document.activeElement);
    const focus = index => { event.preventDefault(); all[(index + all.length) % all.length]?.focus(); };
    if (event.key === 'ArrowDown') focus(at + 1);
    else if (event.key === 'ArrowUp') focus(at - 1);
    else if (event.key === 'Home') focus(0);
    else if (event.key === 'End') focus(all.length - 1);
    else if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); if (at >= 0) commit(all[at]); }
    else if (event.key === 'Tab') closeLayer();
    else if (event.key.length === 1) {
      const key = event.key.toLowerCase();
      (all.slice(at + 1).concat(all).find(item => item.textContent.toLowerCase().startsWith(key)))?.focus();
    }
  });
  new MutationObserver(records => records.some(r => r.type === 'childList') ? build() : sync())
    .observe(select, {childList: true, subtree: true, attributes: true,
                      attributeFilter: ['disabled', 'hidden']});
  // Assignments such as picker.value = name carry no event; mirror them here.
  const native = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value');
  Object.defineProperty(select, 'value', {configurable: true, get: () => native.get.call(select),
                                          set: value => { native.set.call(select, value); sync(); }});
  build();
}
const enhanceSelects = (root = document) => root.querySelectorAll('select').forEach(enhanceSelect);

document.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.nav,.tab').forEach(el => el.classList.remove('active'));
  button.classList.add('active'); $(button.dataset.tab).classList.add('active');
  if (button.dataset.tab === 'train') renderTraining();
}));
function sameCards(action, cards) {
  // Deck copies of the same face are equivalent to the engine's canonical IDs.
  const a = action.cards.map(c => c.id % 54).sort((x,y)=>x-y);
  const b = [...cards].map(c => c % 54).sort((x,y)=>x-y);
  return a.length === b.length && a.every((v,i)=>v===b[i]);
}
function matching() { return game && selected.size ? game.legal.filter(a => sameCards(a, selected)) : []; }
function updateButtons() {
  const active = game && !game.result && !busy;
  const matches = matching();
  $('hint').disabled = !active;
  $('clear').disabled = !active || !selected.size;
  $('pass').disabled = !active || !game.legal.some(a => a.size === 0);
  $('play-button').disabled = !active || !matches.length;
  $('claim').hidden = matches.length < 2;
  const previous = $('claim').value;
  $('claim').innerHTML = matches.map(a => `<option value="${a.index}">${escapeHTML(a.label)}</option>`).join('');
  if (matches.some(a=>String(a.index)===previous)) $('claim').value=previous;
  if (game && !game.result) $('hand-caption').textContent = selected.size ? (matches.length ? `已选 ${selected.size} 张 · ${matches[0].type}` : `已选 ${selected.size} 张 · 暂不能这样出牌`) : '点击选牌，或使用提示';
}
function renderGame() {
  selected.clear();
  $('hand-count').textContent = game.hand.length;
  $('round-label').textContent = `本局打 ${game.level} · 红桃 ${game.level} 为配子`;
  $('table-status').textContent = game.result ? (game.result.won ? '你的队伍获胜' : '对方队伍获胜') : '轮到你出牌';
  $('hand-caption').textContent = game.result ? '本局结束，可以再来一局' : '点击选牌，或使用提示';
  [1,2,3].forEach(p => { $('left-'+p).textContent = game.done.includes(p) ? '已出完' : game.remaining[p] + ' 张'; });
  $('hand').innerHTML = game.hand.map(c => `<button class="playing-card ${c.red?'red':''} ${c.wild?'wild':''}" data-card="${c.id}" aria-label="${escapeHTML(c.suit+c.rank+(c.wild?' 配子':''))}" aria-pressed="false" ${game.result?'disabled':''}><span class="rank">${escapeHTML(c.rank)}</span><span class="suit">${c.suit}</span></button>`).join('');
  $('hand').querySelectorAll('[data-card]').forEach(button => button.addEventListener('click', () => {
    if (busy || game.result) return;
    const c = Number(button.dataset.card);
    if(selected.has(c)) selected.delete(c); else selected.add(c);
    button.classList.toggle('selected', selected.has(c));
    button.setAttribute('aria-pressed', String(selected.has(c))); updateButtons();
  }));
  [0,1,2,3].forEach(p => {
    const last = [...game.history].reverse().find(e=>e.player===p);
    $('move-'+p).textContent = last ? last.move.label : '';
  });
  $('game-log').innerHTML = [...game.history].reverse().map(e=>`<div class="log-entry"><div class="who">${names[e.player]}</div><div class="play">${escapeHTML(e.move.label)}</div></div>`).join('') || '<p class="empty">首轮由你领出。</p>';
  if(game.result) $('game-log').insertAdjacentHTML('afterbegin',`<div class="log-entry"><strong>${game.result.won?'本队获胜':'本队落败'} · ${Math.abs(game.result.rewards[0])} 级</strong><p>${game.result.ranking.map(p=>names[p]).join(' → ')}</p></div>`);
  updateButtons();
}
$('new-game').onclick=()=>gameAction(async()=>{game=await api('/api/game',{agent:$('agent').value,level:Number($('level').value)});renderGame();});
$('clear').onclick=()=>{selected.clear();$('hand').querySelectorAll('.selected').forEach(el=>{el.classList.remove('selected');el.setAttribute('aria-pressed','false');});updateButtons();};
async function play(index) {game=await api('/api/play',{session:game.id,index,version:game.version});renderGame();}
$('play-button').onclick=()=>gameAction(async()=>{const m=matching();await play(m.length>1?Number($('claim').value):m[0].index);});
$('pass').onclick=()=>gameAction(async()=>{await play(game.legal.find(a=>!a.size).index);});
$('hint').onclick=()=>gameAction(async()=>{
  const hint=await api('/api/hint',{session:game.id,index:0,version:game.version});
  const move=game.legal[hint.index];selected=new Set(move.cards.map(c=>c.id));
  $('hand').querySelectorAll('[data-card]').forEach(el=>{const yes=selected.has(Number(el.dataset.card));el.classList.toggle('selected',yes);el.setAttribute('aria-pressed',String(yes));});
  updateButtons();$('claim').value=String(hint.index);if(!move.size)toast('建议：过牌');
});
/* ---------- 训练仪表板 ---------- */
// Categorical hues validated for the paper surface (CVD + contrast); a colour
// always follows the same opponent so it means the same thing in every panel.
const SURFACE = '#fbfaf6', GRID = '#e2e5da', AXIS = '#c7ccbe', INK = '#20392f';
const SERIES = {self:{label:'自我对弈',color:'#8a6a00'}, snapshot:{label:'历史模型快照',color:'#b5407e'},
  'team-rule':{label:'团队规则',color:'#2a6fb0'}, 'program:njupt':{label:'南邮程序',color:'#c4522c'},
  'program:egg-pancake':{label:'蛋饼程序',color:'#0f8a63'}, champion:{label:'历史冠军',color:'#7a4b9c'},
  rule:{label:'单人规则',color:'#5c6b60'}, random:{label:'随机基线',color:'#5c6b60'}};
// Slot order keeps every adjacent pair separable, in this chart and in the mix bar.
const SLOTS = ['self','snapshot','team-rule','program:njupt','program:egg-pancake','champion','rule','random'];
const STATUS = {running:['训练中','running'], completed:['已完成','done'], budget_reached:['时间预算已到','paused'],
  stopped:['已保存并停止','paused'], failed:['训练出错','failed']};
const view = {smooth:true, log:false, table:false, window:0, hovering:false, elapsed:null, stamp:null};
const seriesOf = key => SERIES[key] || {label:key, color:'#5c6b60'};
const slotRank = key => (SLOTS.indexOf(key)+1) || 99;
const fixed = (x, n=3) => typeof x === 'number' && isFinite(x) ? x.toFixed(n) : '—';
const lossRow = run => (run.metrics || []).at(-1) || {};
function clock(seconds) {
  if (!(seconds >= 0)) return '—';
  const s = Math.round(seconds), h = Math.floor(s/3600), m = Math.floor(s%3600/60);
  return (h ? h + ':' + String(m).padStart(2,'0') : String(m)) + ':' + String(s%60).padStart(2,'0');
}
function ago(iso) {
  const t = Date.parse(iso);
  if (!isFinite(t)) return '';
  const d = Math.max(0, (Date.now()-t)/1000);
  return d < 60 ? `${Math.round(d)} 秒前` : d < 3600 ? `${Math.round(d/60)} 分钟前` : `${Math.round(d/3600)} 小时前`;
}
function niceTicks(lo, hi, count) {
  if (!(hi > lo)) return [lo];
  const rough = (hi-lo)/count, mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const step = [1,2,2.5,5,10].map(m => m*mag).find(s => s >= rough-1e-12) || 10*mag;
  const out = [];
  for (let v = Math.ceil(lo/step-1e-9)*step; v <= hi+1e-9; v += step) out.push(Math.abs(v) < step*1e-9 ? 0 : v);
  return out;
}
// Log axes are labelled with round decade values, never with 10^k fractions.
function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of [1, 2, 3, 5, 7]) {
      const v = m*Math.pow(10, e);
      if (v >= lo && v <= hi) out.push(v);
    }
  return out.length >= 3 ? out : niceTicks(lo, hi, 3);
}
const axisNumber = v => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1)
  : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3);
const ema = (values, alpha=.3) => { let m = null; return values.map(v => m = m===null ? v : m+alpha*(v-m)); };

function plotSpec(node, spec) {
  node.classList.toggle('table-view', view.table && spec.table !== false);
  if (!spec.series.some(s => s.points.length)) {
    node.innerHTML = `<p class="empty">${escapeHTML(spec.empty || '暂无数据。')}</p>`;
    return;
  }
  if (view.table && spec.table !== false) return plotTable(node, spec);
  const width = Math.max(240, node.clientWidth || 620), height = node.clientHeight || 230;
  const pad = {l: spec.padLeft ?? 46, r: 16, t: 12, b: 24};
  const iw = width-pad.l-pad.r, ih = height-pad.t-pad.b;
  const bands = spec.bands || [], refs = spec.refs || [], events = spec.events || [];
  const xs = spec.series.flatMap(s => s.points.map(p => p.x));
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  if (x1 === x0) { x0 -= 1; x1 += 1; }
  const values = spec.series.flatMap(s => s.points.map(p => p.y))
    .concat(bands.flatMap(b => b.points.flatMap(p => [p.lo, p.hi])), refs.map(r => r.y));
  const log = view.log && spec.log !== false && values.every(v => v > 0);
  const t = log ? Math.log10 : (v => v), inv = log ? (v => Math.pow(10, v)) : (v => v);
  let y0 = spec.min ?? Math.min(...values), y1 = spec.max ?? Math.max(...values);
  if (spec.min === undefined && spec.max === undefined) {
    const room = (y1-y0)*.14 || Math.abs(y1)*.14 || 1;
    y0 -= room; y1 += room;
    if (spec.floor !== undefined && y0 < spec.floor) y0 = spec.floor;
  }
  if (log && !(y0 > 0)) y0 = Math.min(...values.filter(v => v > 0))*.8;
  const ty0 = t(y0), ty1 = t(y1);
  const sx = x => pad.l + (x-x0)/((x1-x0) || 1)*iw;
  const sy = v => pad.t + ih - (t(Math.min(Math.max(v, y0), y1))-ty0)/((ty1-ty0) || 1)*ih;
  const format = spec.format || (v => fixed(v, 2));
  const line = points => points.map(p => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
  const svg = [];
  for (const v of (log ? logTicks(y0, y1) : niceTicks(y0, y1, 4))) {
    const y = sy(v);
    if (y < pad.t-.5 || y > pad.t+ih+.5) continue;
    svg.push(`<line x1="${pad.l}" y1="${y.toFixed(1)}" x2="${width-pad.r}" y2="${y.toFixed(1)}" stroke="${GRID}"/>`,
             `<text x="${pad.l-8}" y="${(y+3.5).toFixed(1)}" text-anchor="end">${escapeHTML(format(v))}</text>`);
  }
  for (const ref of refs) {
    const y = sy(ref.y);
    svg.push(`<line x1="${pad.l}" y1="${y.toFixed(1)}" x2="${width-pad.r}" y2="${y.toFixed(1)}" stroke="${AXIS}"/>`,
             `<text x="${width-pad.r}" y="${(y-5).toFixed(1)}" text-anchor="end">${escapeHTML(ref.label)}</text>`);
  }
  for (const band of bands) {
    if (band.points.length < 2) continue;
    const top = band.points.map(p => `${sx(p.x).toFixed(1)},${sy(p.hi).toFixed(1)}`).join(' L');
    const bottom = [...band.points].reverse().map(p => `${sx(p.x).toFixed(1)},${sy(p.lo).toFixed(1)}`).join(' L');
    svg.push(`<path d="M${top} L${bottom} Z" fill="${band.color}" fill-opacity=".12"/>`);
  }
  for (const s of spec.series) {
    if (s.points.length === 1) svg.push(`<circle cx="${sx(s.points[0].x).toFixed(1)}" cy="${sy(s.points[0].y).toFixed(1)}" r="3.5" fill="${s.color}"/>`);
    else svg.push(`<polyline points="${line(s.points)}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}"`
      + ` stroke-opacity="${s.opacity || 1}" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>`);
    if (s.markers) for (const p of s.points)
      svg.push(`<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="3.5" fill="${s.color}" stroke="${SURFACE}" stroke-width="2"/>`);
  }
  for (const event of events) {
    const x = sx(event.x);
    svg.push(`<line x1="${x.toFixed(1)}" y1="${pad.t}" x2="${x.toFixed(1)}" y2="${pad.t+ih}" stroke="${event.color}" stroke-opacity=".28"/>`,
             `<path d="M${(x-4).toFixed(1)} ${pad.t+ih} L${(x+4).toFixed(1)} ${pad.t+ih} L${x.toFixed(1)} ${pad.t+ih-6} Z" fill="${event.color}"/>`);
  }
  svg.push(`<line x1="${pad.l}" y1="${pad.t+ih}" x2="${width-pad.r}" y2="${pad.t+ih}" stroke="${AXIS}"/>`);
  const xTicks = [...new Set(niceTicks(x0, x1, Math.max(2, Math.round(iw/110))).map(Math.round))].filter(v => v >= x0 && v <= x1);
  for (const v of xTicks) svg.push(`<text x="${sx(v).toFixed(1)}" y="${height-7}" text-anchor="middle">${v}</text>`);
  if (spec.tail !== false) {
    const last = spec.series[0].points.at(-1);
    svg.push(`<text x="${Math.min(width-pad.r, sx(last.x)+7).toFixed(1)}" y="${(sy(last.y)-7).toFixed(1)}" text-anchor="end" fill="${INK}">${escapeHTML(format(last.y))}</text>`);
  }
  svg.push('<g class="focus"></g>');
  node.innerHTML = `<svg viewBox="0 0 ${width} ${height}" height="${height}" tabindex="0" role="img"`
    + ` aria-label="${escapeHTML(spec.label || '训练曲线')}">${svg.join('')}</svg><div class="tip" hidden></div>`;
  attachHover(node, spec, {sx, sy, pad, iw, ih, width, format});
}

function attachHover(node, spec, geo) {
  const svg = node.querySelector('svg'), focus = node.querySelector('.focus'), tip = node.querySelector('.tip');
  const stacked = new Map();
  for (const s of spec.series) { if (s.ghost) continue; for (const p of s.points) {
    if (!stacked.has(p.x)) stacked.set(p.x, {});
    stacked.get(p.x)[s.key] = p.y;
  } }
  const keys = [...stacked.keys()].sort((a, b) => a-b);
  let active = null;
  const show = index => {
    const x = keys[Math.max(0, Math.min(keys.length-1, index))], row = stacked.get(x);
    active = keys.indexOf(x);
    const px = geo.sx(x);
    const marks = spec.series.filter(s => !s.ghost && row[s.key] !== undefined);
    focus.innerHTML = `<line x1="${px.toFixed(1)}" y1="${geo.pad.t}" x2="${px.toFixed(1)}" y2="${geo.pad.t+geo.ih}" stroke="${INK}" stroke-opacity=".3"/>`
      + marks.map(s => `<circle cx="${px.toFixed(1)}" cy="${geo.sy(row[s.key]).toFixed(1)}" r="4" fill="${s.color}" stroke="${SURFACE}" stroke-width="2"/>`).join('');
    tip.innerHTML = `<b>${escapeHTML(spec.xLabel || '迭代')} ${x}</b>`
      + marks.map(s => `<span><i class="swatch" style="background:${s.color}"></i>${escapeHTML(s.label)} ${escapeHTML(geo.format(row[s.key]))}</span>`).join('<br>');
    tip.hidden = false;
    tip.style.left = Math.min(geo.width-70, Math.max(70, px)) + 'px';
    tip.style.top = Math.max(26, Math.min(...marks.map(s => geo.sy(row[s.key])))) + 'px';
  };
  const hide = () => { focus.innerHTML = ''; tip.hidden = true; active = null; };
  const locate = event => {
    const box = svg.getBoundingClientRect(), scale = geo.width/box.width;
    const value = ((event.clientX-box.left)*scale - geo.pad.l)/geo.iw;
    const target = keys[0] + value*(keys.at(-1)-keys[0]);
    let best = 0;
    keys.forEach((k, i) => { if (Math.abs(k-target) < Math.abs(keys[best]-target)) best = i; });
    show(best);
  };
  svg.addEventListener('pointermove', locate);
  svg.addEventListener('pointerleave', hide);
  svg.addEventListener('focus', () => show(active ?? keys.length-1));
  svg.addEventListener('blur', hide);
  svg.addEventListener('keydown', event => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    show((active ?? keys.length-1) + (event.key === 'ArrowRight' ? 1 : -1));
  });
}

function plotTable(node, spec) {
  const rows = new Map(), columns = spec.series.filter(s => !s.ghost);
  for (const s of columns) for (const p of s.points) {
    if (!rows.has(p.x)) rows.set(p.x, {});
    rows.get(p.x)[s.key] = p.y;
  }
  const format = spec.tableFormat || spec.format || (v => fixed(v, 2));
  const keys = [...rows.keys()].sort((a, b) => b-a).slice(0, 80);
  node.innerHTML = `<table class="plot-table"><thead><tr><th>${escapeHTML(spec.xLabel || '迭代')}</th>`
    + columns.map(s => `<th>${escapeHTML(s.label)}</th>`).join('') + '</tr></thead><tbody>'
    + keys.map(x => `<tr><td>${x}</td>` + columns.map(s =>
        `<td>${rows.get(x)[s.key] === undefined ? '—' : escapeHTML(format(rows.get(x)[s.key]))}</td>`).join('') + '</tr>').join('')
    + '</tbody></table>';
}

function curve(rows, key, color, label) {
  const points = rows.filter(r => typeof r[key] === 'number').map(r => ({x: r.iteration, y: r[key]}));
  const series = [];
  if (view.smooth && points.length > 3) {
    series.push({key, label, color, width: 1, opacity: .3, points});
    series.push({key: key+'-ema', label: label+' EMA', color, width: 2, ghost: true,
                 points: ema(points.map(p => p.y)).map((y, i) => ({x: points[i].x, y}))});
  } else series.push({key, label, color, width: 2, points});
  return series;
}

function renderLoss(rows) {
  plotSpec($('plot-loss'), {label: '训练总损失', series: curve(rows, 'loss', INK, '总损失'),
    format: axisNumber, empty: '本次训练还没有写入损失记录。'});
  for (const [id, key, name] of [['plot-q','q_loss','Q 损失'], ['plot-ntp','ntp_loss','NTP 损失'],
                                 ['plot-belief','belief_loss','信念损失'], ['plot-speed','games_per_minute','局 / 分钟']])
    plotSpec($(id), {label: name, series: curve(rows, key, INK, name), padLeft: 42, format: axisNumber});
}

function renderTeamwork(run) {
  // Same probe states at every iteration, so this curve moves only when the
  // policy's use of "who holds the lead" changes.
  const rows = run.teamwork || [];
  const reference = rows.at(-1)?.reference?.flip_rate;
  plotSpec($('plot-teamwork'), {label: '牌权翻转后改判让牌的比例', xLabel: '迭代',
    min: 0, max: 1, padLeft: 42, format: v => (v*100).toFixed(0)+'%', tableFormat: percent,
    series: [{key: 'teamwork', label: '我的模型', color: INK, width: 2, markers: rows.length <= 24,
              points: rows.map(r => ({x: r.iteration, y: r.flip_rate}))}],
    refs: typeof reference === 'number' ? [{y: reference, label: `团队规则参照 ${percent(reference)}`}] : [],
    empty: run.teamwork ? '还没有诊断点：与开发集评测同频，每次评测记录一次。' : '重启控制台后显示对家意识曲线。'});
  const last = rows.at(-1);
  const ratio = last && last.q_margin ? ` · Q 位移 ${fixed(last.deference)}，为自身典型抉择差距 ${fixed(last.q_margin)} 的 ${percent(last.deference/last.q_margin)}` : '';
  $('teamwork-note').textContent = last
    ? `迭代 ${last.iteration}：${last.states} 个固定局面中 ${percent(last.flip_rate)} 改判让牌`
      + `，团队规则在同一批局面上 ${percent(last.reference?.flip_rate)}${ratio}`
    : '与开发集评测同频记录；诊断只读取模型，不参与训练。';
}

function renderEvaluation(run) {
  const groups = new Map();
  for (const report of run.evaluations || []) {
    const key = report.opponent || 'team-rule';
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(report);
  }
  const order = [...groups.keys()].sort((a, b) => slotRank(a)-slotRank(b));
  const series = [], bands = [];
  for (const key of order) {
    const points = groups.get(key).sort((a, b) => a.iteration-b.iteration);
    // Markers stop helping once the run has many evaluation points.
    series.push({key, label: seriesOf(key).label, color: seriesOf(key).color, markers: points.length <= 24,
                 points: points.map(r => ({x: r.iteration, y: r.win_rate}))});
    bands.push({color: seriesOf(key).color,
                points: points.filter(r => r.ci95).map(r => ({x: r.iteration, lo: r.ci95[0], hi: r.ci95[1]}))});
  }
  const gateThreshold = run.training?.gate_threshold;
  const promotions = (groups.get('champion') || []).filter(r => gateThreshold === undefined
    ? (r.ci95 && r.ci95[0] > .5) : r.win_rate > gateThreshold);
  plotSpec($('plot-eval'), {label: '开发集胜率', series, bands, xLabel: '迭代', min: 0, max: 1,
    format: v => (v*100).toFixed(0)+'%', tableFormat: percent, log: false, tail: false,
    refs: [{y: .5, label: '50% 均势'}],
    events: promotions.map(r => ({x: r.iteration, color: seriesOf('champion').color, label: '晋级'})),
    empty: run.evaluations ? '还没有评测点：每 ' + (run.training?.eval_every || '若干') + ' 次迭代评测一次。' : '重启控制台后显示评测曲线。'});
  const latest = key => (groups.get(key) || []).at(-1);
  $('legend-eval').innerHTML = order.map(key => {
    const r = latest(key);
    return `<span><b><i class="swatch" style="background:${seriesOf(key).color}"></i>对${escapeHTML(seriesOf(key).label)}</b>`
      + `<em>${percent(r.win_rate)}${r.ci95 ? ` (${percent(r.ci95[0])}–${percent(r.ci95[1])})` : ''} · ${r.games} 局</em></span>`;
  }).join('') + (promotions.length ? `<span><b>▲ 晋级 ${promotions.length} 次</b><em>${gateThreshold === undefined
      ? '候选模型置信下界超过 50%' : `候选模型对历史冠军胜率超过 ${percent(gateThreshold)}`}</em></span>` : '');
}

function renderMix(rows) {
  const totals = new Map();
  for (const row of rows) for (const [key, count] of Object.entries(row.modes || {}))
    if (count > 0) totals.set(key, (totals.get(key) || 0) + count);
  const keys = [...totals.keys()].sort((a, b) => slotRank(a)-slotRank(b));
  const sum = [...totals.values()].reduce((a, b) => a+b, 0);
  if (!sum) { $('mix').innerHTML = '<p class="empty">暂无采样记录。</p>'; return; }
  $('mix').innerHTML = `<div class="mix-bar">${keys.map(key =>
      `<i style="width:${(totals.get(key)/sum*100).toFixed(2)}%;background:${seriesOf(key).color}"></i>`).join('')}</div>`
    + `<div class="legend">${keys.map(key => `<span><b><i class="swatch" style="background:${seriesOf(key).color}"></i>`
      + `${escapeHTML(seriesOf(key).label)}</b><em>${percent(totals.get(key)/sum)} · ${number(totals.get(key))} 局</em></span>`).join('')}</div>`;
}

function renderIndicators(run) {
  const status = run.status || {}, cfg = run.training || {}, last = lossRow(run), previous = (run.metrics || []).at(-2);
  const [label, kind] = STATUS[status.status] || ['正在初始化', 'idle'];
  $('run-state').className = 'run-state ' + kind;
  $('run-status').textContent = label + (overview.managed_run === run.name ? ' · 网页任务' : '');
  const iterSeconds = previous ? last.elapsed_seconds-previous.elapsed_seconds : null;
  view.stamp = last.timestamp;
  view.elapsed = status.elapsed_seconds ?? last.elapsed_seconds;
  const budget = (cfg.max_minutes || 0)*60;
  $('budget-bar').style.width = budget ? Math.min(100, (view.elapsed || 0)/budget*100).toFixed(1)+'%' : '0%';
  $('budget-label').textContent = budget && status.status === 'running'
    ? `已运行 ${clock(view.elapsed)} / ${clock(budget)} · 剩余 ${clock(budget-view.elapsed)}`
    : `运行时长 ${clock(view.elapsed)}`;
  updateStamp(run);
  // The headline number stays on one opponent: team-rule is the yardstick every run
  // shares, so the figure means the same thing across iterations and across runs.
  const graded = (run.evaluations || []).filter(r => r.opponent !== 'champion');
  const yardstick = graded.filter(r => r.opponent === 'team-rule');
  const winner = (yardstick.length ? yardstick : graded).at(-1);
  const gate = (run.evaluations || []).filter(r => r.opponent === 'champion').at(-1);
  const cells = [
    ['迭代', number(status.iteration ?? last.iteration),
      cfg.eval_every && cfg.snapshot_every ? `每 ${cfg.eval_every} 次评测 · 每 ${cfg.snapshot_every} 次存档` : '—'],
    ['累计对局', number(status.games ?? last.games), last.games_per_minute ? `${fixed(last.games_per_minute, 0)} 局 / 分钟` : '—'],
    ['训练样本', number(status.samples ?? last.samples), `回放池 ${number(last.replay_samples)} / ${number(cfg.replay_size)}`],
    ['更新次数', number(status.updates ?? last.updates), `每迭代 ${cfg.updates ?? '—'} 次 · 批量 ${cfg.batch_size ?? '—'}`],
    ['总损失', fixed(last.loss, 4), `Q ${fixed(last.q_loss)} · NTP ${fixed(last.ntp_loss)} · 信念 ${fixed(last.belief_loss)}`],
    ['探索率 ε', fixed(last.epsilon), `下限 ${fixed(cfg.epsilon_final, 2)} · 随迭代衰减`],
    ['采样占比', iterSeconds > 0 ? percent(last.collect_seconds/iterSeconds) : '—',
      `采样 ${fixed(last.collect_seconds, 2)}s / 迭代 ${fixed(iterSeconds, 2)}s`],
    [winner ? `开发集胜率 · 对${seriesOf(winner.opponent).label}` : '开发集胜率', winner ? percent(winner.win_rate) : '—',
      winner ? `${winner.ci95 ? `95% ${percent(winner.ci95[0])}–${percent(winner.ci95[1])} · ` : ''}${winner.games} 局 · 迭代 ${winner.iteration}`
             : (run.evaluations ? '评测尚未开始' : '重启控制台后显示')],
  ];
  $('metrics').innerHTML = cells.map(([k, v, hint]) =>
    `<div class="metric"><span class="k">${escapeHTML(k)}</span><span class="v">${escapeHTML(v)}</span><span class="h">${escapeHTML(hint)}</span></div>`).join('');
  // Only state what this run actually reported; never fill a gap with a default.
  const facts = [(status.device || cfg.device || '—').toUpperCase(),
    cfg.workers ? `${cfg.workers} 采样进程` : '', cfg.model_size ? `${cfg.model_size} 网络` : '',
    run.snapshots ? `${run.snapshots} 个历史快照` : '', megabytes(run.size),
    run.archived ? '已归档' : '', run.lock && !run.lock.alive ? '有残留训练锁' : '',
    gate ? `最近晋级赛：对历史冠军 ${percent(gate.win_rate)}`
         + (cfg.gate_threshold === undefined ? '' : `（门槛 ${percent(cfg.gate_threshold)}）`)
         : (run.champion ? '已产生冠军' : '')].filter(Boolean);
  $('run-facts').dataset.facts = facts.join(' · ');
}

function updateStamp(run) {
  const facts = $('run-facts').dataset.facts || '';
  const fresh = view.stamp ? ` · 数据更新于 ${ago(view.stamp)}` : '';
  $('run-facts').textContent = facts + fresh;
  if (run) $('run-status').title = run.name;
}

function renderConfiguration(run) {
  const cfg = run.training || {}, status = run.status || {};
  const rows = [['运行设备', (status.device || cfg.device || '—').toUpperCase()], ['采样进程', cfg.workers ?? '—'],
    ['每迭代对局', cfg.games_per_iteration ?? '—'], ['每迭代更新', cfg.updates ?? '—'],
    ['批量大小', cfg.batch_size ?? '—'], ['学习率', cfg.learning_rate ?? '—'],
    ['回放容量', number(cfg.replay_size)], ['ε 起点 / 下限', cfg.epsilon !== undefined ? `${cfg.epsilon} / ${cfg.epsilon_final}` : '—'],
    ['NTP / 信念权重', cfg.ntp_weight !== undefined ? `${cfg.ntp_weight} / ${cfg.belief_weight}` : '—'],
    ['历史模型池', cfg.pool_size === undefined ? '—'
      : `${cfg.pool_size} 个${cfg.pool_recent === undefined ? '' : `（近 ${cfg.pool_recent} + 历史每 ${cfg.pool_archive_every}）`} · 混合 ${percent(cfg.pool_fraction)}`],
    ['评测节奏', cfg.eval_every !== undefined ? `每 ${cfg.eval_every} 次 · ${cfg.eval_pairs} 对` : '—'],
    ['规则对手', (cfg.rule_opponents || []).map(k => seriesOf(k).label).join('、') || '—'],
    ['随机种子', cfg.seed ?? '—'], ['检查点', `${run.snapshots || 0} 个快照 · 冠军${run.champion ? '已产生' : '未产生'}`]];
  $('run-config').innerHTML = rows.map(([k, v]) =>
    `<div><dt>${escapeHTML(k)}</dt><dd>${escapeHTML(String(v))}</dd></div>`).join('');
  const profile = run.name === 'smoke' ? 'smoke' : run.name.startsWith('web-') || run.name.startsWith('league') ? 'league' : 'mac';
  $('resume-command').textContent = `uv run python -m guandan train --config configs/${profile}.toml --out runs/${run.name} --resume runs/${run.name}/latest.pt`;
}

async function refreshLog(run) {
  const node = $('train-log');
  if (run.console === false) {
    node.textContent = '该训练记录没有控制台日志：只有从本页启动的训练会写入 console.log。';
    $('log-note').textContent = '—';
    return;
  }
  if (!capability.log) return;
  try {
    const data = await api(`/api/run/${encodeURIComponent(run.name)}/log`);
    const bottom = node.scrollHeight-node.scrollTop-node.clientHeight < 30;
    node.textContent = data.lines.join('\n') || '（日志为空）';
    $('log-note').textContent = `runs/${run.name}/console.log · ${data.lines.length} 行`;
    if (bottom) node.scrollTop = node.scrollHeight;
  } catch {
    capability.log = false;
    node.textContent = '当前控制台版本没有日志接口：重启 uv run python -m guandan serve 后，这里会实时显示训练输出。';
    $('log-note').textContent = '—';
  }
}

function renderTraining() {
  if (!overview || !overview.runs.length) return;
  const run = overview.runs.find(r => r.name === $('run-picker').value) || overview.runs[0];
  renderIndicators(run);
  renderConfiguration(run);
  renderNodes(run);
  renderResume(run);
  if (document.activeElement !== $('run-alias')) $('run-alias').value = run.label || '';
  if (view.hovering) return;
  const rows = view.window ? (run.metrics || []).slice(-view.window) : (run.metrics || []);
  renderLoss(rows);
  renderEvaluation(run);
  renderTeamwork(run);
  renderMix(run.metrics || []);
}

document.querySelectorAll('.chip[data-option]').forEach(chip => chip.addEventListener('click', () => {
  view[chip.dataset.option] = !view[chip.dataset.option];
  chip.classList.toggle('active', view[chip.dataset.option]);
  chip.setAttribute('aria-pressed', String(view[chip.dataset.option]));
  renderTraining();
}));
$('window-size').onchange = () => { view.window = Number($('window-size').value); renderTraining(); };
$('run-picker').onchange = () => {
  $('run-confirm').hidden = $('node-confirm').hidden = true;
  refresh();
};
document.querySelector('.panels').addEventListener('pointerenter', () => view.hovering = true);
document.querySelector('.panels').addEventListener('pointerleave', () => { view.hovering = false; });
let resizeTimer;
addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderTraining, 180); });
setInterval(() => { if (view.stamp) updateStamp(null); }, 1000);

let refreshSeq = 0;
async function refresh() {
  const seq = ++refreshSeq;
  try {
    const selected = $('run-picker').value;
    const data = await api('/api/overview' + (selected ? '?run='+encodeURIComponent(selected) : ''));
    // A slower earlier poll must not put the picker back on the record it was
    // started for, nor render that record's curves over the one now selected.
    if (seq !== refreshSeq) return;
    overview = data;
    $('runtime').textContent = `${overview.system.platform.startsWith('macOS') ? 'APPLE SILICON' : 'LOCAL'} / ${overview.system.default_device.toUpperCase()} / ${overview.system.memory_gb} GB`;
    $('expert-link').hidden = !overview.expert_available;
    const option = r => {
      const state = (STATUS[r.status?.status] || ['—'])[0];
      const iteration = r.status?.iteration;
      return `<option value="${escapeHTML(r.name)}">${escapeHTML(runName(r))}`
        + `${iteration ? ' · ' + iteration + ' 迭代' : ''} · ${escapeHTML(state)}</option>`;
    };
    const archived = overview.runs.filter(r => r.archived);
    const markup = overview.runs.filter(r => !r.archived).map(option).join('')
      + (archived.length ? `<optgroup label="已归档">${archived.map(option).join('')}</optgroup>` : '');
    // Rewriting the list would drop the open panel's focus on every poll.
    if (markup !== pickerMarkup) { pickerMarkup = markup; $('run-picker').innerHTML = markup; }
    if (overview.runs.some(r => r.name === selected)) $('run-picker').value = selected;
    renderAgentOptions();
    $('start-training').disabled = overview.managed_training || !setupValid;
    renderProfileOptions();
    $('stop-training').disabled = !overview.managed_training;
    renderTraining();
    const run = overview.runs.find(r => r.name === $('run-picker').value) || overview.runs[0];
    if (run) refreshLog(run);
    $('report-list').innerHTML = overview.reports.map(r => {
      const invalid = r.valid === false;const wr = r.win_rate??r.win_rate_a;
      return `<div class="report-row"><div class="name">${escapeHTML(agentLabel(r.agent_a)||r.name)}<small>对手 ${escapeHTML(agentLabel(r.agent_b))}</small></div><div class="rate">${invalid?'无效':percent(wr)}<small>队伍胜率</small></div><div class="value">${number(r.games??r.num_games)} 局<small>${r.ci95?'同牌换队':'上游独立引擎'}</small></div><div class="value">${r.ci95?percent(r.ci95[0])+' – '+percent(r.ci95[1]):'无本地配对区间'}<small>${r.ci95?'95% 置信区间':'不能直接与本地训练成绩比较'}</small></div></div>`;
    }).join('')||'<p class="empty">暂无评测报告。运行下方命令生成。</p>';
  } catch (error) { $('runtime').textContent = '连接中断'; }
}

/* ---------- 模型节点：名称、强度与清理 ---------- */
let pickerMarkup = '';
const runOf = name => (overview?.runs || []).find(r => r.name === name);
const runName = run => typeof run === 'string' ? (runOf(run)?.label || run) : (run.label || run.name);
const megabytes = bytes => typeof bytes === 'number' ? (bytes/2**20).toFixed(1) + ' MB' : '—';
function stamp(epoch) {
  if (!epoch) return '—';
  const d = new Date(epoch*1000), pad = n => String(n).padStart(2, '0');
  return `${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
function nodeLabel(node) {
  // A name that says which model this is, who it was measured against, and how it scored.
  const parts = [node.label];
  if (node.kind !== 'pool' && node.iteration) parts.push(`迭代 ${node.iteration}`);
  if (node.score) {
    const against = `对${seriesOf(node.score.opponent).label} ${percent(node.score.win_rate)}`;
    parts.push(node.score.iteration === node.iteration ? against : `${against}（迭代 ${node.score.iteration} 时测得）`);
  }
  return parts.join(' · ');
}
function renderNodes(run) {
  const nodes = run.checkpoints || [];
  if (!nodes.length) { $('nodes').innerHTML = '<p class="empty">该记录还没有保存检查点。</p>'; return; }
  $('nodes').innerHTML = '<table class="node-table"><thead><tr><th>节点</th><th>迭代</th><th>开发集胜率</th>'
    + '<th>95% 区间</th><th>对手</th><th>文件</th><th>生成时间</th><th></th></tr></thead><tbody>'
    + nodes.map(node => {
        const score = node.score, exact = score && score.iteration === node.iteration;
        return `<tr><td><b>${escapeHTML(node.label)}</b>`
          + (node.kind === 'pool' ? '' : `<span class="kind ${node.kind}">${node.kind === 'latest' ? '可续训' : '晋级'}</span>`)
          + `</td><td>${node.iteration ?? '—'}</td>`
          + `<td>${score ? percent(score.win_rate) + (exact ? '' : `<small>迭代 ${score.iteration} 时测得</small>`) : '—'}</td>`
          + `<td>${score?.ci95 ? percent(score.ci95[0]) + '–' + percent(score.ci95[1]) : '—'}</td>`
          + `<td>${score ? escapeHTML(seriesOf(score.opponent).label) : '—'}</td>`
          + `<td>${megabytes(node.size)}</td><td>${stamp(node.created_at)}</td>`
          + `<td class="node-actions"><button data-play="${escapeHTML(node.path)}">试玩</button>`
          + `<button data-bench="${escapeHTML(node.path)}">评测</button>`
          + `<button data-export="${escapeHTML(node.path)}">导出</button>`
          + `<button data-remove="${escapeHTML(node.path)}">删除</button></td></tr>`;
      }).join('') + '</tbody></table>';
}
$('nodes').addEventListener('click', async event => {
  const play = event.target.dataset?.play, bench = event.target.dataset?.bench;
  const exported = event.target.dataset?.export;
  const pick = (id, value) => { if ([...$(id).options].some(o => o.value === value)) $(id).value = value; };
  if (play) {
    pick('agent', play);
    document.querySelector('[data-tab="play"]').click();
    toast('已选择该模型，点击「开始一局」发牌');
  } else if (bench) {
    pick('eval-a', bench);
    document.querySelector('[data-tab="eval"]').click();
  } else if (exported) {
    event.target.disabled = true;
    try { await exportNode($('run-picker').value, exported); }
    finally { event.target.disabled = false; }
  } else if (event.target.dataset?.remove) {
    event.target.disabled = true;
    try { await askNodeDelete($('run-picker').value, event.target.dataset.remove); }
    finally { event.target.disabled = false; }
  }
});

/* ---------- 单个模型节点的删除 ---------- */
// What the run loses differs per node, so the prompt names the actual cost.
const NODE_COST = {
  latest: '这是续训用的检查点，删除后该记录无法再继续训练；曲线和评测报告会保留',
  champion: '这是晋级赛保留的最佳模型，删除后下一次评测会重新建立冠军',
};
async function askNodeDelete(run, path) {
  const node = (runOf(run)?.checkpoints || []).find(item => item.path === path);
  if (!node) return;
  let cost = NODE_COST[node.kind] || '历史快照，删除后续训会少一个历史对手';
  if (!NODE_COST[node.kind]) {
    // Only the checkpoint knows the live opponent pool; ask instead of guessing.
    try {
      const kept = (await api(`/api/run/${encodeURIComponent(run)}/prune`)).kept
        .find(item => item.path === path);
      cost = kept ? `${kept.reason}；删除后续训会自动跳过它` : '已不在对手池中的历史快照';
    } catch { /* 保留通用提示 */ }
  }
  $('node-confirm').hidden = false;
  $('node-confirm').innerHTML = `<b>删除「${escapeHTML(node.label)}」？</b>`
    + `<span>${escapeHTML(`${megabytes(node.size)} · ${cost}。文件删除后无法恢复。`)}</span>`
    + `<span class="action-spacer"></span><button data-node="cancel">取消</button>`
    + `<button class="danger" data-node="${escapeHTML(path)}">确认删除</button>`;
}
$('node-confirm').addEventListener('click', async event => {
  const target = event.target.closest('[data-node]')?.dataset.node;
  if (!target) return;
  $('node-confirm').hidden = true;
  if (target === 'cancel') return;
  try {
    const run = $('run-picker').value;
    const data = await api(`/api/run/${encodeURIComponent(run)}/node?path=${encodeURIComponent(target)}`,
                           {}, 'DELETE');
    toast(`已删除「${data.label}」，释放 ${megabytes(data.freed)}；该记录还有 ${data.remaining} 个模型节点`);
    agentSignature = '';
    await refresh();
  } catch (error) { toast(error.message); }
});
$('run-alias').addEventListener('keydown', event => {
  if (event.key === 'Enter') event.target.blur();
  if (event.key === 'Escape') { event.target.value = runOf($('run-picker').value)?.label || ''; event.target.blur(); }
});
$('run-alias').addEventListener('blur', async () => {
  const name = $('run-picker').value, label = $('run-alias').value.trim();
  if (!name || label === (runOf(name)?.label || '')) return;
  try { await api(`/api/run/${encodeURIComponent(name)}/label`, {label}, 'PUT'); agentSignature = ''; await refresh(); }
  catch (error) { toast(error.message); }
});
$('prune-preview').onclick = async () => {
  const name = $('run-picker').value;
  try {
    const plan = await api(`/api/run/${encodeURIComponent(name)}/prune?keep_pool=${$('keep-pool').checked}`);
    $('prune-confirm').hidden = !plan.removed.length;
    $('prune-note').textContent = plan.removed.length
      ? `将删除 ${plan.removed.length} 个快照、释放 ${megabytes(plan.freed)}；保留 ${plan.kept.length} 个`
        + (plan.best ? `（含对${seriesOf(plan.best.opponent).label}胜率最高 ${percent(plan.best.win_rate)} 的迭代 ${plan.best.iteration}）` : '')
      : '没有可清理的快照。';
  } catch (error) { toast(error.message); }
};
$('prune-confirm').onclick = async () => {
  const name = $('run-picker').value;
  try {
    const plan = await api(`/api/run/${encodeURIComponent(name)}/prune?keep_pool=${$('keep-pool').checked}`, {});
    toast(`已删除 ${plan.removed.length} 个快照，释放 ${megabytes(plan.freed)}`);
    $('prune-confirm').hidden = true;
    $('prune-note').textContent = '训练时会自动清理不再使用的快照，只保留对手池和开发集对团队规则胜率最高的一个。';
    agentSignature = '';
    await refresh();
  } catch (error) { toast(error.message); }
};
$('keep-pool').addEventListener('change', () => {
  $('keep-pool').closest('.pick').classList.toggle('on', $('keep-pool').checked);
  $('prune-confirm').hidden = true;
});

/* ---------- 续训：从记录直接接着训练 ---------- */
// Says why, not just that it is unavailable: every refusal here is recoverable.
function resumeBlocker(run) {
  if (!run) return '还没有训练记录';
  if (!run.checkpoint) return '该记录没有 latest.pt，无法续训';
  if (overview?.managed_run === run.name) return '该记录正在训练中';
  if (overview?.managed_training) return `控制台正在训练「${runName(overview.managed_run)}」，同一时间只能跑一个`;
  if (run.lock?.alive) return `进程 ${run.lock.pid} 仍持有训练锁`;
  if (run.lock) return '该记录有残留训练锁，先在「管理记录」里清理';
  return null;
}
function renderResume(run) {
  const blocked = resumeBlocker(run);
  $('resume-run').disabled = Boolean(blocked);
  $('resume-run').title = blocked || `从迭代 ${run.status?.iteration ?? 0} 接着训练，参数沿用该记录的配置`;
}
$('resume-run').onclick = () => {
  const run = runOf($('run-picker').value), blocked = resumeBlocker(run);
  if (blocked) return toast(blocked);
  // Point the form at this record first: the request carries its own saved settings.
  $('profile').value = 'resume:' + run.name;
  renderSetup();
  schedulePreview();
  const cfg = run.training || {};
  const facts = [`从迭代 ${run.status?.iteration ?? 0} 接着跑`, `再训练 ${cfg.iterations ?? '—'} 次迭代`,
                 `最多 ${$('max-minutes').value} 分钟`, (cfg.device || 'auto').toUpperCase()].join(' · ');
  $('run-confirm').hidden = false;
  $('run-confirm').className = 'run-confirm go';
  $('run-confirm').innerHTML = `<b>继续训练「${escapeHTML(runName(run))}」？</b>`
    + `<span>${escapeHTML(facts)}。参数已切换为该记录保存的配置，种子、网络规模、学习率等九项锁定不可改；`
    + `要调整时长或迭代数，先点「取消」再到上方「训练参数」里改。</span>`
    + `<span class="action-spacer"></span><button data-confirm="cancel">取消</button>`
    + `<button class="primary" data-resume="${escapeHTML(run.name)}">开始续训</button>`;
};

/* ---------- 记录管理：归档、清理训练锁、导出、删除 ---------- */
async function exportNode(run, path) {
  try {
    const data = await api(`/api/run/${encodeURIComponent(run)}/export`, {path});
    toast(`已导出 ${data.path} · ${megabytes(data.size)}（仅权重，可复制到其他机器）`);
  } catch (error) { toast(error.message); }
}
function manageActions(run) {
  const lock = run.lock, latest = (run.checkpoints || []).find(node => node.kind === 'latest');
  return [
    {key: 'archive', label: run.archived ? '取消归档' : '归档记录',
     hint: run.archived ? '移回工作列表' : '收进下拉框底部，磁盘文件保持原样'},
    {key: 'export', label: '导出最新模型', disabled: !latest,
     hint: latest ? '仅权重的 npz，写入 exports/' : '该记录还没有可导出的检查点'},
    ...(lock ? [{key: 'lock', label: '清理训练锁', disabled: lock.alive,
      hint: lock.alive ? `进程 ${lock.pid} 仍在运行，先停止它`
          : lock.pid ? `进程 ${lock.pid} 已退出，清理后才能续训`
                     : '锁文件没有记录进程号，可直接清理'}] : []),
    {key: 'delete', label: '删除记录…', danger: true, hint: `释放 ${megabytes(run.size)}，不可撤销`},
  ];
}
function askDelete(run) {
  $('run-confirm').className = 'run-confirm';
  const facts = [`${run.status?.iteration ?? 0} 次迭代`, `${run.snapshots || 0} 个快照`,
                 run.champion ? '含冠军模型' : '', megabytes(run.size)].filter(Boolean).join(' · ');
  $('run-confirm').hidden = false;
  $('run-confirm').innerHTML = `<b>删除「${escapeHTML(runName(run))}」？</b>`
    + `<span>${escapeHTML(facts)}。整个目录会被移除且无法恢复；需要保留模型请先「导出」。</span>`
    + `<span class="action-spacer"></span><button data-confirm="cancel">取消</button>`
    + `<button class="danger" data-confirm="${escapeHTML(run.name)}">确认删除</button>`;
}
$('manage-run').onclick = () => {
  const run = runOf($('run-picker').value);
  if (!run) return toast('还没有训练记录。');
  if (!$('run-menu').hidden) return closeLayer();
  $('run-menu').innerHTML = manageActions(run).map(item =>
    `<button role="menuitem" data-action="${item.key}"${item.danger ? ' class="danger"' : ''}`
    + `${item.disabled ? ' disabled' : ''}><b>${escapeHTML(item.label)}</b>`
    + `<small>${escapeHTML(item.hint)}</small></button>`).join('');
  showLayer($('manage-run'), $('run-menu'), () => {
    $('run-menu').hidden = true;
    $('manage-run').setAttribute('aria-expanded', 'false');
  });
  $('manage-run').setAttribute('aria-expanded', 'true');
};
$('run-menu').addEventListener('click', async event => {
  const action = event.target.closest('[data-action]')?.dataset.action, run = runOf($('run-picker').value);
  if (!action || !run) return;
  closeLayer();
  const name = encodeURIComponent(run.name);
  if (action === 'delete') return askDelete(run);
  if (action === 'export') return exportNode(run.name, run.checkpoints.find(n => n.kind === 'latest').path);
  try {
    if (action === 'archive') {
      await api(`/api/run/${name}/archive`, {archived: !run.archived}, 'PUT');
      toast(run.archived ? '已取消归档' : '已归档，可在记录下拉框底部找到');
    } else {
      const data = await api(`/api/run/${name}/lock`, {}, 'DELETE');
      toast(data.pid ? `已清理进程 ${data.pid} 留下的训练锁，现在可以续训` : '已清理遗留训练锁，现在可以续训');
    }
    await refresh();
  } catch (error) { toast(error.message); }
});
$('run-confirm').addEventListener('click', async event => {
  const resume = event.target.closest('[data-resume]')?.dataset.resume;
  const target = event.target.closest('[data-confirm]')?.dataset.confirm;
  if (!resume && !target) return;
  $('run-confirm').hidden = true;
  if (resume) return startTraining();
  if (target === 'cancel') return;
  try {
    const data = await api(`/api/run/${encodeURIComponent(target)}`, {}, 'DELETE');
    toast(`已删除「${target}」，释放 ${megabytes(data.freed)}`);
    $('run-picker').value = '';
    agentSignature = '';
    await refresh();
  } catch (error) { toast(error.message); }
});

/* ---------- 模型节点选择与即时评测 ---------- */
const BASELINES = [['team-rule','团队规则基线'], ['program:njupt','竞赛程序 · 南邮'],
                   ['program:egg-pancake','竞赛程序 · 蛋饼'], ['random','随机基线']];
let agentSignature = '';
// An older console lacks the log / evaluation endpoints: say so once, stop polling.
const capability = {log: true, evaluate: true};

function agentOptionsHTML() {
  let html = '<optgroup label="基线对手">' + BASELINES.map(([value, label]) =>
    `<option value="${value}">${escapeHTML(label)}</option>`).join('') + '</optgroup>';
  for (const run of overview?.runs || []) {
    if (!run.checkpoints?.length) continue;
    html += `<optgroup label="${escapeHTML(runName(run))}">` + run.checkpoints.map(node =>
      `<option value="${escapeHTML(node.path)}">${escapeHTML(nodeLabel(node))}</option>`).join('') + '</optgroup>';
  }
  return html;
}
function renderAgentOptions() {
  // Every checkpoint is playable, so the table and the evaluator share one list.
  const signature = (overview?.runs || []).map(r =>
    [r.label, r.name, (r.checkpoints || []).length, r.checkpoints?.[0]?.score?.iteration].join(':')).join('|');
  const targets = ['agent', 'eval-a', 'eval-b'];
  if (signature !== agentSignature) {
    agentSignature = signature;
    const html = agentOptionsHTML();
    for (const id of targets) {
      const previous = $(id).value;
      $(id).innerHTML = html;
      if ([...$(id).options].some(o => o.value === previous)) $(id).value = previous;
      else if (id === 'eval-a') $(id).value = firstCheckpoint() || 'team-rule';
    }
  }
  if (!$('eval-device').options.length)
    $('eval-device').innerHTML = ((options?.devices || ['cpu']).filter(d => d !== 'auto'))
      .map(d => `<option>${escapeHTML(d)}</option>`).join('');
}
const firstCheckpoint = () => (overview?.runs || []).find(r => r.checkpoints?.length)?.checkpoints[0].path;

function renderEvaluationState(data) {
  const running = data.state === 'running';
  $('start-eval').disabled = running;
  $('stop-eval').disabled = !running;
  const bar = $('eval-bar'), state = $('eval-state');
  state.classList.toggle('bad', data.state === 'failed');
  if (data.state === 'idle') { bar.style.width = '0%'; state.textContent = '未开始'; return; }
  const request = data.request || {};
  const pairLabel = `${agentLabel(request.a)} vs ${agentLabel(request.b)} · ${request.pairs} 组`;
  if (running) {
    const p = data.progress;
    bar.style.width = p ? (p.done/p.total*100).toFixed(1)+'%' : '4%';
    state.textContent = p ? `${pairLabel} · 已完成 ${p.done}/${p.total} 组 · 当前胜率 ${percent(p.win_rate)}`
                          : `${pairLabel} · 正在加载模型…`;
  } else if (data.state === 'completed' && data.result) {
    bar.style.width = '100%';
    const r = data.result;
    state.textContent = `${pairLabel} · 胜率 ${percent(r.win_rate)} · 95% ${percent(r.ci95[0])}–${percent(r.ci95[1])}`
      + ` · 平均收益 ${fixed(r.average_reward, 2)} 级 · 用时 ${clock(r.elapsed_seconds)}`;
  } else {
    bar.style.width = '100%';
    state.textContent = `${pairLabel} · 评测未完成，请查看下方日志。`;
  }
  const log = $('eval-log'), bottom = log.scrollHeight-log.scrollTop-log.clientHeight < 30;
  log.textContent = (data.lines || []).join('\n') || '（暂无输出）';
  if (bottom) log.scrollTop = log.scrollHeight;
}
function agentLabel(spec) {
  if (!spec) return '—';
  const baseline = BASELINES.find(([value]) => value === spec);
  if (baseline) return baseline[1];
  const match = /(?:^|\/)runs\/([^/]+)\/(.+)\.pt$/.exec(spec);
  if (!match) return spec;
  const [, run, file] = match;
  const node = file === 'latest' ? '最新模型' : file === 'champion' ? '当前冠军'
    : file === 'pool/initial' ? '初始模型' : `迭代 ${Number((file.match(/\d+/) || ['0'])[0])} 快照`;
  return `${runName(run)} · ${node}`;
}
async function refreshEvaluation() {
  if (!capability.evaluate) return;
  try {
    renderEvaluationState(await api('/api/evaluate'));
  } catch {
    capability.evaluate = false;
    $('eval-state').textContent = '当前控制台版本不能在网页发起评测：重启 uv run python -m guandan serve 后可用。';
    $('eval-log').textContent = '仍可使用命令行：uv run python -m guandan eval --a runs/<记录>/latest.pt --b team-rule --pairs 500';
    $('start-eval').disabled = true;
  }
}
$('bench-run').onclick = () => {
  const path = `runs/${$('run-picker').value}/latest.pt`;
  if ([...$('eval-a').options].some(o => o.value === path)) $('eval-a').value = path;
  document.querySelector('[data-tab="eval"]').click();
  $('eval-a').focus();
};
$('start-eval').onclick = async () => {
  try {
    $('start-eval').disabled = true;
    await api('/api/evaluate', {a: $('eval-a').value, b: $('eval-b').value,
      pairs: Number($('eval-pairs').value), seed: Number($('eval-seed').value), device: $('eval-device').value});
    toast('评测已开始');
    refreshEvaluation();
  } catch (error) { toast(error.message); $('start-eval').disabled = false; }
};
$('stop-eval').onclick = async () => {
  try { await api('/api/evaluate/stop', {}); toast('已终止评测'); refreshEvaluation(); }
  catch (error) { toast(error.message); }
};

/* ---------- 训练参数配置 ---------- */
// Mirrors TrainConfig; max_minutes lives in the toolbar next to 开始训练.
const FIELD_GROUPS = [
  ['采样与调度', [['games_per_iteration','每迭代对局',{step:1,min:1}], ['workers','采样进程',{step:1,min:1}],
    ['iterations','迭代上限',{step:1,min:1}], ['threads','学习线程',{step:1,min:1}],
    ['device','运行设备',{options:'devices'}], ['seed','随机种子',{step:1,min:0}],
    ['pipeline','采样学习并行',{options:['true','false']}], ['save_every','存盘间隔',{step:1,min:1}]]],
  ['优化器与网络', [['batch_size','批量大小',{step:1,min:1}], ['updates','每迭代更新',{step:1,min:1}],
    ['learning_rate','学习率',{step:.00001,min:0}], ['replay_size','回放容量',{step:1024,min:64}],
    ['ntp_weight','NTP 权重',{step:.01,min:0}], ['belief_weight','信念权重',{step:.01,min:0}],
    ['model_size','网络规模',{options:['small','full']}], ['bucket_batches','等长分桶',{options:['true','false']}]]],
  ['探索、对手池与评测', [['epsilon','ε 起点',{step:.01,min:0,max:1}], ['epsilon_final','ε 下限',{step:.01,min:0,max:1}],
    ['pool_fraction','历史模型比例',{step:.05,min:0,max:1}], ['pool_size','历史模型数量',{step:1,min:1}],
    ['pool_recent','其中最近快照',{step:1,min:1}], ['pool_archive_every','历史快照间隔',{step:25,min:1}],
    ['snapshot_every','快照间隔',{step:1,min:1}], ['eval_every','评测间隔',{step:1,min:1}],
    ['eval_pairs','评测对数',{step:1,min:1}], ['gate_threshold','晋级胜率门槛',{step:.01,min:.01,max:.99}]]],
];
let options = null, setupValid = true, previewTimer = null, profileSignature = '';

function currentTarget() {
  const raw = $('profile').value || 'new:league', cut = raw.indexOf(':');
  const kind = raw.slice(0, cut), value = raw.slice(cut+1);
  return kind === 'resume' ? {profile: 'league', resume: value} : {profile: value, resume: null};
}
function baseValues() {
  const target = currentTarget();
  if (target.resume) return {...options.defaults, ...(overview?.runs.find(r => r.name === target.resume)?.training || {})};
  return {...options.defaults, ...(options.profiles[target.profile] || {})};
}
function renderProfileOptions() {
  if (!options) return;
  const resumable = (overview?.runs || []).filter(r => r.checkpoint);
  const signature = Object.keys(options.profiles).join() + '|' + resumable.map(r => r.name).join();
  if (signature === profileSignature) return;
  profileSignature = signature;
  const previous = $('profile').value;
  $('profile').innerHTML = '<optgroup label="新建训练">' + Object.keys(options.profiles).map(name =>
      `<option value="new:${escapeHTML(name)}">${escapeHTML(name)}.toml</option>`).join('')
    + '</optgroup>' + (resumable.length ? '<optgroup label="继续已有训练">' + resumable.map(r =>
      `<option value="resume:${escapeHTML(r.name)}">继续 ${escapeHTML(r.name)}</option>`).join('') + '</optgroup>' : '');
  $('profile').value = [...$('profile').options].some(o => o.value === previous) ? previous : 'new:league';
}
function renderSetup() {
  if (!options) return;
  const base = baseValues(), target = currentTarget();
  const frozen = target.resume ? options.frozen_on_resume : [];
  $('setup-fields').innerHTML = FIELD_GROUPS.map(([title, fields]) =>
    `<div class="setup-block"><span class="eyebrow">${escapeHTML(title)}</span><div class="setup-grid">`
    + fields.map(([key, label, opt]) => {
        const locked = frozen.includes(key), value = base[key];
        const control = opt.options
          ? `<select data-field="${key}"${locked ? ' disabled' : ''}>` + (Array.isArray(opt.options) ? opt.options : options[opt.options])
              .map(o => `<option${String(o) === String(value) ? ' selected' : ''}>${escapeHTML(o)}</option>`).join('') + '</select>'
          : `<input data-field="${key}" type="number" step="${opt.step}" min="${opt.min}"`
              + (opt.max !== undefined ? ` max="${opt.max}"` : '') + ` value="${escapeHTML(String(value ?? ''))}"${locked ? ' disabled' : ''}>`;
        return `<label class="${locked ? 'locked' : ''}"${locked ? ' title="续训必须与检查点保持一致"' : ''}>`
          + `${escapeHTML(label)}${locked ? ' · 锁定' : ''}${control}</label>`;
      }).join('') + '</div></div>').join('');
  enhanceSelects($('setup-fields'));
  $('opponent-picks').innerHTML = options.opponents.map(key => {
    const on = (base.rule_opponents || []).includes(key);
    return `<label class="pick${on ? ' on' : ''}"><input type="checkbox" data-opponent="${escapeHTML(key)}"${on ? ' checked' : ''}>`
      + `${escapeHTML(seriesOf(key).label)}</label>`;
  }).join('');
  $('max-minutes').value = base.max_minutes ?? 30;
  $('run-name').parentElement.hidden = Boolean(target.resume);
  $('setup-summary').textContent = target.resume
    ? `继续 ${target.resume} · 沿用检查点配置，锁定项不可更改` : `基于 configs/${target.profile}.toml`;
}
function setupOverrides() {
  const out = {};
  $('setup-fields').querySelectorAll('[data-field]').forEach(el => {
    if (el.disabled || el.value === '') return;
    out[el.dataset.field] = el.tagName === 'SELECT' ? el.value : Number(el.value);
  });
  const opponents = [...$('opponent-picks').querySelectorAll('input:checked')].map(el => el.dataset.opponent);
  if (opponents.length) out.rule_opponents = opponents;
  if ($('max-minutes').value !== '') out.max_minutes = Number($('max-minutes').value);
  return out;
}
async function previewSetup() {
  if (!options) return;
  const target = currentTarget();
  try {
    const data = await api('/api/train/preview', {...target, overrides: setupOverrides()});
    $('setup-preview').textContent = data.toml + '\n# 等价命令\nuv run python -m guandan train'
      + ` --config runs/${target.resume || '<名称>'}/request.toml --out runs/${target.resume || '<名称>'}`
      + (target.resume ? ` --resume runs/${target.resume}/latest.pt` : '');
    $('setup-note').textContent = '参数会写入该训练目录的 request.toml，命令行可直接复用。';
    $('setup-note').classList.remove('bad');
    $('setup-fields').querySelectorAll('label.bad').forEach(el => el.classList.remove('bad'));
    setupValid = true;
  } catch (error) {
    $('setup-preview').textContent = '—';
    $('setup-note').textContent = '参数无效：' + error.message;
    $('setup-note').classList.add('bad');
    setupValid = false;
    // The server names the offending field; point at it instead of only saying so.
    $('setup-fields').querySelectorAll('[data-field]').forEach(el =>
      el.closest('label')?.classList.toggle('bad', error.message.includes(el.dataset.field)));
  }
  $('start-training').disabled = !setupValid || Boolean(overview?.managed_training);
}
const schedulePreview = () => { clearTimeout(previewTimer); previewTimer = setTimeout(previewSetup, 320); };
$('profile').onchange = () => { renderSetup(); schedulePreview(); };
$('setup-fields').addEventListener('input', schedulePreview);
$('opponent-picks').addEventListener('change', event => {
  event.target.closest('.pick').classList.toggle('on', event.target.checked);
  schedulePreview();
});
$('max-minutes').addEventListener('input', schedulePreview);
$('reset-setup').onclick = () => { renderSetup(); schedulePreview(); };
$('toggle-setup').onclick = () => { $('train-setup').open = !$('train-setup').open; };
$('train-setup').addEventListener('toggle', () =>
  $('toggle-setup').setAttribute('aria-expanded', String($('train-setup').open)));
async function loadOptions() {
  try {
    options = await api('/api/train/options');
  } catch {
    $('setup-fields').innerHTML = '<p class="empty">重启控制台（uv run python -m guandan serve）后可在此配置训练参数。</p>';
    $('setup-note').textContent = '当前控制台版本不支持网页配置训练参数。';
    return;
  }
  renderProfileOptions();
  renderSetup();
  previewSetup();
}

async function startTraining() {
  const target = currentTarget();
  try {
    $('start-training').disabled = true;
    const name = $('run-name').value.trim();
    const data = await api('/api/train', {...target, name: name || null, overrides: setupOverrides()});
    toast(`${data.resumed ? '已继续训练' : '已启动'} ${data.name} · 上限 ${data.training.max_minutes} 分钟`);
    await refresh();
    // Follow the run that was just started, whichever record it belongs to.
    $('run-picker').value = data.name;
    await refresh();
  } catch (error) { toast(error.message); $('start-training').disabled = false; }
}
$('start-training').onclick = startTraining;
$('stop-training').onclick=async()=>{try{await api('/api/train/stop',{});toast('将在当前迭代完成后保存并停止');}catch(e){toast(e.message);}};
enhanceSelects();loadOptions();refresh();refreshEvaluation();setInterval(refresh,5000);setInterval(refreshEvaluation,5000);
