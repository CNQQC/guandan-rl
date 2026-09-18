'use strict';
const $ = id => document.getElementById(id);
let game = null, selected = new Set(), overview = null, busy = false;
let toastTimer;
const names = ['你', '右侧对手', '你的搭档', '左侧对手'];
const escapeHTML = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = x => typeof x === 'number' ? x.toLocaleString('zh-CN') : '—';
const percent = x => typeof x === 'number' ? (x * 100).toFixed(1) + '%' : '—';
function toast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 4500); }
async function api(path, payload) {
  const r = await fetch(path, payload === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data = await r.json();
  if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `请求失败 (${r.status})`);
  return data;
}
async function gameAction(fn) {
  if (busy) return;
  busy = true; updateButtons(); $('new-game').disabled = true;
  try { await fn(); } catch (error) { toast(error.message); }
  finally { busy = false; $('new-game').disabled = false; updateButtons(); }
}
document.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.nav,.tab').forEach(el => el.classList.remove('active'));
  button.classList.add('active'); $(button.dataset.tab).classList.add('active');
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
function renderChart(rows) {
  if(!rows.length){$('chart').innerHTML='<p class="empty">暂无训练数据。</p>';return;}
  const w=1000,h=190,pad=36, values=rows.map(r=>r.loss), hi=Math.max(...values)*1.1 || 1,lo=Math.min(0,...values);
  const points=rows.map((r,i)=>`${pad+i/Math.max(1,rows.length-1)*(w-pad-10)},${h-pad-(r.loss-lo)/(hi-lo)*(h-pad-12)}`).join(' ');
  const grid=[0,.5,1].map(t=>{const y=12+t*(h-pad-12);return `<line x1="${pad}" y1="${y}" x2="${w}" y2="${y}" stroke="#dde2d5"/><text x="0" y="${y+4}">${(hi-t*(hi-lo)).toFixed(2)}</text>`;}).join('');
  $('chart').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="训练损失曲线"><title>训练损失，${rows.length} 次记录</title>${grid}<polyline points="${points}" fill="none" stroke="#2e6650" stroke-width="2" vector-effect="non-scaling-stroke"/><text x="${pad}" y="${h-5}">迭代 ${rows[0].iteration}</text><text text-anchor="end" x="${w}" y="${h-5}">迭代 ${rows.at(-1).iteration}</text></svg>`;
}
function renderRun() {
  const run=overview?.runs.find(r=>r.name===$('run-picker').value);if(!run)return;
  const s=run.status;
  const labels={running:'训练中',completed:'已完成',budget_reached:'时间预算已到',stopped:'已保存并停止',failed:'训练出错'};
  $('run-status').textContent=labels[s.status]||'正在初始化';
  $('metric-games').textContent=number(s.games);$('metric-samples').textContent=number(s.samples);$('metric-updates').textContent=number(s.updates);$('metric-device').textContent=(s.device||'—').toUpperCase();
  renderChart(run.metrics);
  const profile=run.name==='smoke'?'smoke':run.name.startsWith('web-')||run.name.startsWith('league')?'league':'mac';
  $('resume-command').textContent=`uv run python -m guandan train --config configs/${profile}.toml --out runs/${run.name} --resume runs/${run.name}/latest.pt`;
}
$('run-picker').onchange=renderRun;
async function refresh() {
  try {
    overview=await api('/api/overview');
    $('runtime').textContent=`${overview.system.platform.startsWith('macOS')?'APPLE SILICON':'LOCAL'} / ${overview.system.default_device.toUpperCase()} / ${overview.system.memory_gb} GB`;
    $('expert-link').hidden=!overview.expert_available;
    const oldRun=$('run-picker').value;
    $('run-picker').innerHTML=overview.runs.map(r=>`<option>${escapeHTML(r.name)}</option>`).join('');
    if(overview.runs.some(r=>r.name===oldRun))$('run-picker').value=oldRun;
    const oldAgent=$('agent').value;
    $('agent').innerHTML='<option value="team-rule">团队规则基线</option><option value="program:njupt">竞赛程序 · 南邮</option><option value="program:egg-pancake">竞赛程序 · 蛋饼</option><option value="random">随机基线</option>'+overview.runs.filter(r=>r.checkpoint && r.status.samples>0).map(r=>`<option value="runs/${escapeHTML(r.name)}/latest.pt">自训练 · ${escapeHTML(r.name)}</option>`).join('');
    if([...$('agent').options].some(o=>o.value===oldAgent))$('agent').value=oldAgent;
    $('start-training').disabled=overview.managed_training;$('stop-training').disabled=!overview.managed_training;
    renderRun();
    $('report-list').innerHTML=overview.reports.map(r=>{
      const invalid=r.valid===false;const wr=r.win_rate??r.win_rate_a;
      return `<div class="report-row"><div class="name">${escapeHTML(r.agent_a||r.name)}<small>对手 ${escapeHTML(r.agent_b||'—')}</small></div><div class="rate">${invalid?'无效':percent(wr)}<small>队伍胜率</small></div><div class="value">${number(r.games??r.num_games)} 局<small>${r.ci95?'同牌换队':'上游独立引擎'}</small></div><div class="value">${r.ci95?percent(r.ci95[0])+' – '+percent(r.ci95[1]):'无本地配对区间'}<small>${r.ci95?'95% 置信区间':'不能直接与本地训练成绩比较'}</small></div></div>`;
    }).join('')||'<p class="empty">暂无评测报告。运行下方命令生成。</p>';
  } catch(error) { $('runtime').textContent='连接中断'; }
}
$('start-training').onclick=async()=>{try{$('start-training').disabled=true;const data=await api('/api/train',{minutes:Number($('minutes').value)});toast(`已启动 ${data.minutes} 分钟训练`);await refresh();$('run-picker').value=data.name;renderRun();}catch(e){toast(e.message);$('start-training').disabled=false;}};
$('stop-training').onclick=async()=>{try{await api('/api/train/stop',{});toast('将在当前迭代完成后保存并停止');}catch(e){toast(e.message);}};
refresh();setInterval(refresh,5000);
