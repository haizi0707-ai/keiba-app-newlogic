const REQUIRED_PRED_COLS = ["日付","開催","R","レース名","馬名","種牡馬","調教師","距離","馬場状態","前開催","前距離","間隔"];
let summaryData = null;
let rankedRows = [];
let previewBlobUrl = null;

function parseCSV(text){
  const lines = text.replace(/\r/g,'').split('\n').filter(x=>x.trim() !== '');
  if(!lines.length) return [];
  const rows = [];
  const parseLine = (line) => {
    const out=[]; let cur=''; let q=false;
    for(let i=0;i<line.length;i++){
      const ch=line[i];
      if(ch === '"'){
        if(q && line[i+1] === '"'){ cur += '"'; i++; }
        else { q=!q; }
      } else if(ch === ',' && !q){ out.push(cur); cur=''; }
      else { cur += ch; }
    }
    out.push(cur);
    return out;
  };
  const headers = parseLine(lines[0]).map(h => h.trim());
  for(let i=1;i<lines.length;i++){
    const vals = parseLine(lines[i]);
    const obj = {};
    headers.forEach((h, idx) => obj[h] = (vals[idx] ?? '').trim());
    rows.push(obj);
  }
  return rows;
}

function readFileAsText(file){
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsText(file, 'utf-8');
  });
}

function showStatus(id, text, cls='note'){
  const el = document.getElementById(id);
  el.className = cls;
  el.textContent = text;
}

function distanceBand(val){
  const s = String(val||'');
  const n = Number(s.replace(/[^0-9]/g,''));
  if(!n) return "不明";
  if(n <= 1400) return "短距離";
  if(n <= 1700) return "マイル";
  if(n <= 2000) return "中距離";
  if(n <= 2400) return "中長距離";
  return "長距離";
}

function goingGroup(g){
  return String(g||'').trim() === '良' ? '良' : '道悪';
}

function intervalCategory(v){
  const s = String(v||'').trim();
  if(!s) return "不明";
  if(s.includes('連闘')) return '連闘';
  const n = Number(s.replace(/[^0-9.-]/g,''));
  if(!Number.isNaN(n)){
    if(n <= 0) return '連闘';
    if(n <= 2) return '中1〜2週';
    if(n <= 5) return '中3〜5週';
    if(n <= 9) return '中6〜9週';
    return '10週以上';
  }
  return '不明';
}

function distanceChange(curr, prev){
  const a = Number(String(curr||'').replace(/[^0-9]/g,''));
  const b = Number(String(prev||'').replace(/[^0-9]/g,''));
  if(!a || !b) return '不明';
  if(a > b) return '延長';
  if(a < b) return '短縮';
  return '同距離';
}

function trackChange(curr, prev){
  if(!curr || !prev) return '不明';
  return String(curr).trim() === String(prev).trim() ? '同場' : '場替わり';
}

function classifyRank(rank){
  return ({S:'本線', A:'本線', B:'相手', C:'注意', D:'軽視'})[rank] || '軽視';
}

function rankFromScore(score){
  if(score >= 28) return 'S';
  if(score >= 24) return 'A';
  if(score >= 22) return 'B';
  if(score >= 20) return 'C';
  return 'D';
}

function lookupScore(mapName, key, minCount=10){
  const x = summaryData?.[mapName]?.[key];
  if(!x || x.count < minCount) return {score:null, count:null};
  return {score:(x.place_rate || 0) * 100, count:x.count};
}

function uniqueRaceKey(r){
  return [r["日付"]||'', r["開催"]||'', r["R"]||'', r["レース名"]||''].join(' | ');
}

function updateMetrics(){
  document.getElementById('mHistory').textContent = summaryData?.source_rows || 0;
  const condCount = summaryData ? [
    'sire_track','sire_dist','sire_going','trainer_track','trainer_dist','trainer_interval','trainer_distchg','trainer_trackchg'
  ].reduce((a,k)=>a + Object.keys(summaryData[k] || {}).length,0) : 0;
  document.getElementById('mConditions').textContent = condCount;
  document.getElementById('mRaces').textContent = [...new Set(rankedRows.map(uniqueRaceKey))].length;
  document.getElementById('mHorses').textContent = rankedRows.length;
}

function fillRaceSelect(rows){
  const sel = document.getElementById('raceSelect');
  sel.innerHTML = '';
  const keys = [...new Set(rows.map(uniqueRaceKey))];
  if(!keys.length){
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '先にCSVを読み込んでください';
    sel.appendChild(opt);
    return;
  }
  keys.forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = k;
    sel.appendChild(opt);
  });
}

function renderSummaryTable(rows){
  const tbody = document.querySelector('#summaryTable tbody');
  tbody.innerHTML = '';
  if(!rows.length){
    tbody.innerHTML = '<tr><td colspan="5">予想CSVを読み込むと表示されます。</td></tr>';
    return;
  }
  rows.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r["馬名"] || ''}</td>
      <td><span class="rank-pill rank-${r["信頼度"] || 'D'}">${r["信頼度"] || ''}</span></td>
      <td>${r["分類"] || ''}</td>
      <td>${r["条件"] || ''}</td>
      <td>${r["母数"] ?? ''}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderPreview(rows, title){
  const box = document.getElementById('preview');
  box.innerHTML = '';
  if(!rows.length){
    box.textContent = '予想CSVを読み込んでください';
    return;
  }
  const canvas = document.createElement('canvas');
  const rowH = 36;
  const width = 980;
  const height = 90 + rowH * (rows.length + 1);
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#081324';
  ctx.fillRect(0,0,width,height);

  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 28px sans-serif';
  ctx.fillText(title, 24, 40);

  const headers = ['馬名','縦軸点','横軸点','総合点','ランク'];
  const xs = [24, 420, 570, 720, 860];
  ctx.fillStyle = '#12284a';
  ctx.fillRect(20, 60, width-40, rowH);
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 20px sans-serif';
  headers.forEach((h,i)=>ctx.fillText(h, xs[i], 84));

  ctx.font = '18px sans-serif';
  rows.forEach((r, idx) => {
    const y = 96 + idx*rowH;
    ctx.fillStyle = idx % 2 === 0 ? '#0c1b33' : '#10203b';
    ctx.fillRect(20, y, width-40, rowH);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(String(r["馬名"]||''), xs[0], y+24);
    ctx.fillText(String(r["縦軸点"]??''), xs[1], y+24);
    ctx.fillText(String(r["横軸点"]??''), xs[2], y+24);
    ctx.fillText(String(r["総合点"]??''), xs[3], y+24);
    ctx.fillText(String(r["信頼度"]??''), xs[4], y+24);
  });

  box.appendChild(canvas);

  if(previewBlobUrl) URL.revokeObjectURL(previewBlobUrl);
  canvas.toBlob(blob => {
    previewBlobUrl = URL.createObjectURL(blob);
  });
}

async function loadSummaryJson(){
  try{
    const res = await fetch('history_summary.json', {cache:'no-store'});
    if(!res.ok) throw new Error('history_summary.json が見つかりません');
    summaryData = await res.json();
    showStatus('summaryStatus', `summary JSON を読み込みました。元件数: ${summaryData.source_rows || 0}`, 'success');
    updateMetrics();
  }catch(e){
    showStatus('summaryStatus', `summary JSON の読込に失敗しました: ${e.message}`, 'error');
  }
}

document.getElementById('loadSummaryBtn').addEventListener('click', loadSummaryJson);

document.getElementById('loadPredictionBtn').addEventListener('click', async () => {
  const file = document.getElementById('predictionFile').files[0];
  if(!file){
    showStatus('predictionStatus','予想CSVを選択してください。','error');
    return;
  }
  if(!summaryData){
    showStatus('predictionStatus','先に summary JSON を読み込んでください。','error');
    return;
  }

  const text = await readFileAsText(file);
  const predictionRows = parseCSV(text);

  const missing = REQUIRED_PRED_COLS.filter(c => !(c in (predictionRows[0] || {})));
  if(missing.length){
    showStatus('predictionStatus', `予想CSVの読み込みでエラーが出ました: 必須列不足 ${missing.join(', ')}`, 'error');
    return;
  }

  rankedRows = predictionRows.map(r => {
    const sire = r["種牡馬"] || "";
    const trainer = r["調教師"] || "";
    const track = r["開催"] || "";
    const distBand = distanceBand(r["距離"]);
    const going = goingGroup(r["馬場状態"]);
    const interval = intervalCategory(r["間隔"]);
    const distchg = distanceChange(r["距離"], r["前距離"]);
    const trchg = trackChange(r["開催"], r["前開催"]);

    const s1 = lookupScore('sire_track', `${sire}|||${track}`);
    const s2 = lookupScore('sire_dist', `${sire}|||${distBand}`);
    const s3 = lookupScore('sire_going', `${sire}|||${going}`);

    const t1 = lookupScore('trainer_track', `${trainer}|||${track}`);
    const t2 = lookupScore('trainer_dist', `${trainer}|||${distBand}`);
    const t3 = lookupScore('trainer_interval', `${trainer}|||${interval}`);
    const t4 = lookupScore('trainer_distchg', `${trainer}|||${distchg}`);
    const t5 = lookupScore('trainer_trackchg', `${trainer}|||${trchg}`);

    const vArr = [s1.score, s2.score, s3.score].filter(v => v !== null);
    const hArr = [t1.score, t2.score, t3.score, t4.score, t5.score].filter(v => v !== null);
    const vertical = vArr.length ? (vArr.reduce((a,b)=>a+b,0)/vArr.length) : 0;
    const horizontal = hArr.length ? (hArr.reduce((a,b)=>a+b,0)/hArr.length) : 0;
    const total = (vertical + horizontal) / 2;
    const rank = rankFromScore(total);
    const counts = [s1.count,s2.count,s3.count,t1.count,t2.count,t3.count,t4.count,t5.count].filter(v => v != null);
    const minCount = counts.length ? Math.min(...counts) : '';

    return {
      ...r,
      "縦軸点": Number(vertical.toFixed(1)),
      "横軸点": Number(horizontal.toFixed(1)),
      "総合点": Number(total.toFixed(1)),
      "信頼度": rank,
      "ランク": rank,
      "分類": classifyRank(rank),
      "条件": `縦:${sire}×${track}×${distBand}×${going} / 横:${trainer}×${interval}×${distchg}×${trchg}`,
      "母数": minCount
    };
  });

  fillRaceSelect(rankedRows);
  showStatus('predictionStatus', `予想データを読み込みました。件数: ${rankedRows.length}`, 'success');
  updateMetrics();
  renderSummaryTable([]);
  document.getElementById('preview').textContent = '画像を作成してください';
});

document.getElementById('renderImageBtn').addEventListener('click', () => {
  if(!rankedRows.length){
    showStatus('predictionStatus','先に予想CSVを読み込んでください。','error');
    return;
  }
  const key = document.getElementById('raceSelect').value;
  const rows = key ? rankedRows.filter(r => uniqueRaceKey(r) === key) : rankedRows;
  renderPreview(rows, key || 'レースランキング');
  renderSummaryTable(rows);
  showStatus('predictionStatus','画像を作成しました。','success');
});

document.getElementById('saveImageBtn').addEventListener('click', () => {
  if(!previewBlobUrl){
    showStatus('predictionStatus','先に画像を作成してください。','error');
    return;
  }
  const a = document.createElement('a');
  a.href = previewBlobUrl;
  a.download = 'keiba_rank_image.png';
  a.click();
});

loadSummaryJson();
