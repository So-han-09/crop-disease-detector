'use strict';
let currentLang = 'en', selectedFile = null;

// ── Theme ──────────────────────────────────────────────────────────────────
function toggleTheme() {
  const html = document.documentElement;
  const dark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', dark ? 'light' : 'dark');
  document.getElementById('theme-icon').textContent = dark ? '🌙' : '☀️';
  localStorage.setItem('theme', dark ? 'light' : 'dark');
}
function loadTheme() {
  const t = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', t);
  const icon = document.getElementById('theme-icon');
  if (icon) icon.textContent = t === 'dark' ? '☀️' : '🌙';
}

// ── Strings ────────────────────────────────────────────────────────────────
const EN = { title:'Crop Disease Detection System', subtitle:'AI-powered crop protection', upload_hint:'Drop your leaf image here', analyze_btn:'Analyze Plant', disease_label:'Disease', confidence_label:'Confidence', language_label:'Language', remedy_label:'RECOMMENDED REMEDY', top3_label:'TOP PREDICTIONS', reset_btn:'↩ Analyze Another', processing:'Analyzing…', error_no_file:'Please upload an image first.', healthy_status:'✅ Plant is Healthy!', disease_status:'⚠️ Disease Detected' };
const OD = { title:'ଉଦ୍ଭିଦ ରୋଗ ଚିହ୍ନଟ', subtitle:'AI ଆଧାରିତ ଫସଲ ସୁରକ୍ଷା', upload_hint:'ଏଠାରେ ପତ୍ର ଫଟୋ ଛାଡ଼ନ୍ତୁ', analyze_btn:'ବିଶ୍ଳେଷଣ କରନ୍ତୁ', disease_label:'ରୋଗ', confidence_label:'ନିଶ୍ଚିତତା', language_label:'ଭାଷା', remedy_label:'ଅନୁସୂଚିତ ଉପଚାର', top3_label:'ଶ୍ରେଷ୍ଠ ଅନୁମାନ', reset_btn:'↩ ଅନ୍ୟ ପତ୍ର', processing:'ଚାଲୁ ଅଛି…', error_no_file:'ଛବି ଅପଲୋଡ କରନ୍ତୁ।', healthy_status:'✅ ଗଛ ସୁସ୍ଥ!', disease_status:'⚠️ ରୋଗ ଚିହ୍ନଟ' };
const t = k => (currentLang === 'od' ? OD : EN)[k] || k;

// ── Language ───────────────────────────────────────────────────────────────
function setLang(lang) {
  currentLang = lang;
  document.querySelectorAll('.lang-btn').forEach(b => b.classList.toggle('active', b.dataset.lang === lang));
  const ids = { 'site-title':'title','site-subtitle':'subtitle','upload-hint':'upload_hint','btn-text':'analyze_btn','btn-reset':'reset_btn','lbl-disease':'disease_label','lbl-confidence':'confidence_label','lbl-language':'language_label','lbl-remedy':'remedy_label','lbl-top3':'top3_label' };
  for (const [id, key] of Object.entries(ids)) { const el = document.getElementById(id); if (el) el.textContent = t(key); }
  if (document.getElementById('result-card').style.display !== 'none' && selectedFile) submitPrediction();
}

// ── File Handling ──────────────────────────────────────────────────────────
function handleFileSelect(e) { const f = e.target.files[0]; if (f) loadFile(f); }
function handleDragOver(e) { e.preventDefault(); document.getElementById('upload-zone').classList.add('drag-over'); }
function handleDragLeave() { document.getElementById('upload-zone').classList.remove('drag-over'); }
function handleDrop(e) { e.preventDefault(); document.getElementById('upload-zone').classList.remove('drag-over'); const f = e.dataTransfer.files[0]; if (f) loadFile(f); }

function loadFile(file) {
  if (!['image/png','image/jpeg','image/jpg','image/bmp','image/webp'].includes(file.type)) { showToast('Unsupported format'); return; }
  if (file.size > 16*1024*1024) { showToast('File too large (max 16MB)'); return; }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('preview-img').src = e.target.result;
    document.getElementById('preview-img').style.display = 'block';
    document.getElementById('upload-placeholder').style.display = 'none';
  };
  reader.readAsDataURL(file);
  document.getElementById('btn-analyze').disabled = false;
  document.getElementById('result-card').style.display = 'none';
}

// ── Prediction ─────────────────────────────────────────────────────────────
function analyzeImage() { if (!selectedFile) { showToast(t('error_no_file')); return; } submitPrediction(); }

function submitPrediction() {
  document.getElementById('btn-text').textContent = t('processing');
  document.getElementById('btn-spinner').style.display = 'inline-block';
  document.getElementById('btn-analyze').disabled = true;
  const fd = new FormData();
  fd.append('file', selectedFile); fd.append('lang', currentLang);
  fetch('/predict', { method:'POST', body:fd })
    .then(r => r.ok ? r.json() : r.json().then(d => Promise.reject(d.error||'Error')))
    .then(renderResult)
    .catch(err => showToast(typeof err==='string' ? err : 'Prediction failed'))
    .finally(() => { document.getElementById('btn-text').textContent=t('analyze_btn'); document.getElementById('btn-spinner').style.display='none'; document.getElementById('btn-analyze').disabled=false; });
}

function renderResult(data) {
  const rc = document.getElementById('result-card');
  rc.style.display = 'flex';
  rc.scrollIntoView({ behavior:'smooth', block:'start' });

  const status = document.getElementById('result-status');
  status.classList.remove('healthy','disease');
  if (data.is_healthy) { status.classList.add('healthy'); document.getElementById('status-icon').textContent='✅'; document.getElementById('status-text').textContent=t('healthy_status'); }
  else { status.classList.add('disease'); document.getElementById('status-icon').textContent='⚠️'; document.getElementById('status-text').textContent=t('disease_status'); }

  document.getElementById('res-disease').textContent = data.disease_name||'—';
  const pct = data.confidence||0;
  document.getElementById('res-confidence').textContent = pct.toFixed(1)+'%';
  setTimeout(() => { document.getElementById('conf-bar').style.width = Math.min(pct,100)+'%'; }, 100);
  document.getElementById('res-language').textContent = data.language||'—';

  // Severity
  const sevEl = document.getElementById('severity-box');
  if (sevEl && data.severity_level) {
    const colorMap = { green:'#10b981', lime:'#4ade80', yellow:'#f59e0b', orange:'#f97316', red:'#ef4444' };
    const col = colorMap[data.severity_color] || '#818cf8';
    sevEl.style.display = 'flex';
    sevEl.style.borderColor = col;
    sevEl.style.background = col + '18';
    document.getElementById('sev-icon').textContent = data.severity_icon || '🔬';
    document.getElementById('sev-level').textContent = data.severity_level;
    document.getElementById('sev-level').style.color = col;
    document.getElementById('sev-advice').textContent = data.severity_advice || '';
  }

  const list = document.getElementById('top3-list'); list.innerHTML = '';
  (data.top3||[]).forEach((item,i) => {
    const li = document.createElement('li'); li.className='top3-item'; li.style.animationDelay=(i*0.08)+'s';
    li.innerHTML=`<span class="t3-rank">${i+1}</span><span class="t3-name">${item.name}</span><span class="t3-pct">${item.confidence.toFixed(1)}%</span>`;
    list.appendChild(li);
  });
  document.getElementById('res-remedy').textContent = data.remedy||'—';
}

function resetUI() {
  selectedFile = null;
  document.getElementById('preview-img').style.display = 'none';
  document.getElementById('preview-img').src = '';
  document.getElementById('upload-placeholder').style.display = 'flex';
  document.getElementById('btn-analyze').disabled = true;
  document.getElementById('result-card').style.display = 'none';
  document.getElementById('conf-bar').style.width = '0%';
  document.getElementById('file-input').value = '';
  window.scrollTo({ top:0, behavior:'smooth' });
}

// ── Toast ──────────────────────────────────────────────────────────────────
let _tt;
function showToast(msg) {
  const toast = document.getElementById('toast');
  toast.textContent = msg; toast.classList.add('show');
  if (_tt) clearTimeout(_tt);
  _tt = setTimeout(() => toast.classList.remove('show'), 3500);
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadTheme();
  setLang('en');
  document.addEventListener('paste', e => {
    for (const item of (e.clipboardData||e.originalEvent.clipboardData).items)
      if (item.kind==='file'&&item.type.startsWith('image/')) { loadFile(item.getAsFile()); break; }
  });
});
