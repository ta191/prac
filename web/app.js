'use strict';
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = {voices: [], jobs: [], status: {}, latest: {vc: null, tts: null}, editing: null, preview: null, signature: '', polling: false, busy: new Set()};
const labels = {queued:'順番待ち', running:'処理中', done:'完成', failed:'エラー', cancelled:'キャンセル済み'};
const titles = {convert:'音声変換', tts:'テキスト読み上げ', voices:'マイボイス', history:'生成履歴'};
const token = $('meta[name="voiceai-token"]').content;
let toastTimer;
function toast(message) { const el=$('#toast'); el.textContent=message; el.hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>el.hidden=true, 6500); }
async function api(url, options={}) {
  const response = await fetch(url, {...options, headers:{'X-VoiceAI-Token':token, ...options.headers}});
  let value;
  try {value=await response.json();} catch {throw Error('応答を確認できません。起動用ウィンドウを確認してください。');}
  if (!response.ok) throw Error(typeof value.detail === 'string' ? value.detail : '入力内容を確認してください。');
  return value;
}
function tab(name) { $$('.panel').forEach(p=>p.hidden=p.id!==`panel-${name}`); $$('.nav-item').forEach(b=>b.classList.toggle('active',b.dataset.tab===name)); $('#crumb-title').textContent=titles[name]; }
$$('.nav-item').forEach(b=>b.addEventListener('click',()=>tab(b.dataset.tab)));
const wave = () => `<div class="waveform" aria-hidden="true">${Array.from({length:19},()=>'<i></i>').join('')}</div>`;
// Use attributes to avoid inline event handlers; waveform heights are fixed visual decoration.
function emptyResult() {return `<div class="empty-result">${wave()}<h3>ここから、新しい声が生まれます。</h3><p>生成が完了すると、ここで試聴できます。<br>WAV・MP3で保存できます。</p></div>`;}
function controls(job) {
  if (job.status==='done') return `<audio controls preload="none" src="/api/jobs/${job.id}/file/wav" aria-label="${esc(job.voice_name)}の生成音声"></audio><div class="downloads"><a class="button secondary" href="/api/jobs/${job.id}/file/wav" download>↓ WAV</a><a class="button secondary" href="/api/jobs/${job.id}/file/mp3" download>↓ MP3</a></div>`;
  if (['queued','running'].includes(job.status)) return `<div class="job-progress" aria-label="処理が完了するまでお待ちください"></div><button class="text-button" data-cancel="${job.id}">キャンセル</button>`;
  if(job.status==='failed') return `<a class="text-button" href="/api/jobs/${job.id}/log" target="_blank" rel="noopener">ログを確認</a>`;
  return '';
}
function renderResult(kind) {
  const job=state.jobs.find(j=>j.id===state.latest[kind]) || state.jobs.find(j=>j.kind===kind);
  const target=$(`#${kind==='vc'?'convert':'tts'}-result`);
  if(!job) {target.innerHTML=emptyResult(); return;}
  let detail='';
  if(job.seconds!=null) detail=`<p class="small muted">処理時間 ${Math.round(job.seconds)}秒</p>`;
  target.innerHTML=`<span class="status-label ${job.status}">${labels[job.status]||esc(job.status)}</span><h3 class="job-title">${esc(job.voice_name)}</h3><div class="job-message">${esc(job.message)}</div>${controls(job)}${detail}`;
}
function updateButtons() {
  $('#convert-button').disabled=!state.status.vc || !state.voices.length || state.busy.has('vc');
  $('#tts-button').disabled=!state.status.tts || !state.voices.length || state.busy.has('tts');
}
async function refreshVoices(selected) {
  state.voices=await api('/api/voices'); $('#voice-count').textContent=state.voices.length;
  $$('.voice-select').forEach(select=>{const prior=selected||select.value; select.innerHTML=state.voices.length ? state.voices.map(v=>`<option value="${v.id}">${esc(v.name)}</option>`).join('') : '<option value="">先にボイスを登録してください</option>'; if(state.voices.some(v=>v.id===prior)) select.value=prior;});
  $('#voice-library').innerHTML=state.voices.length ? state.voices.map(v=>`<article class="card voice-card"><div class="voice-avatar" aria-hidden="true">◉</div><h3>${esc(v.name)}</h3><div class="meta">参照音声 ${v.duration.toFixed(1)}秒 · 自動解析対応</div><audio controls preload="none" src="/api/voices/${v.id}/audio" aria-label="${esc(v.name)}の参照音声"></audio><div class="voice-actions"><button class="text-button" data-use="${v.id}">この声で変換 →</button><button class="text-button" data-edit="${v.id}">編集</button><button class="text-button danger" data-delete="${v.id}">削除</button></div></article>`).join('') : '<div class="card empty-library"><div class="eyebrow">YOUR FIRST VOICE</div><h3>はじめての声を登録しましょう。</h3><p>短い音声サンプルから、あなただけのボイスライブラリを。<br>登録した声は、音声変換と読み上げの両方で使えます。</p><button class="button primary add-voice">＋ ボイスを登録</button></div>';
  updateButtons();
}
function renderHistory() {
  $('#job-history').innerHTML=state.jobs.length ? state.jobs.map(j=>`<article class="card history-row"><div><span class="status-label ${j.status}">${labels[j.status]||esc(j.status)}</span><h3>${j.kind==='tts'?'読み上げ':'音声変換'} · ${esc(j.voice_name)}</h3><div class="history-date">${esc(new Date(j.created).toLocaleString('ja-JP'))}${j.seconds!=null?' · '+Math.round(j.seconds)+'秒':''}</div><p class="job-message">${esc(j.message)}</p>${j.text?`<p class="small muted">${esc(j.text.slice(0,90))}${j.text.length>90?'…':''}</p>`:''}</div><div class="history-details">${controls(j)}</div></article>`).join('') : '<div class="card empty-library"><h3>制作の記録が、ここに並びます。</h3><p>音声を生成すると、試聴・保存できます。</p></div>';
}
async function refreshJobs(force=false) {
  if(state.polling) return;
  state.polling=true;
  try {const jobs=await api('/api/jobs');const signature=JSON.stringify(jobs);if(force||signature!==state.signature){state.signature=signature;state.jobs=jobs;renderResult('vc');renderResult('tts');renderHistory();}}
  finally {state.polling=false;}
}
function openVoice(id=null) {
  state.editing=id;const voice=state.voices.find(v=>v.id===id);$('#voice-form').reset();$('#voice-error').textContent='';
  $('#dialog-title').textContent=id?'ボイスを編集':'新しいボイスを登録';$('#save-voice').textContent=id?'変更を保存':'ボイスを登録';
  $('#reference-fields').hidden=!!id;$('#reference-file').required=!id;$('#consent-label').hidden=!!id;$('#voice-consent').required=!id;
  if(voice){$('#voice-name').value=voice.name;}
  $('#voice-dialog').showModal();$('#voice-name').focus();
}
$('#close-dialog').addEventListener('click',()=>$('#voice-dialog').close());
document.addEventListener('click',async event=>{
  const el=event.target.closest('button');if(!el) return;
  try {
    if(el.classList.contains('add-voice')) openVoice();
    if(el.dataset.edit) openVoice(el.dataset.edit);
    if(el.dataset.use){$('#convert-voice').value=el.dataset.use;tab('convert');}
    if(el.dataset.delete){const voice=state.voices.find(v=>v.id===el.dataset.delete);if(confirm(`「${voice.name}」の登録と参照音声を削除しますか？生成済みの音声は履歴に残ります。`)){await api(`/api/voices/${voice.id}`,{method:'DELETE'});await refreshVoices();toast('ボイスを削除しました。');}}
    if(el.dataset.cancel){await api(`/api/jobs/${el.dataset.cancel}/cancel`,{method:'POST'});await refreshJobs(true);}
  } catch(error){toast(error.message);}
});
$('#voice-form').addEventListener('submit',async e=>{
  e.preventDefault();const btn=$('#save-voice');btn.disabled=true;$('#close-dialog').disabled=true;$('#voice-error').textContent='';
  try {let voice;
    if(state.editing) voice=await api(`/api/voices/${state.editing}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:$('#voice-name').value})});
    else {const data=new FormData(e.target);data.set('consent',String($('#voice-consent').checked));voice=await api('/api/voices',{method:'POST',body:data});}
    await refreshVoices(voice.id);$('#voice-dialog').close();toast(state.editing?'ボイスを更新しました。':'ボイスを登録しました。変換と読み上げの両方で使えます。');
  }catch(error){$('#voice-error').textContent=error.message;}finally{btn.disabled=false;$('#close-dialog').disabled=false;}
});
function sourceChanged(){const file=$('#source-file').files[0];$('#source-name').textContent=file?file.name:'音声をドロップ、または選択';if(state.preview)URL.revokeObjectURL(state.preview);$('#source-preview').hidden=!file;if(file){state.preview=URL.createObjectURL(file);$('#source-preview').src=state.preview;}}
$('#source-file').addEventListener('change',sourceChanged);
['dragenter','dragover'].forEach(name=>$('#source-drop').addEventListener(name,e=>{e.preventDefault();$('#source-drop').classList.add('dragover');}));
['dragleave','drop'].forEach(name=>$('#source-drop').addEventListener(name,e=>{e.preventDefault();$('#source-drop').classList.remove('dragover');if(name==='drop'&&e.dataTransfer.files.length){const transfer=new DataTransfer();transfer.items.add(e.dataTransfer.files[0]);$('#source-file').files=transfer.files;sourceChanged();}}));
$('#tts-text').addEventListener('input',()=>$('#text-count').textContent=$('#tts-text').value.length);
async function submit(kind, body) {
  state.busy.add(kind); updateButtons();
  try {const options={method:'POST',body};if(kind==='tts') options.headers={'Content-Type':'application/json'};const job=await api(`/api/jobs/${kind}`,options);state.latest[kind]=job.id;await refreshJobs(true);toast('処理を受け付けました。順番に実行します。');}
  catch(error){toast(error.message);}finally{state.busy.delete(kind);updateButtons();}
}
$('#convert-form').addEventListener('submit',e=>{e.preventDefault();const data=new FormData();data.append('file',$('#source-file').files[0]);data.append('voice_id',$('#convert-voice').value);data.append('steps',$('#steps').value);submit('vc',data);});
$('#tts-form').addEventListener('submit',e=>{e.preventDefault();submit('tts',JSON.stringify({voice_id:$('#tts-voice').value,text:$('#tts-text').value}));});
$('#refresh-history').addEventListener('click',()=>refreshJobs(true).catch(e=>toast(e.message)));
async function start(){renderResult('vc');renderResult('tts');updateButtons();try{state.status=await api('/api/status');const banner=$('#setup-banner');banner.hidden=state.status.tts&&state.status.vc&&state.status.audio;banner.textContent=state.status.message;await refreshVoices();await refreshJobs(true);setInterval(()=>refreshJobs().catch(()=>{const banner=$('#setup-banner');banner.hidden=false;banner.textContent='アプリとの接続が切れました。起動用ウィンドウを確認し、画面を更新してください。';}),2500);}catch(error){toast(error.message);}}
start();
