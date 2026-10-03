const $ = id => document.getElementById(id);
const MODEL_NAMES = {baseline: 'Baseline CNN', residual: 'Residual CNN'};
const CATEGORY_NAMES = {TP: 'Correct seizure', TN: 'Correct non-seizure', FP: 'False positive', FN: 'Missed seizure'};
const state = {research: null, model: null, examples: [], selected: null, view: null, prediction: null, window: 0, pendingWindow: null, enabled: new Set(Array.from({length: 18}, (_, i) => i)), request: 0};
const count = value => value.toLocaleString('en-US');
const percentage = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'The research service is unavailable.');
  return body;
}
function status(text, kind = '') {
  $('status-text').textContent = text;
  $('status').className = `status ${kind}`;
}
function readyStatus() { status('Verified held-out example · real model inference · frozen study', 'ready'); }
function draw() {
  EEGPlot.waveform($('waveform'), state.view, state.enabled, state.prediction, state.window);
  EEGPlot.timeline($('timeline'), state.prediction, state.window, state.view?.example.seizures || []);
}
// Rounding must never make a below-threshold score appear at or above the threshold.
// The label always comes from the API; this function changes presentation only.
function formatScore(score, threshold, predictedLabel) {
  for (let precision = 6; precision <= 15; precision++) {
    const display = score.toFixed(precision);
    if ((Number(display) >= threshold) === Boolean(predictedLabel)) return display;
  }
  return String(score);
}
function resultTable(title, rows, columns) {
  const wrap = document.createElement('div');
  wrap.className = 'table-scroll'; wrap.tabIndex = 0;
  wrap.setAttribute('role', 'region'); wrap.setAttribute('aria-label', title);
  const table = document.createElement('table'), caption = document.createElement('caption');
  caption.textContent = title; table.append(caption);
  const head = document.createElement('thead'), tr = document.createElement('tr');
  for (const [, label] of columns) { const th = document.createElement('th'); th.scope = 'col'; th.textContent = label; tr.append(th); }
  head.append(tr); table.append(head);
  const body = document.createElement('tbody');
  const integerKeys = ['n_windows', 'positive_windows', 'tn', 'fp', 'fn', 'tp'];
  for (const row of rows) {
    const tr = document.createElement('tr');
    for (const [index, [key]] of columns.entries()) {
      const td = document.createElement(index === 0 ? 'th' : 'td'), value = row[key];
      if (index === 0) td.scope = 'row';
      td.textContent = value == null ? '—' : typeof value === 'number' ? (integerKeys.includes(key) ? count(value) : value.toFixed(6)) : value;
      tr.append(td);
    }
    body.append(tr);
  }
  table.append(body); wrap.append(table); return wrap;
}
function showMetrics() {
  const model = state.research.models[state.model];
  if (!model) return;
  const report = model.metrics, pooled = report.pooled;
  for (const key of ['sensitivity', 'specificity', 'auprc_ap']) $(key).textContent = percentage(pooled[key]);
  $('detected-windows').textContent = `${count(pooled.tp)} / ${count(pooled.positive_windows)}`;
  $('false-positives').textContent = count(pooled.fp);
  $('heldout-individuals').textContent = count(report.patients.length);
  $('heldout-hours').textContent = pooled.duration_hours.toFixed(1);
  $('test-windows').textContent = count(pooled.n_windows);
  $('summary-model').textContent = `· ${MODEL_NAMES[state.model]}`;
  $('model-context').textContent = `${state.model === state.research.default_model ? 'Validation-selected default. ' : 'Same frozen data protocol. ' }Threshold ${model.threshold.toFixed(3)}; fixed before testing.`;
  $('metrics-caption').textContent = `Pooled window-level results · ${count(pooled.positive_windows)} seizure-positive / ${count(pooled.negative_windows)} non-seizure windows · all eligible test windows retained.`;
  $('result-details').hidden = false;
  $('table-model').textContent = `· ${MODEL_NAMES[state.model]}`;
  $('result-context').textContent = `${count(pooled.n_windows)} windows · ${count(pooled.positive_windows)} positive · ${(pooled.prevalence * 100).toFixed(4)}% prevalence · ${pooled.duration_hours.toFixed(2)} hours. Patient macro gives equal weight to each individual; chb01 + chb21 are grouped. AP means average precision. Undefined metrics appear as —. Values below are fractions, not percentages. Baseline remains the validation-selected default.`;
  const metrics = [['accuracy', 'Accuracy'], ['sensitivity', 'Sensitivity'], ['specificity', 'Specificity'], ['precision', 'Precision'], ['f1', 'F1'], ['auroc', 'AUROC'], ['auprc_ap', 'AP']];
  const counts = [['n_windows', 'Windows'], ['positive_windows', 'Positive'], ['tn', 'TN'], ['fp', 'FP'], ['fn', 'FN'], ['tp', 'TP']];
  $('result-tables').replaceChildren(
    resultTable('Pooled and patient-macro results', [{scope: 'Pooled', ...pooled}, {scope: 'Patient macro', ...report.patient_macro}], [['scope', 'Scope'], ...metrics]),
    resultTable('Canonical individuals', report.patients, [['individual_id', 'Individual'], ...counts, ...metrics]),
    resultTable('Case breakdown', report.cases, [['case_id', 'Case'], ...counts, ...metrics])
  );
}
function curatedCategory(example) {
  return example.curations?.[state.model] || (example.curation_model === state.model ? example.curation_category : null);
}
function eligibleExamples() {
  return state.examples.filter(e => curatedCategory(e) && ($('category').value === 'all' || curatedCategory(e) === $('category').value));
}
function renderExamples() {
  const restoreFocus = document.activeElement?.closest('.example-button')?.dataset.exampleId;
  $('examples').replaceChildren();
  const selected = eligibleExamples();
  $('example-count').textContent = String(selected.length);
  for (const example of selected) {
    const category = curatedCategory(example), active = state.selected === example.id;
    const button = document.createElement('button');
    button.className = `example-button${active ? ' active' : ''}${category.startsWith('F') ? ' failure' : ''}`;
    button.dataset.exampleId = example.id;
    button.setAttribute('aria-pressed', String(active));
    const title = document.createElement('strong'); title.textContent = CATEGORY_NAMES[category] + ' ';
    const code = document.createElement('span'); code.className = 'outcome-code'; code.textContent = category; title.append(code);
    const file = document.createElement('span'); file.className = 'example-file'; file.textContent = example.recording.split('/')[1];
    const time = document.createElement('span'); time.className = 'example-time'; time.textContent = `${example.focus_start_seconds}–${example.focus_start_seconds + 8} s`;
    button.append(title, file, time); button.onclick = () => selectExample(example.id); $('examples').append(button);
    if (restoreFocus === example.id) button.focus({preventScroll: true});
  }
  if (!selected.length) {
    const p = document.createElement('p'); p.className = 'empty-list'; p.textContent = 'No verified examples in this category.'; $('examples').append(p);
  }
}
function clearReadout() {
  state.view = null; state.prediction = null; state.pendingWindow = null;
  for (const id of ['prediction', 'truth', 'score', 'threshold', 'window-range', 'view-range', 'recording', 'individual', 'case', 'source-hash']) $(id).textContent = '—';
  $('score').removeAttribute('title'); $('score').removeAttribute('aria-label');
  $('window-outcome').textContent = ''; $('window-outcome').className = 'window-outcome';
  $('score-comparison').textContent = 'Classification uses the full-precision score.';
  $('source-link').hidden = true; $('patient-context').textContent = 'Source details appear with the selected example.';
  $('previous').disabled = true; $('next').disabled = true;
  $('timeline').setAttribute('aria-disabled', 'true'); $('timeline').setAttribute('aria-valuetext', 'No example loaded');
  $('timeline').setAttribute('aria-valuenow', '1'); $('timeline').setAttribute('aria-valuemax', '1');
  $('waveform-empty').hidden = false;
  $('empty-title').textContent = 'Loading EEG'; $('empty-description').textContent = 'Loading the real signal and running the selected frozen model.';
  draw();
}
async function selectExample(id) {
  const token = ++state.request; state.selected = id; clearReadout(); renderExamples();
  $('workspace').setAttribute('aria-busy', 'true'); status('Loading EEG and running the selected model…');
  try {
    const [view, prediction] = await Promise.all([
      api(`/api/examples/${encodeURIComponent(id)}`),
      api('/api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({example_id: id, model_id: state.model})})
    ]);
    if (token !== state.request) return;
    state.view = view; state.prediction = prediction;
    state.window = Math.max(0, prediction.windows.findIndex(w => w.start_seconds === view.example.focus_start_seconds));
    $('waveform-empty').hidden = true;
    for (const [id, key] of [['recording', 'recording'], ['individual', 'individual_id'], ['case', 'case_id'], ['source-hash', 'source_sha256']]) $(id).textContent = view.example[key];
    $('source-link').href = view.example.source_url; $('source-link').hidden = false;
    $('patient-context').textContent = 'Held-out test partition. This individual was not used for training or validation. Clip curated after final evaluation.';
    updateWindow(); readyStatus();
  } catch (error) {
    if (token === state.request) {
      $('empty-title').textContent = 'Example unavailable'; $('empty-description').textContent = 'Select an example to retry. No signal or scores have been substituted.';
      status(error.message, 'error');
    }
  } finally { if (token === state.request) $('workspace').setAttribute('aria-busy', 'false'); }
}
function updateWindow() {
  const w = state.prediction?.windows[state.window]; if (!w) return;
  const threshold = state.prediction.threshold, score = formatScore(w.score, threshold, w.predicted_label);
  const category = (w.predicted_label === w.true_label ? 'T' : 'F') + (w.predicted_label ? 'P' : 'N');
  $('truth').textContent = w.true_label ? 'Seizure' : 'Non-seizure';
  $('prediction').textContent = w.predicted_label ? 'Seizure-positive' : 'Non-seizure';
  $('score').textContent = score; $('score').title = `Full-precision uncalibrated score: ${w.score}`;
  $('score').setAttribute('aria-label', `Uncalibrated seizure score ${score}`);
  $('threshold').textContent = threshold.toFixed(3);
  $('window-range').textContent = `${w.start_seconds}–${w.end_seconds} s`;
  $('window-outcome').textContent = `${CATEGORY_NAMES[category]} (${category})`;
  $('window-outcome').className = `window-outcome${category.startsWith('F') ? ' failure' : ''}`;
  $('score-comparison').textContent = `Score ${score} ${w.predicted_label ? '≥' : '<'} ${threshold.toFixed(3)} · ${w.predicted_label ? 'at or above' : 'below'} the frozen threshold. Display precision increases near the threshold.`;
  const end = state.view.view_start_seconds + state.view.signal_uv[0].length / state.view.sampling_rate;
  $('view-range').textContent = `${state.view.view_start_seconds.toFixed(1)}–${end.toFixed(1)} s · recording time`;
  $('previous').disabled = state.window === 0; $('next').disabled = state.window === state.prediction.windows.length - 1;
  $('timeline').setAttribute('aria-disabled', 'false');
  $('timeline').setAttribute('aria-valuemax', String(state.prediction.windows.length));
  $('timeline').setAttribute('aria-valuenow', String(state.window + 1));
  $('timeline').setAttribute('aria-valuetext', `Window ${state.window + 1} of ${state.prediction.windows.length}, ${w.start_seconds} to ${w.end_seconds} seconds. Score ${score}, threshold ${threshold}. ${CATEGORY_NAMES[category]}.`);
  draw();
}
async function navigateWindow(index) {
  if (!state.prediction) return;
  const selected = Math.max(0, Math.min(index, state.prediction.windows.length - 1));
  const token = ++state.request; state.pendingWindow = selected;
  status('Loading selected window…');
  try {
    const view = await api(`/api/examples/${encodeURIComponent(state.selected)}?start_seconds=${state.prediction.windows[selected].start_seconds - 4}`);
    if (token !== state.request) return;
    state.view = view; state.window = selected; state.pendingWindow = null; updateWindow(); readyStatus();
  } catch (error) { if (token === state.request) { state.pendingWindow = null; status(error.message, 'error'); } }
}
$('previous').onclick = () => navigateWindow((state.pendingWindow ?? state.window) - 1);
$('next').onclick = () => navigateWindow((state.pendingWindow ?? state.window) + 1);
$('timeline').onclick = event => {
  if (!state.prediction) return;
  const rect = $('timeline').getBoundingClientRect(), fraction = (event.clientX - rect.left - 32) / (rect.width - 44);
  navigateWindow(Math.floor(fraction * state.prediction.windows.length));
};
$('timeline').onkeydown = event => {
  if (!state.prediction) return;
  const index = state.pendingWindow ?? state.window;
  const keys = {ArrowLeft: index - 1, ArrowDown: index - 1, ArrowRight: index + 1, ArrowUp: index + 1, Home: 0, End: state.prediction.windows.length - 1};
  if (event.key in keys) { event.preventDefault(); navigateWindow(keys[event.key]); }
};
$('channels-toggle').onclick = () => {
  const open = $('channel-options').hidden; $('channel-options').hidden = !open;
  $('channels-toggle').setAttribute('aria-expanded', String(open));
};
function selectCurated() {
  const eligible = eligibleExamples(), selected = eligible.find(e => e.id === state.selected) || eligible[0];
  if (selected) selectExample(selected.id);
  else { ++state.request; state.selected = null; clearReadout(); renderExamples(); $('empty-title').textContent = 'No example selected'; $('empty-description').textContent = 'Choose another example type.'; status('No examples were curated in this category for this model.', 'error'); }
}
$('category').onchange = selectCurated;
for (const radio of document.querySelectorAll('input[name="model"]')) radio.onchange = () => {
  if (!radio.checked || !state.research?.models[radio.value]) return;
  state.model = radio.value; showMetrics(); selectCurated();
};
new ResizeObserver(draw).observe($('waveform'));
document.fonts.ready.then(draw);
async function boot() {
  try {
    state.research = await api('/api/research');
    for (const [i, name] of state.research.channels.entries()) {
      const label = document.createElement('label'), checkbox = document.createElement('input');
      checkbox.type = 'checkbox'; checkbox.checked = true;
      checkbox.onchange = () => { checkbox.checked ? state.enabled.add(i) : state.enabled.delete(i); $('channels-toggle').querySelector('span').textContent = state.enabled.size; draw(); };
      label.append(checkbox, document.createTextNode(name)); $('channel-options').append(label);
    }
    if (state.research.status !== 'ready') {
      $('empty-title').textContent = 'Verified artifacts unavailable'; $('empty-description').textContent = 'Real EEG and inference require the verified study bundle. No research metrics or scores are substituted.';
      $('examples').textContent = 'No verified examples loaded.';
      status('Verified study artifacts are unavailable. Research metrics and inference cannot be displayed.', 'error'); draw(); return;
    }
    state.model = state.research.default_model;
    for (const radio of document.querySelectorAll('input[name="model"]')) { radio.disabled = !state.research.models[radio.value]; radio.checked = radio.value === state.model; }
    $('category').disabled = false;
    state.examples = (await api('/api/examples')).examples;
    showMetrics(); selectCurated();
  } catch (error) {
    $('empty-title').textContent = 'Research service unavailable'; $('empty-description').textContent = 'Reload the page to retry. Real artifacts are required for signal and inference.';
    status(error.message, 'error');
  }
}
boot();
