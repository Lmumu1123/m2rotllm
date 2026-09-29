const fs = require('fs');
const path = require('path');
const assert = require('assert');
const {JSDOM, VirtualConsole} = require('./.browser_check/node_modules/jsdom');
const html = fs.readFileSync(path.join(__dirname, 'replay/模型输入输出演示.html'), 'utf8');
const errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => errors.push(e.message));
const dom = new JSDOM(html, {runScripts:'dangerously', virtualConsole:vc, pretendToBeVisual:true});
const {window:w} = dom, d = w.document;
const data = JSON.parse(d.getElementById('payload').textContent);
assert.equal(d.querySelectorAll('#record option').length, 12);
assert.equal(d.querySelectorAll('#contact-probs .probrow').length, 4);
assert.equal(d.querySelectorAll('#radar-probs .probrow').length, 4);
assert.equal(d.querySelectorAll('#xyz path').length, 3);
assert.equal(d.querySelectorAll('#iq path').length, 2);
assert.equal(d.querySelectorAll('#hidden path').length, 2);
const select=d.getElementById('record');
d.getElementById('next').click(); assert.equal(select.value, '1');
d.getElementById('prev').click(); assert.equal(select.value, '0');
for (let i=0; i<data.items.length; i++) {
  select.value=String(i);select.dispatchEvent(new w.Event('change'));
  assert(d.getElementById('identity').textContent.includes(data.items[i].bag_id));
  assert(d.getElementById('contact-result').textContent.includes(data.items[i].contact_prediction));
  if (!data.items[i].geometry_valid) assert(d.getElementById('status').textContent.includes('距离门无效'));
  if (data.items[i].contact_correct===false) assert(d.getElementById('contact-result').textContent.includes('判错'));
}
const old=d.querySelector('#feature path').getAttribute('d');
d.getElementById('feature-mode').value='radar_standardized';
d.getElementById('feature-mode').dispatchEvent(new w.Event('change'));
assert.notEqual(d.querySelector('#feature path').getAttribute('d'),old);
d.getElementById('band').value='99';d.getElementById('band').dispatchEvent(new w.Event('input'));
assert.equal(d.getElementById('band-id').textContent,'99');
assert.equal(errors.length,0,errors.join('\n'));
const result={status:'passed',recordings:12,all_predictions_and_invalid_ROI_labels_rendered:true,
  previous_next:true,scaler_toggle_changes_plot:true,band_inspector:true,uncaught_js_errors:errors,
  scope:'DOM execution and interactions using jsdom; not a browser screenshot or full layout check'};
fs.writeFileSync(path.join(__dirname,'replay/ui_verification.json'),JSON.stringify(result,null,2));
console.log(JSON.stringify(result,null,2));
w.close();
