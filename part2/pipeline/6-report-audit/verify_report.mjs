import { spawn } from 'node:child_process';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import assert from 'node:assert/strict';

const profile = await mkdtemp('/tmp/messytable-report-chrome-');
const browser = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new', '--remote-debugging-port=0', '--disable-background-networking',
  '--no-first-run', '--no-default-browser-check', '--user-data-dir=' + profile, 'about:blank'
], {stdio:'ignore'});
try {
  const start = Date.now();
  let port;
  while (!port && Date.now() - start < 10000) {
    try { port = (await readFile(profile + '/DevToolsActivePort','utf8')).split('\n')[0]; }
    catch { await new Promise(r => setTimeout(r,100)); }
  }
  assert(port, 'Chrome DevTools unavailable');
  const targets = await (await fetch('http://localhost:' + port + '/json/list')).json();
  const ws = new WebSocket(targets.find(x => x.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve,reject) => {ws.onopen=resolve; ws.onerror=reject;});
  const pending = new Map();
  let id = 0;
  ws.onmessage = ({data}) => {
    const event = JSON.parse(data);
    if (pending.has(event.id)) {
      pending.get(event.id)(event);
      pending.delete(event.id);
    }
  };
  async function cdp(method, params={}) {
    const key = ++id;
    const response = new Promise(r => pending.set(key,r));
    ws.send(JSON.stringify({id:key,method,params}));
    const event = await response;
    assert(!event.error, JSON.stringify(event.error));
    return event.result;
  }
  async function evaluate(expression) {
    const result = await cdp('Runtime.evaluate',{expression,returnByValue:true});
    assert(!result.exceptionDetails,JSON.stringify(result.exceptionDetails));
    return result.result.value;
  }




  const report=resolve('evaluation/messytable-part-two/report.html');
  await cdp('Emulation.setDeviceMetricsOverride',{width:1400,height:1050,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate',{url:pathToFileURL(report).href});
  for(let i=0;i<80;i++){
    if(await evaluate("document.querySelectorAll('table').length===5 && [...document.images].every(img=>img.complete&&img.naturalWidth>0)"))break;
    await new Promise(r=>setTimeout(r,100));
  }
  await evaluate("document.documentElement.style.scrollBehavior='auto'");
  assert(await evaluate("JSON.stringify([...document.querySelectorAll('table tbody')].map(t=>t.rows.length))==='[2,8,4,8,4]'"));
  assert(await evaluate("[...document.images].length===3 && [...document.images].every(img=>img.complete&&img.naturalWidth>0)"));
  assert(await evaluate("document.documentElement.scrollWidth<=innerWidth"));
  await cdp('Page.captureScreenshot',{format:'png'}).then(r=>writeFile('/tmp/part-two-rewrite-desktop.png',Buffer.from(r.data,'base64')));
  await evaluate("document.querySelectorAll('h2')[3].scrollIntoView()");
  await cdp('Page.captureScreenshot',{format:'png'}).then(r=>writeFile('/tmp/part-two-rewrite-results.png',Buffer.from(r.data,'base64')));
  await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await evaluate("window.scrollTo(0,0)");
  assert(await evaluate("document.documentElement.scrollWidth<=390"));
  await cdp('Page.captureScreenshot',{format:'png'}).then(r=>writeFile('/tmp/part-two-rewrite-mobile.png',Buffer.from(r.data,'base64')));
  const expected=JSON.parse(await readFile(resolve('research/detector-scale-and-integration/step1-results.json'),'utf8'));
  const tableText=await evaluate("document.querySelector('#detection-results').textContent");
  for(const r of Object.values(expected)){
    assert(tableText.includes((100*r.metrics.recall).toFixed(2)+'%'));
    assert(tableText.includes(r.counts.unmapped.toLocaleString('en-US')));
  }
  const evidence=JSON.parse(await readFile(resolve('evaluation/messytable-part-two/evidence.json'),'utf8'));
  const actualRows=await evaluate("[...document.querySelector('#detection-results tbody').rows].map(r=>[...r.cells].map(c=>c.textContent))");
  for(let i=0;i<evidence.rows.length;i++){
    const r=evidence.rows[i];
    const n=v=>v.toLocaleString('en-US');const p=v=>(100*v).toFixed(2)+'%';
    assert.deepEqual(actualRows[i],[r.letter+' · '+r.name,n(r.counts.localized),p(r.recall),p(r.precision),n(r.counts.unmapped),p(r.coverage)]);
  }
  assert(await evaluate("!document.querySelector('#matching') && !document.querySelector('#matching-results') && document.querySelectorAll('#pipeline .flow-box').length===4"));
  assert(await evaluate("!/[R]oMa|[D]INO|[V]iT|78,335|3,600 view|[S]tep 2/.test(document.body.textContent)"));
  for(const width of [768,1024,1400]){
    await cdp('Emulation.setDeviceMetricsOverride',{width,height:1050,deviceScaleFactor:1,mobile:false});
    await evaluate("[...document.querySelectorAll('details')].forEach(d=>d.open=true)");
    assert(await evaluate('document.documentElement.scrollWidth<=innerWidth'),'Page overflow at '+width);
  }
  await evaluate("document.querySelector('#pipeline').scrollIntoView()");
  await cdp('Page.captureScreenshot',{format:'png'}).then(r=>writeFile('/tmp/part-two-rewrite-flow.png',Buffer.from(r.data,'base64')));
  await evaluate("document.querySelector('#learning').scrollIntoView()");
  await cdp('Page.captureScreenshot',{format:'png'}).then(r=>writeFile('/tmp/part-two-rewrite-learning.png',Buffer.from(r.data,'base64')));
  console.log('Part Two passed desktop/mobile, expanded details, images, table counts and exact detection data and Part Two scope checks.');
  ws.close();
} finally {
  browser.kill();
  await new Promise(r => setTimeout(r,350));
  await rm(profile,{recursive:true,force:true,maxRetries:3});
}
