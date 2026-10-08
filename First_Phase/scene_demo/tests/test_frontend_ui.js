#!/usr/bin/env node
'use strict';
/*
 * Model-free inline-JavaScript contract tests for the scene_demo page.
 *
 * The externally supplied rendered HTML path (process.argv[2]) is read, its
 * single inline <script> is extracted and executed inside a Node `vm` context
 * with a minimal fake DOM/Image/fetch.  Only Node's built-in modules are used
 * (no npm dependencies, no network, no model).
 *
 * Covered contracts:
 *  - repeated terminal polling retains video node identity + playback position;
 *  - an identical image key does not re-render (stable frame nodes);
 *  - a failed/stale image keeps the previously good frame;
 *  - a slow poll never overlaps and an old request response is ignored;
 *  - Stop is enabled during planning, posts the exact owner payload and reports
 *    an honest failure (re-enabling the button).
 */

const fs = require('fs');
const vm = require('vm');

const htmlPath = process.argv[2];
if (!htmlPath) {
  console.error('usage: node test_frontend_ui.js <rendered.html>');
  process.exit(2);
}
const html = fs.readFileSync(htmlPath, 'utf8');
const match = html.match(/<script>([\s\S]*?)<\/script>/);
if (!match) {
  console.error('no inline <script> block found in ' + htmlPath);
  process.exit(2);
}
const inlineCode = match[1];

let failures = 0;
function check(name, cond, extra) {
  if (cond) {
    console.log('ok - ' + name);
  } else {
    failures += 1;
    console.error('FAIL - ' + name + (extra ? ' :: ' + extra : ''));
  }
}

// --------------------------------------------------------------------------- //
// fake DOM / Image / fetch
// --------------------------------------------------------------------------- //
const elements = new Map();

function makeElement(tag) {
  const el = {
    tagName: String(tag || 'div').toUpperCase(),
    id: '',
    className: '',
    textContent: '',
    innerHTML: '',
    value: '',
    disabled: false,
    controls: false,
    preload: '',
    src: '',
    currentTime: 0,
    style: {},
    children: [],
    replaceChildrenCalls: 0,
    _listeners: {},
    addEventListener(type, fn) {
      (this._listeners[type] = this._listeners[type] || []).push(fn);
    },
    appendChild(node) { this.children.push(node); return node; },
    setAttribute() {},
    getAttribute() { return null; },
    querySelector() { return makeElement('tbody'); },
  };
  el.replaceChildren = function replaceChildren(...nodes) {
    this.replaceChildrenCalls += 1;
    this.children = nodes;
    this.innerHTML = '';
  };
  return el;
}

function getElementById(id) {
  if (!elements.has(id)) {
    const el = makeElement('div');
    el.id = id;
    elements.set(id, el);
  }
  return elements.get(id);
}

const fakeDocument = {
  getElementById: getElementById,
  createElement: function createElement(tag) { return makeElement(tag); },
};

let imageFailTest = null;
function FakeImage() {
  const img = makeElement('img');
  img.alt = '';
  img.onload = null;
  img.onerror = null;
  Object.defineProperty(img, 'src', {
    configurable: true,
    get: function () { return img._src; },
    set: function (value) {
      img._src = value;
      const fail = imageFailTest ? imageFailTest(value) : false;
      Promise.resolve().then(function () {
        if (fail) {
          if (typeof img.onerror === 'function') img.onerror(new Error('image load failed'));
        } else if (typeof img.onload === 'function') {
          img.onload();
        }
      });
    },
  });
  return img;
}

let fetchRouter = null;
function fakeFetch(url, options) {
  return Promise.resolve().then(function () {
    if (fetchRouter) return fetchRouter(url, options || {});
    return new Promise(function () {}); // default: never resolves (parks the init tick)
  });
}

let timerSeq = 0;
const timers = [];
function fakeSetTimeout(fn, ms) {
  const id = { id: ++timerSeq, fn: fn, ms: ms, cancelled: false };
  timers.push(id);
  return id;
}
function fakeClearTimeout(id) { if (id) id.cancelled = true; }

// Active (non-cancelled) timers scheduled for the 2s polling tick.
function pendingTickTimers() {
  return timers.filter(function (t) { return !t.cancelled && t.ms === 2000; }).length;
}

function jsonResponse(obj, status) {
  const code = status || 200;
  return {
    ok: code >= 200 && code < 300,
    status: code,
    headers: {
      get: function (key) {
        return String(key).toLowerCase() === 'content-type' ? 'application/json' : null;
      },
    },
    json: function () { return Promise.resolve(obj); },
    text: function () { return Promise.resolve(JSON.stringify(obj)); },
  };
}

// --------------------------------------------------------------------------- //
// run the inline script
// --------------------------------------------------------------------------- //
const sandbox = {
  document: fakeDocument,
  Image: FakeImage,
  fetch: fakeFetch,
  setTimeout: fakeSetTimeout,
  clearTimeout: fakeClearTimeout,
  console: console,
};
const context = vm.createContext(sandbox);
vm.runInContext(inlineCode, context, { filename: 'inline.js' });

function run(expr) { return vm.runInContext(expr, context); }
async function flush(rounds) {
  const n = rounds || 6;
  for (let i = 0; i < n; i += 1) {
    await new Promise(function (r) { setImmediate(r); });
  }
}

// --------------------------------------------------------------------------- //
// tests
// --------------------------------------------------------------------------- //
function testStaticMarkup() {
  check('Stop button exists and is initially disabled',
    /<button[^>]*id="stop"[^>]*disabled/.test(html));
  check('Stop button keeps a red style hook', /button\.stop/.test(html) &&
    /#dc2626/.test(html));
  check('Stop hint explains preserve-current-scene behaviour',
    html.indexOf('停止将在当前动作或推理结束后生效，并保留当前场景') !== -1);
  check('stable img.frame aspect-ratio present', html.indexOf('aspect-ratio: 1 / 1') !== -1);
  check('square video rule mirrors img.frame',
    /video\s*\{[^}]*aspect-ratio:\s*1\s*\/\s*1[^}]*\}/.test(html),
    'video must set width 100% + aspect-ratio 1 / 1');
  check('cancelled status label says stopped in Chinese',
    inlineCode.indexOf('"已停止"') !== -1 && inlineCode.indexOf('"已取消"') === -1);
  check('unified tick uses a single setTimeout (no setInterval)',
    inlineCode.indexOf('setTimeout(tick') !== -1 && inlineCode.indexOf('setInterval(') === -1);
}

async function testVideoRetention() {
  run(`
    currentRequest = "req1";
    currentSessionId = "s1";
    globalThis.__jobsA = [
      {job_id:"j1", capability_id:"cap", request_id:"req1", session_id:"s1",
       rollout_path: RUNS_ROOT + "web_x/rollout.mp4"}
    ];
    renderVideos("req1", globalThis.__jobsA, true);
  `);
  await flush();
  check('terminal render mounts label + video',
    run('document.getElementById("videos").children.length') === 2);

  run('globalThis.__v0 = document.getElementById("videos").children[1];' +
      'globalThis.__v0.currentTime = 3.5;');
  run('renderVideos("req1", globalThis.__jobsA, true);'); // identical key
  await flush();
  check('identical key retains video node identity and playback position',
    run('document.getElementById("videos").children[1] === globalThis.__v0 &&' +
        ' globalThis.__v0.currentTime === 3.5') === true);

  // A new request invalidates the cache and rebuilds the nodes.
  run('renderVideos("req2", globalThis.__jobsA, true);');
  await flush();
  check('new request rebuilds video nodes',
    run('document.getElementById("videos").children[1] !== globalThis.__v0') === true);
}

async function testImageKeyIdempotent() {
  run(`
    currentSessionId = "s1";
    displayedImageKey = null;
    pendingImageKey = null;
    document.getElementById("images").replaceChildrenCalls = 0;
    renderSessionImages({session_id:"s1", scene_version:0,
      images:[{view:"agentview", image_path: RUNS_ROOT+"s1/agentview.png", sha256:"h1"}]});
  `);
  await flush();
  const first = run('document.getElementById("images").replaceChildrenCalls');
  check('first image key commits frame nodes', first >= 1, 'calls=' + first);

  run(`renderSessionImages({session_id:"s1", scene_version:0,
      images:[{view:"agentview", image_path: RUNS_ROOT+"s1/agentview.png", sha256:"h1"}]});`);
  await flush();
  const second = run('document.getElementById("images").replaceChildrenCalls');
  check('identical image key does not update the DOM', second === first,
    'before=' + first + ' after=' + second);
}

async function testFailedImageKeepsGoodFrame() {
  run(`
    currentSessionId = "s2";
    displayedImageKey = null;
    pendingImageKey = null;
    renderSessionImages({session_id:"s2", scene_version:0,
      images:[{view:"agentview", image_path: RUNS_ROOT+"s2/good.png", sha256:"g1"}]});
  `);
  await flush();
  check('good frame displayed', run('document.getElementById("images").children.length') >= 1);
  run('globalThis.__good = document.getElementById("images").children;');

  imageFailTest = function (v) { return String(v).indexOf('bad.png') !== -1; };
  run(`renderSessionImages({session_id:"s2", scene_version:0,
      images:[{view:"agentview", image_path: RUNS_ROOT+"s2/bad.png", sha256:"b1"}]});`);
  await flush();
  imageFailTest = null;

  check('failed/stale image keeps previous good frame nodes',
    run('document.getElementById("images").children === globalThis.__good') === true);
}

async function testSlowPollNoOverlap() {
  run(`
    currentRequest = "reqA";
    currentSessionId = "sA";
    currentSession = {state:"ready", session_id:"sA"};
  `);
  let resolveAgent = null;
  let agentFetches = 0;
  let planFetches = 0;
  fetchRouter = function (url) {
    if (url === '/api/agent/reqA') {
      agentFetches += 1;
      return new Promise(function (resolve) {
        resolveAgent = function () {
          resolve(jsonResponse({status: 'running', request_id: 'reqA',
            session_id: 'sA', result: null}));
        };
      });
    }
    if (url.indexOf('/api/plan/') === 0) { planFetches += 1; return jsonResponse({}, 404); }
    return new Promise(function () {});
  };

  const first = run('pollCurrent()');
  await flush();
  const second = run('pollCurrent()'); // must be refused by the in-flight guard
  await flush();
  check('slow poll never overlaps', agentFetches === 1, 'agentFetches=' + agentFetches);

  // A newer request supersedes reqA while its poll is still in flight.
  run('currentRequest = "reqB";');
  if (typeof resolveAgent === 'function') resolveAgent();
  await first;
  await second;
  await flush();
  check('old response does not rewind a newer request',
    run('currentRequest') === 'reqB' && planFetches === 0,
    'currentRequest=' + run('currentRequest') + ' planFetches=' + planFetches);
}

async function testStopFlow() {
  run(`
    currentSessionId = "sZ";
    currentSession = {state:"ready", session_id:"sZ", capabilities:[]};
    currentRequest = null;
    currentAgentStatus = null;
    stopRequested = false;
    document.getElementById("request").value = "整理桌面";
    document.getElementById("case").value = "";
  `);
  let cancelUrl = null;
  let cancelBody = null;
  let failCancel = false;
  fetchRouter = function (url, options) {
    if (url === '/api/agent') {
      return jsonResponse({request_id: 'reqZ', session_id: 'sZ', status: 'queued'}, 202);
    }
    if (url === '/api/agent/reqZ/cancel') {
      cancelUrl = url;
      try { cancelBody = JSON.parse(options.body); } catch (e) { cancelBody = options.body; }
      if (failCancel) return jsonResponse({error: 'service down'}, 502);
      return jsonResponse({ok: true, request_id: 'reqZ', session_id: 'sZ',
        cancel_requested: true, status: 'cancelling'});
    }
    return new Promise(function () {}); // park the unified tick
  };

  await run('submitRequest()');
  await flush();
  check('Stop enabled after acceptance, before any plan exists',
    run('document.getElementById("stop").disabled') === false);
  check('new submission disabled while queued',
    run('document.getElementById("run").disabled') === true);
  check('no cancel posted before Stop is clicked', cancelUrl === null);

  failCancel = true;
  await run('stopRequest()');
  await flush();
  check('Stop posts the exact current request/session',
    cancelUrl === '/api/agent/reqZ/cancel' && cancelBody &&
    cancelBody.session_id === 'sZ', JSON.stringify(cancelBody));
  check('Stop failure re-enables the button',
    run('document.getElementById("stop").disabled') === false);
  check('Stop failure displays an honest reason',
    /停止失败/.test(run('document.getElementById("runmsg").textContent')));

  failCancel = false;
  await run('stopRequest()');
  await flush();
  check('successful Stop keeps the button disabled',
    run('document.getElementById("stop").disabled') === true);
  check('Stop never clears the current request',
    run('currentRequest') === 'reqZ');
}

async function testStopResponseOwnershipRaces() {
  const cases = [
    { status: 'completed', expect: '任务已结束，无需停止。' },
    { status: 'error', expect: '任务已结束，无需停止。' },
    { status: 'cancelled', expect: '已停止' },
    { status: 'cancelling',
      expect: '停止请求已受理，当前动作或推理结束后生效（保留当前场景）。' },
  ];
  for (let i = 0; i < cases.length; i += 1) {
    const c = cases[i];
    run(`
      currentSessionId = "sRace";
      currentSession = {state:"ready", session_id:"sRace", capabilities:[]};
      currentRequest = "reqRace";
      currentAgentStatus = "running";
      stopRequested = false;
      stopInFlight = false;
      document.getElementById("runmsg").textContent = "";
      document.getElementById("request").value = "整理桌面";
      document.getElementById("images").replaceChildrenCalls = 0;
      document.getElementById("videos").replaceChildrenCalls = 0;
    `);
    fetchRouter = function (url) {
      if (url === '/api/agent/reqRace/cancel') {
        // A successful cancel response whose *actual* status is a terminal race
        // (completed/error/cancelled) or the active acknowledging cancelling.
        return jsonResponse({ok: true, request_id: 'reqRace', session_id: 'sRace',
          cancel_requested: true, status: c.status,
          noop: c.status !== 'cancelling'});
      }
      return new Promise(function () {});
    };
    await run('stopRequest()');
    await flush();
    check('Stop ack (' + c.status + ') shows an honest message',
      run('document.getElementById("runmsg").textContent') === c.expect,
      run('document.getElementById("runmsg").textContent'));
    check('Stop ack (' + c.status + ') preserves request/session ownership',
      run('currentRequest') === 'reqRace' && run('currentSessionId') === 'sRace' &&
      run('currentSession.state') === 'ready');
    check('Stop ack (' + c.status + ') clears no media',
      run('document.getElementById("images").replaceChildrenCalls') === 0 &&
      run('document.getElementById("videos").replaceChildrenCalls') === 0);
  }
}

async function testPollCurrentStopRunmsgRefresh() {
  // Exercise the *actual* pollCurrent function: after a real Stop, the visible
  // runmsg must be refreshed from the exact-owned agentJob status instead of
  // retaining "stop accepted" forever. Plan is always 404 (no plan submitted).
  run(`
    currentSessionId = "sP";
    currentSession = {state:"ready", session_id:"sP", capabilities:[], images:[]};
    currentRequest = "reqP";
    currentAgentStatus = "running";
    stopRequested = false;
    stopInFlight = false;
    pollInFlight = false;
    showCache.clear();
    document.getElementById("runmsg").textContent = "停止失败：service down";
    document.getElementById("images").replaceChildrenCalls = 0;
    document.getElementById("videos").replaceChildrenCalls = 0;
  `);

  // ---- queued/running after a failed Stop: keep the honest failure text ----
  const activeStates = ['queued', 'running'];
  for (let i = 0; i < activeStates.length; i += 1) {
    const st = activeStates[i];
    fetchRouter = function (url) {
      if (url === '/api/agent/reqP') {
        return jsonResponse({status: st, request_id: 'reqP', session_id: 'sP', result: null});
      }
      if (url.indexOf('/api/plan/') === 0) return jsonResponse({}, 404);
      return new Promise(function () {});
    };
    await run('pollCurrent()');
    await flush();
    check('failed Stop text retained while job is ' + st,
      run('document.getElementById("runmsg").textContent') === '停止失败：service down',
      run('document.getElementById("runmsg").textContent'));
  }

  // ---- cancelling: says stopping, never claims stopped ----
  fetchRouter = function (url) {
    if (url === '/api/agent/reqP') {
      return jsonResponse({status: 'cancelling', request_id: 'reqP',
        session_id: 'sP', result: null});
    }
    if (url.indexOf('/api/plan/') === 0) return jsonResponse({}, 404);
    return new Promise(function () {});
  };
  await run('pollCurrent()');
  await flush();
  check('cancelling poll shows the stopping message',
    run('document.getElementById("runmsg").textContent') === '正在停止…',
    run('document.getElementById("runmsg").textContent'));

  const note = '真实计划已终态，但归属本次请求的终态 job 尚未就绪：保持 cancelling 并等待真实 job 确认。';
  fetchRouter = function (url) {
    if (url === '/api/agent/reqP') {
      return jsonResponse({status: 'cancelling', request_id: 'reqP',
        session_id: 'sP', result: null, cancel_note: note});
    }
    if (url.indexOf('/api/plan/') === 0) return jsonResponse({}, 404);
    return new Promise(function () {});
  };
  await run('pollCurrent()');
  await flush();
  check('cancelling poll shows the exact nonempty cancel_note',
    run('document.getElementById("runmsg").textContent') === note,
    run('document.getElementById("runmsg").textContent'));

  // ---- exact cancelled result + plan 404: say stopped, no media recreation ----
  run(`
    currentAgentStatus = "cancelling";
    stopRequested = true;                      // a real Stop was accepted
    showCache.clear();
    document.getElementById("runmsg").textContent =
      "停止请求已受理，当前动作或推理结束后生效（保留当前场景）。";
    document.getElementById("run").disabled = true;
    document.getElementById("stop").disabled = false;
    document.getElementById("images").replaceChildrenCalls = 0;
    document.getElementById("videos").replaceChildrenCalls = 0;
  `);
  fetchRouter = function (url) {
    if (url === '/api/agent/reqP') {
      return jsonResponse({status: 'cancelled', request_id: 'reqP', session_id: 'sP',
        result: {cancelled_by_user: true, plan: null, jobs: [], steps: 0}});
    }
    if (url.indexOf('/api/plan/') === 0) return jsonResponse({error: 'not found'}, 404);
    return new Promise(function () {});
  };
  await run('pollCurrent()');
  await flush();
  check('cancelled poll updates runmsg to the stopped message',
    run('document.getElementById("runmsg").textContent') === '已停止（保留当前场景）。',
    run('document.getElementById("runmsg").textContent'));
  check('cancelled poll shows the plan panel as stopped (no stale planning state)',
    run('document.getElementById("plan").innerHTML').indexOf('已停止') !== -1 &&
    run('document.getElementById("plan").innerHTML').indexOf('正在规划') === -1);
  check('cancelled poll enables Submit for a ready session',
    run('document.getElementById("run").disabled') === false);
  check('cancelled poll disables Stop',
    run('document.getElementById("stop").disabled') === true);
  check('cancelled poll recreates no media',
    run('document.getElementById("images").replaceChildrenCalls') === 0 &&
    run('document.getElementById("videos").replaceChildrenCalls') === 0);
  check('cancelled poll retains the current request/session',
    run('currentRequest') === 'reqP' && run('currentSessionId') === 'sP');

  // ---- completed/error Stop races: honest "task has ended" message ----
  const races = ['completed', 'error'];
  for (let i = 0; i < races.length; i += 1) {
    const st = races[i];
    run(`
      currentAgentStatus = "running";
      stopRequested = true;                    // Stop was requested (race)
      showCache.clear();
      document.getElementById("runmsg").textContent =
        "停止请求已受理，当前动作或推理结束后生效（保留当前场景）。";
    `);
    fetchRouter = function (url) {
      if (url === '/api/agent/reqP') {
        return jsonResponse({status: st, request_id: 'reqP', session_id: 'sP', result: null});
      }
      if (url.indexOf('/api/plan/') === 0) return jsonResponse({}, 404);
      return new Promise(function () {});
    };
    await run('pollCurrent()');
    await flush();
    check('Stop race (' + st + ') shows the honest ended message',
      run('document.getElementById("runmsg").textContent') === '任务已结束，无需停止。',
      run('document.getElementById("runmsg").textContent'));
  }

  // ---- completed without any Stop intent: leave the message untouched ----
  run(`
    currentAgentStatus = "running";
    stopRequested = false;
    showCache.clear();
    document.getElementById("runmsg").textContent = "请求已受理：reqP（执行中）";
  `);
  fetchRouter = function (url) {
    if (url === '/api/agent/reqP') {
      return jsonResponse({status: 'completed', request_id: 'reqP',
        session_id: 'sP', result: null});
    }
    if (url.indexOf('/api/plan/') === 0) return jsonResponse({}, 404);
    return new Promise(function () {});
  };
  await run('pollCurrent()');
  await flush();
  check('completed without a Stop intent keeps the existing runmsg',
    run('document.getElementById("runmsg").textContent') === '请求已受理：reqP（执行中）',
    run('document.getElementById("runmsg").textContent'));
}

async function testBootRecoveryIgnoresTerminal() {
  run('currentRequest = null; currentSessionId = null; currentAgentStatus = null;' +
      ' stopRequested = false; stopInFlight = false;');
  fetchRouter = function (url) {
    if (url === '/api/agent') {
      return jsonResponse({jobs: [
        {request_id: 't1', session_id: 's1', status: 'completed'},
        {request_id: 't2', session_id: 's2', status: 'error'}]});
    }
    return new Promise(function () {}); // never create/reset a scene during recovery
  };
  await run('recoverActiveJob()');
  await flush();
  check('recovery never adopts a historical terminal job',
    run('currentRequest') === null && run('currentSessionId') === null,
    'currentRequest=' + run('currentRequest'));
}

async function testBootRecoveryZeroOrMultiple() {
  run('currentRequest = null; currentSessionId = null; currentAgentStatus = null;');
  fetchRouter = function (url) {
    if (url === '/api/agent') return jsonResponse({jobs: []});
    return new Promise(function () {});
  };
  await run('recoverActiveJob()');
  await flush();
  check('zero active leaves the initial view', run('currentRequest') === null);

  fetchRouter = function (url) {
    if (url === '/api/agent') {
      return jsonResponse({jobs: [
        {request_id: 'a1', session_id: 's1', status: 'running'},
        {request_id: 'a2', session_id: 's2', status: 'queued'}]});
    }
    return new Promise(function () {});
  };
  await run('recoverActiveJob()');
  await flush();
  check('multiple active selects none', run('currentRequest') === null);
}

async function testBootRecoverySingleActive() {
  run('currentRequest = null; currentSessionId = null; currentAgentStatus = null;' +
      ' stopRequested = false; stopInFlight = false;');
  let sessionUrl = null;
  const seenUrls = [];
  fetchRouter = function (url) {
    seenUrls.push(url);
    if (url === '/api/agent') {
      return jsonResponse({jobs: [
        {request_id: 'hist', session_id: 'sOld', status: 'completed'},
        {request_id: 'reqR', session_id: 'sR', status: 'running'}]});
    }
    if (url === '/api/session/sR') {
      sessionUrl = url;
      return jsonResponse({session_id: 'sR', state: 'ready', images: [],
        capabilities: [], storage_policy: {}});
    }
    return new Promise(function () {});
  };
  await run('recoverActiveJob()');
  await flush();
  check('recovery adopts the exact single active request/session',
    run('currentRequest') === 'reqR' && run('currentSessionId') === 'sR',
    'request=' + run('currentRequest') + ' session=' + run('currentSessionId'));
  check('recovery reads the exact session and never creates one',
    sessionUrl === '/api/session/sR' &&
    seenUrls.every(function (u) { return u.indexOf('/api/session') !== 0 || u === '/api/session/sR'; }));
  check('recovery enables Stop for the single active job',
    run('document.getElementById("stop").disabled') === false);
}

async function testSinglePollingChain() {
  timers.length = 0;
  run('pollTimer = null; tickInFlight = false; pollInFlight = false;' +
      ' currentRequest = null; currentSessionId = "sB";' +
      ' currentSession = {state:"ready", session_id:"sB"};' +
      ' document.getElementById("request").value = "整理";' +
      ' document.getElementById("case").value = "";');
  let healthFetches = 0;
  let resolveHealth = null;
  fetchRouter = function (url, options) {
    if (url === '/api/health') {
      healthFetches += 1;
      return new Promise(function (resolve) {
        resolveHealth = function () { resolve(jsonResponse({ready: true})); };
      });
    }
    if (url === '/api/agent') {
      return jsonResponse({request_id: 'reqB', session_id: 'sB', status: 'queued'}, 202);
    }
    if (url.indexOf('/api/session/') === 0) {
      return jsonResponse({session_id: 'sB', state: 'ready', images: [], capabilities: []});
    }
    if (url.indexOf('/api/plan/') === 0) { return jsonResponse({}, 404); }
    if (url.indexOf('/api/agent/') === 0) {
      return jsonResponse({status: 'queued', request_id: 'reqB',
        session_id: 'sB', result: null});
    }
    return new Promise(function () {});
  };

  const tickPromise = run('tick()');   // a slow boot tick, parked on /api/health
  await flush();
  check('slow boot tick issues exactly one health request',
    healthFetches === 1, 'healthFetches=' + healthFetches);

  await run('submitRequest()');        // calls startPolling while the tick is in flight
  await flush();
  check('startPolling during an in-flight tick starts no second chain',
    healthFetches === 1, 'healthFetches=' + healthFetches);

  if (resolveHealth) resolveHealth();
  await tickPromise;
  await flush();
  check('the slow tick schedules exactly one next timeout',
    pendingTickTimers() === 1, 'pending=' + pendingTickTimers());
}

// --------------------------------------------------------------------------- //
// actual renderSubgoals diagnostics (grasp phase / stopping reason + wine rubric)
// --------------------------------------------------------------------------- //
// Runs the *actual* renderSubgoals(plan, jobs) and returns the committed table HTML.
function renderSubgoalsHtml(plan, jobs) {
  run('globalThis.__sgPlan = ' + JSON.stringify(plan) + ';' +
      'globalThis.__sgJobs = ' + JSON.stringify(jobs) + ';' +
      'currentSession = {capabilities: []};' +
      'renderSubgoals(globalThis.__sgPlan, globalThis.__sgJobs);');
  return run('document.getElementById("subgoals").innerHTML');
}

function sgPlan(caps) {
  return {
    capability_ids: caps,
    completed_capability_ids: [],
    pending_capability_ids: [],
    state: 'completed',
    decision: 'execute',
    plan_success: false,
  };
}

async function testRenderSubgoalsDiagnostics() {
  // 1. enforce failed grasp -> task stopped (Chinese label).
  const enforce = renderSubgoalsHtml(sgPlan(['wine_to_rack']), [
    {job_id: 'j1', capability_id: 'wine_to_rack', state: 'completed', success: false,
     ended_reason: 'failed_grasp', grasp_guard_mode: 'enforce'}
  ]);
  check('renderSubgoals: enforce failed_grasp is labelled task stopped',
    enforce.indexOf('未抓起，任务已停止') !== -1, enforce);

  // 2. shadow failed grasp -> observation mode, never claims stopped.
  const shadow = renderSubgoalsHtml(sgPlan(['wine_to_rack']), [
    {job_id: 'j1', capability_id: 'wine_to_rack', state: 'completed', success: false,
     ended_reason: 'failed_grasp', grasp_guard_mode: 'shadow'}
  ]);
  check('renderSubgoals: shadow failed_grasp is observation mode and never stopped',
    shadow.indexOf('检测到未抓起（观察模式）') !== -1 &&
    shadow.indexOf('未抓起，任务已停止') === -1, shadow);

  // 3. grasp_confirmed is a past confirmation, not a current hold.
  const confirmed = renderSubgoalsHtml(sgPlan(['wine_to_rack']), [
    {job_id: 'j1', capability_id: 'wine_to_rack', state: 'completed', success: true,
     grasp_stage: 'grasp_confirmed', grasp_guard_mode: 'enforce'}
  ]);
  check('renderSubgoals: grasp_confirmed shown as past confirmation',
    confirmed.indexOf('已确认抓起过') !== -1, confirmed);

  // 4. attempting / unknown native grasp stages.
  const stages = renderSubgoalsHtml(sgPlan(['wine_to_rack']), [
    {job_id: 'j1', capability_id: 'wine_to_rack', state: 'running', success: null,
     grasp_stage: 'attempting'},
    {job_id: 'j2', capability_id: 'wine_to_rack', state: 'running', success: null,
     grasp_stage: 'unknown'}
  ]);
  check('renderSubgoals: attempting maps to trying to grasp',
    stages.indexOf('正在尝试抓取') !== -1, stages);
  check('renderSubgoals: unknown maps to awaiting confirmation',
    stages.indexOf('抓取状态待确认') !== -1, stages);

  // 5. native miss + valid semantic true: dual display while job.success stays false.
  const dual = renderSubgoalsHtml(sgPlan(['wine_to_rack']), [
    {job_id: 'j1', capability_id: 'wine_to_rack', state: 'completed', success: false,
     native_wine_predicate: false, semantic_spec_id: 'semantic_wine_rack_v1',
     semantic_success: true, semantic_observation_samples: 3,
     semantic_candidate_streak: 2, semantic_state: 'released'}
  ]);
  check('renderSubgoals: native miss + semantic true dual display',
    dual.indexOf('未命中') !== -1 &&
    dual.indexOf('已松手并稳定支撑在架子上') !== -1 &&
    dual.indexOf('语义达成，标准区域未命中') !== -1, dual);
  check('renderSubgoals: semantic success never promotes job.success',
    dual.indexOf('class="bad-text">false') !== -1, dual);

  // 6. null native / null spec -> cannot confirm yet, never a false failure.
  const unknown = renderSubgoalsHtml(sgPlan(['wine_to_rack']), [
    {job_id: 'j1', capability_id: 'wine_to_rack', state: 'completed', success: null,
     native_wine_predicate: null, semantic_spec_id: null}
  ]);
  check('renderSubgoals: null native/spec shows cannot-confirm, not false failure',
    unknown.indexOf('尚无法确认') !== -1 &&
    unknown.indexOf('未命中') === -1 &&
    unknown.indexOf('尚未满足') === -1, unknown);

  // 7. wine before any sample -> still unknown.
  const noSample = renderSubgoalsHtml(sgPlan(['wine_to_rack']), [
    {job_id: 'j1', capability_id: 'wine_to_rack', state: 'running', success: null,
     native_wine_predicate: true, semantic_spec_id: 'semantic_wine_rack_v1',
     semantic_success: null, semantic_observation_samples: 0}
  ]);
  check('renderSubgoals: wine with no samples yet stays unknown',
    noSample.indexOf('尚无法确认') !== -1 &&
    noSample.indexOf('已松手并稳定支撑在架子上') === -1, noSample);

  // 8. non-wine job bypasses wine diagnostics (disabled, never a false failure).
  const nonWine = renderSubgoalsHtml(sgPlan(['stack_block']), [
    {job_id: 'j1', capability_id: 'stack_block', state: 'completed', success: false}
  ]);
  check('renderSubgoals: non-wine bypasses wine diagnostics without false failure',
    nonWine.indexOf('未启用') !== -1 &&
    nonWine.indexOf('未命中') === -1 &&
    nonWine.indexOf('尚无法确认') === -1, nonWine);

  // 9. unknown spec cannot claim semantic success.
  const badSpec = renderSubgoalsHtml(sgPlan(['wine_to_rack']), [
    {job_id: 'j1', capability_id: 'wine_to_rack', state: 'completed', success: false,
     native_wine_predicate: false, semantic_spec_id: 'other_spec',
     semantic_success: true, semantic_observation_samples: 5}
  ]);
  check('renderSubgoals: unknown spec cannot claim semantic success',
    badSpec.indexOf('已松手并稳定支撑在架子上') === -1 &&
    badSpec.indexOf('尚无法确认') !== -1, badSpec);
}

async function testRenderSubgoalsEscapeAndCounters() {
  // Malicious dynamic strings must be escaped, never rendered as raw HTML.
  const evil = renderSubgoalsHtml(sgPlan(['wine_to_rack']), [
    {job_id: '<img src=x onerror=alert(1)>', capability_id: 'wine_to_rack',
     state: 'completed', success: null,
     grasp_stage: '<img src=x onerror=alert(1)>',
     ended_reason: '<script>bad</script>',
     native_wine_predicate: null,
     semantic_spec_id: 'semantic_wine_rack_v1',
     semantic_success: false, semantic_observation_samples: 2,
     semantic_state: '<b>evil</b>'}
  ]);
  check('renderSubgoals: dynamic strings are escaped (no raw HTML injection)',
    evil.indexOf('<img') === -1 && evil.indexOf('<script>bad') === -1 &&
    evil.indexOf('<b>evil') === -1 &&
    evil.indexOf('&lt;img') !== -1 && evil.indexOf('&lt;script&gt;bad') !== -1 &&
    evil.indexOf('&lt;b&gt;evil') !== -1, evil);
  check('renderSubgoals: valid sample counter is displayed',
    evil.indexOf('样本 2') !== -1, evil);

  // Invalid counters (negative / non-integer) must render as '-', never the raw value.
  const badCounters = renderSubgoalsHtml(sgPlan(['wine_to_rack']), [
    {job_id: 'j1', capability_id: 'wine_to_rack', state: 'running', success: null,
     native_wine_predicate: null, semantic_spec_id: 'semantic_wine_rack_v1',
     semantic_success: null, semantic_observation_samples: -4,
     semantic_candidate_streak: 1.5}
  ]);
  check('renderSubgoals: invalid counters render as dash, not the raw value',
    badCounters.indexOf('样本 -') !== -1 && badCounters.indexOf('-4') === -1 &&
    badCounters.indexOf('1.5') === -1, badCounters);
}

(async function main() {
  testStaticMarkup();
  await testVideoRetention();
  await testImageKeyIdempotent();
  await testFailedImageKeepsGoodFrame();
  await testSlowPollNoOverlap();
  await testBootRecoveryIgnoresTerminal();
  await testBootRecoveryZeroOrMultiple();
  await testBootRecoverySingleActive();
  await testSinglePollingChain();
  await testStopFlow();
  await testStopResponseOwnershipRaces();
  await testPollCurrentStopRunmsgRefresh();
  await testRenderSubgoalsDiagnostics();
  await testRenderSubgoalsEscapeAndCounters();
  if (failures) {
    console.error(failures + ' frontend contract test(s) failed');
    process.exit(1);
  }
  console.log('all frontend contract tests passed');
})().catch(function (err) {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
