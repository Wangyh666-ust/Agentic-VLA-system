"""Local interactive X-VLA console. Import/startup never creates a simulator.

The Hermes bridge records a draft only. Only the user's execute endpoint calls
the existing XvlaSceneService.submit_plan; motion/check/finish code is reused.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import secrets
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / 'results'
OFFLINE = RESULTS / '2026-10-14-partial-scope-offline'
PREVIOUS = RESULTS / '2026-10-14-qwen38-three-requests'
HOME = Path('/home/yhwang/fyp/xvla_native')
TERMINAL = {'completed', 'blocked', 'error', 'cancelled'}
EXAMPLES = [
    {'id': 'table_tidy_stove', 'scene': 'goal_table', 'label': '整理桌面', 'request': '整理桌面，包括碗、酒瓶，并且打开灶台'},
    {'id': 'heat_yellow_mug', 'scene': 'microwave_mug', 'label': '加热黄杯', 'request': '帮我将黄杯子加热一下'},
    {'id': 'discard_two_cans', 'scene': 'basket_two', 'label': '处理双罐', 'request': '帮我扔一下桌面上的两个罐头'},
]


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def frames(capture):
    values = capture.get('checker_frames') or capture.get('frames') or []
    return [{'view': f['view'], 'path': f['rgb_path'],
             'timestamp': f.get('timestamp', capture.get('timestamp'))}
            for view in ('wrist', 'agentview', 'sideview') for f in values if f.get('view') == view]


def task_text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError('请看过场景后填写任务（不超过2000字）。')
    return value.strip()


def catalog_data():
    from scene_demo import catalog
    return [{'id': k, 'label': v.get('label', k), 'description': v.get('description'),
             'capabilities': catalog.scene_capabilities(k, backend='xvla')}
            for k, v in catalog.SCENES.items() if 'xvla' in v.get('backends', ['xvla'])]


def validate_draft(session, payload, request_id):
    """Same production validators, plus bridge ownership and mandatory coverage."""
    from scene_demo import service
    from scene_demo.planning_scope import validate_request_scope
    from scene_demo.xvla_grounding import validate_selection
    if payload.get('request_id') != request_id or payload.get('session_id') != session['session_id']:
        raise ValueError('计划不属于本次请求')
    if payload.get('request_scope') is None:
        raise ValueError('缺少能力覆盖说明 request_scope')
    validate_request_scope(payload['request_scope'], payload.get('decision'))
    error = service.validate_plan_request(session, None, payload)
    if error:
        raise ValueError(json.dumps(error, ensure_ascii=False))
    if payload['decision'] == 'execute':
        caps = {c['capability_id']: c for c in session['capabilities']}
        validate_selection(session['scene_inventory'], payload.get('target_bindings'),
                           [caps[c] for c in payload['capability_ids']])
    # Do not accept model-supplied private route hints or arbitrary extra fields.
    keys = ('request_id', 'session_id', 'scene_version', 'decision', 'capability_ids',
            'rationale', 'request_scope', 'target_bindings')
    return {**{k: copy.deepcopy(payload[k]) for k in keys if k in payload},
            'audit': False, 'budget_per_subgoal': 800 if session['scene_id'] == 'goal_table' else 900}


class Console:
    def __init__(self, root=None):
        self.root = Path(root or RESULTS / 'interactive-console')
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self.bridge_token = secrets.token_urlsafe(32)
        self.url = ''
        self.svc = None
        self.model = None
        self.cancel = threading.Event()
        self.busy = False
        self.session = None
        self.draft = None
        self.directory = None
        self.state = {'mode': 'welcome', 'stage': 'idle', 'message': '选择历史示例，或开始自己的仿真。', 'events': []}

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            if 'message' in values:
                self.state.setdefault('events', []).append({'time': time.strftime('%H:%M:%S'), 'text': values['message']})
            if self.directory:
                write(self.directory / 'console_state.json', self.state)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy({**self.state, 'busy': self.busy})

    def start(self, fn):
        with self.lock:
            if self.busy:
                raise ValueError('当前步骤尚未结束，请等待或停止。')
            self.busy = True
            self.cancel.clear()
        def work():
            try:
                fn()
            except Exception as exc:
                if self.directory:
                    (self.directory / 'console_error.log').write_text(traceback.format_exc(), encoding='utf-8')
                self.update(stage='cancelled' if self.cancel.is_set() else 'error', message=str(exc))
            finally:
                with self.lock:
                    self.busy = False
        threading.Thread(target=work, daemon=True).start()

    def check_cancel(self):
        if self.cancel.is_set():
            raise RuntimeError('操作已停止；未提交新的动作计划。')

    def history(self, name):
        with self.lock:
            if self.busy or (self.state.get('mode') == 'live' and self.state.get('stage') in ('ready', 'draft')):
                raise ValueError('请先停止当前尝试，再切换历史示例。')
            spec = next(x for x in EXAMPLES if x['id'] == name)
            directory = OFFLINE / name
            session = read(directory / 'session_public.json')
            capture = read(directory / 'grounding/capture.json')
            plan = read(directory / 'submitted_plans.json')[0]
            self.directory = None
            self.session = self.draft = None
            self.state = {'mode': 'history', 'stage': 'history', 'message': '历史离线规划 · 2026-10-14 · 没有执行动作',
                          'request': spec['request'], 'scene': spec['scene'], 'frames': frames(capture),
                          'inventory': session['scene_inventory'], 'capabilities': session['capabilities'],
                          'draft': plan, 'events': [], 'history_note':
                          '双罐外部视角可识别，腕部左右对应仍有错误；这些模型标注可以有误，不是人工真值。'
                          if name == 'discard_two_cans' else '这是已有照片与模型原始规划，可用于熟悉界面；不代表新一轮执行结果。'}

    def preflight(self, scene, request=None):
        """Review gate before even the initial ten settling actions."""
        base = read(HERE / 'plans/20261014-qwen38-three-requests.json')
        base.update(id=self.directory.name, question_key='interactive-xvla-console',
                    purpose='用户主动操作交互界面，检查计划经确认才执行，保留当前原生动作、收尾与复位；不作成功率推断。',
                    prior_evidence=['20261014-qwen38-three-requests', '20261014-partial-scope-offline'],
                    history_review=[{'experiment_id': x, 'disposition': 'reuse', 'reason': r} for x, r in [
                        ('20261014-qwen38-three-requests', '复用同一X-VLA原生控制与收尾，已知整理通过、黄杯最终稳定筛查失败、双罐初识别网络失败。'),
                        ('20261014-partial-scope-offline', '部分覆盖新规则已离线验证；入炉关门可为加热准备，丢弃不能用入篮冒充。')]],
                    hypothesis='页面传递原文和三视图，草稿阶段零VLA；确认后服务执行同一计划并如实报告限制。',
                    information_gain='首次用户分步操作当前流程；验证草稿与执行隔离，不重做原生基线。',
                    changed_factors=[{'name': 'interactive_entry', 'before': '批处理直接提交',
                                      'after': '用户查看真实Hermes草稿后主动确认', 'reason': '用户请求交互式操作全流程。'}],
                    fixed_factors={'seed': 142, 'init_state_index': 0, 'helpers': False,
                                   'finish_retreat_reset': '原XvlaSceneService与默认配置',
                                   'planner': 'qwen3.8-max', 'terminal_checker': 'qwen3-vl-plus'},
                    success_criteria='草稿零VLA；确认分支实际提交一次，记录真实动作/松手三图/检查；独立完成筛查按原服务执行。部分完成不等于完整需求完成。',
                    failure_criteria='草稿执行、重复提交、身份绑定丢失为接口失败；动作与收尾失败分开记录。',
                    unknown_criteria='模型超时、证据不足或用户中止为未知，不算物理成功。',
                    scope={'max_cases': 1, 'max_vla_actions_per_case': 8000, 'max_helper_actions_per_case': 8000,
                           'stop_rule': '一次用户尝试，无自动修复/重跑；总服务上限8000步；应触发未触发停止依赖验证；用户停止沿用原取消接口。'},
                    new_physical_actions=8000, repeat={'needed': True, 'reason': '用户主动交互复现，记录界面路径及确认边界。'},
                    configuration_complete=True,
                    configuration={'scene': scene, 'request': request, 'seed': 142, 'init_state_index': 0,
                                   'task_selection': '任务在预览后由用户提交，记录于task_request.json；载入不绑定任务、不调用Qwen/Hermes。',
                                   'budget_per_subgoal': 800 if scene == 'goal_table' else 900,
                                   'session_limit': 8000, 'source_sha256': {}})
        snapshot = self.directory / 'source_snapshot'
        for source in sorted(HERE.glob('*.py')) + sorted((HERE / 'finish_v1').rglob('*.py')):
            relative = source.relative_to(HERE)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            base['configuration']['source_sha256'][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        write(self.directory / 'proposal.json', base)
        self.update(stage='preflight', message='核对实验历史与本次范围（首次读取证据可能需要数分钟）。')
        command = [sys.executable, str(ROOT / '.agents/skills/fyp-experiment-review/scripts/review.py'),
                   'check', '--proposal', str(self.directory / 'proposal.json')]
        deadline = time.monotonic() + 900
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            while True:
                try:
                    stdout, stderr = proc.communicate(timeout=.5)
                    break
                except subprocess.TimeoutExpired:
                    if self.cancel.is_set() or time.monotonic() > deadline:
                        proc.terminate()
                        stdout, stderr = proc.communicate(timeout=5)
                        (self.directory / 'review_check.stderr').write_bytes(stderr)
                        self.check_cancel()
                        raise RuntimeError('实验历史检查超时，未创建仿真。')
        (self.directory / 'review_check.json').write_bytes(stdout)
        (self.directory / 'review_check.stderr').write_bytes(stderr)
        if proc.returncode:
            raise RuntimeError('实验历史检查未通过，未创建仿真；详情已保存在本次记录。')

    def ensure_runtime(self):
        if self.svc:
            if not self.svc.health().get('ready'):
                raise RuntimeError('X-VLA服务未就绪，请查看本次记录。')
            return
        self.update(stage='loading', message='正在加载 X-VLA；尚未创建场景。')
        spec = copy.deepcopy(read(RESULTS / '2026-10-13-xvla-timeline-trial/ready.json')['future_server'])
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(('127.0.0.1', 8794)) == 0:
                raise RuntimeError('8794端口已被其他进程使用；未启动第二个模型，请先关闭原服务。')
        spec['argv'][-1] = str(self.root / 'model_output')
        env = dict(os.environ, **spec['env'])
        with (self.root / 'model.log').open('ab') as stream:
            self.model = subprocess.Popen(spec['argv'], cwd=spec['cwd'], env=env, stdout=stream,
                                          stderr=subprocess.STDOUT, start_new_session=True)
        write(self.root / 'model_process.json', {'pid': self.model.pid, 'spec': spec})
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 300
            while True:
                self.check_cancel()
                if self.model.poll() is not None:
                    raise RuntimeError('X-VLA模型加载失败，请查看 model.log。')
                try:
                    with opener.open('http://127.0.0.1:8794/openapi.json', timeout=2) as response:
                        if 'post' in json.load(response)['paths']['/act']:
                            break
                except (OSError, ValueError, KeyError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('X-VLA模型加载超时。')
                self.cancel.wait(1)
            from scene_demo.xvla_service import XvlaSceneService
            self.svc = XvlaSceneService(str(self.root / 'runs'), expected_capabilities=None, grounding_enabled=True)
            self.svc.start()
            deadline = time.monotonic() + 60
            while not self.svc.health().get('ready'):
                self.check_cancel()
                if self.svc.health().get('worker_error') or time.monotonic() > deadline:
                    raise RuntimeError('场景服务启动失败。')
                self.cancel.wait(.25)
        except Exception:
            self.close_runtime()
            raise

    def close_runtime(self):
        if self.svc:
            self.svc.stop()
            self.svc = None
        if self.model and self.model.poll() is None:
            self.model.terminate()
            try:
                self.model.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.model.kill()
                self.model.wait()
        self.model = None

    def preview_scene(self):
        """Render the same checker rig on the existing worker, without a model call."""
        sid = self.session['session_id']
        def capture_preview():
            from scene_demo.xvla_grounding import attach_checker_cameras
            from scene_demo.xvla_process_evidence import capture_frames
            from scene_demo.finish_v1.sensors import SensorPort
            record = self.svc._sessions[sid]
            port = attach_checker_cameras(SensorPort(self.svc._env, record.home_reference,
                                                    record.run_dir / 'scene_preview'))
            capture = port.capture()
            capture_frames(capture)  # Require all three synchronized real views.
            write(self.directory / 'preview_capture.json', capture)
            return {'ok': True, 'frames': frames(capture)}
        result = self.svc._sync_work('scene_preview', capture_preview, timeout=60.)
        if result.get('ok') is not True:
            raise RuntimeError('场景三视图拍摄失败：' + json.dumps(result, ensure_ascii=False))
        return result['frames']

    def load(self, scene):
        if scene not in {s['id'] for s in catalog_data()}:
            raise ValueError('请选择场景。')
        self.directory = self.root / (time.strftime('%Y%m%d-%H%M%S-') + secrets.token_hex(4))
        self.directory.mkdir(parents=True)
        self.session = self.draft = None
        with self.lock:
            self.state = {'mode': 'live', 'stage': 'preflight', 'request': '', 'scene': scene,
                          'request_id': 'console-' + secrets.token_hex(12), 'events': [], 'run_path': str(self.directory)}
        self.preflight(scene)
        self.check_cancel()
        self.ensure_runtime()
        self.check_cancel()
        self.update(stage='loading', message='创建初始场景（含原流程的10步静置）。')
        result = self.svc.create_session(scene, seed=142, init_state_index=0)
        if result.get('ok') is not True:
            raise RuntimeError(json.dumps(result, ensure_ascii=False))
        self.session = result.get('session', result)
        if not self.session.get('session_id'):
            raise RuntimeError('未取得新场景编号')
        self.session = self.svc.session(self.session['session_id'])
        write(self.directory / 'initial_session.json', self.session)
        self.check_cancel()
        self.update(message='正在拍摄场景三视图；尚未提交任务，也不调用识别或规划模型。')
        preview = self.preview_scene()
        self.check_cancel()
        self.update(stage='ready', message='三视图已就绪。请先看看场景，再在下方填写你想做的任务。',
                    session_id=self.session['session_id'], capabilities=self.session['capabilities'],
                    frames=preview, preview_only=True)

    def plan(self, request):
        if self.state.get('stage') != 'ready' or not self.session:
            raise ValueError('请先载入新场景。每次尝试只规划一次。')
        request = task_text(request)
        rid, sid = self.state['request_id'], self.session['session_id']
        write(self.directory / 'task_request.json', {'request_id': rid, 'session_id': sid,
              'scene_version': self.session['scene_version'], 'request': request, 'submitted_at': time.time()})
        self.update(stage='recognizing', request=request, message='任务已提交，开始三视图联合识别与复核；此阶段不执行任务动作。')
        result = self.svc.bind_request_context(rid, sid, self.state['request'])
        if result.get('ok') is not True:
            raise RuntimeError(json.dumps(result, ensure_ascii=False))
        self.check_cancel()
        self.session = self.svc.session(sid)
        capture = self.svc._sessions[sid].planning_capture
        write(self.directory / 'planning_session.json', self.session)
        self.update(stage='planning', message='Hermes 正在查看三视图与物品记录，选择动作；提交的计划只保存为草稿。',
                    frames=frames(capture), inventory=self.session['scene_inventory'], preview_only=False)
        import yaml
        from scene_demo import run_agent, setup_profile
        profile_home = HOME / 'console_profiles' / self.directory.name
        profile_home.mkdir(parents=True, mode=0o700)
        shutil.copyfile(HOME / 'hermes_home/.env', profile_home / '.env')
        (profile_home / '.env').chmod(0o600)
        config = yaml.safe_load(setup_profile.CONFIG_YAML)
        config['mcp_servers']['scene_tools']['env'].update(
            SCENE_SERVICE_URL=self.url + '/bridge/' + self.bridge_token,
            SCENE_BUDGET_PER_SUBGOAL='800' if self.session['scene_id'] == 'goal_table' else '900')
        (profile_home / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
        (profile_home / 'SOUL.md').write_text(setup_profile.SOUL_MD, encoding='utf-8')
        prompt = run_agent.build_initial_prompt(self.session, self.state['request'], rid,
                                               [str(self.session['planning_image'])])
        (self.directory / 'prompt.txt').write_text(prompt, encoding='utf-8')
        runner = run_agent.HermesRunner(home=str(profile_home), cwd=str(HERE), cancel_check=self.cancel.is_set)
        result = runner(prompt, self.session['planning_image'], str(self.directory / 'usage.json'), 600)
        (self.directory / 'hermes.log').write_bytes(result.get('output') or b'')
        self.check_cancel()
        if result.get('exit_code') != 0 or not self.draft:
            raise RuntimeError('Hermes未生成完整草稿：' + str(result.get('error') or '请查看 hermes.log'))
        self.update(stage='draft', draft=self.draft, message='规划完成，尚未执行。请核对目标与能力范围，再点击执行。')

    def capture_plan(self, payload):
        with self.lock:
            if self.state.get('stage') != 'planning' or self.cancel.is_set() or self.draft is not None:
                raise ValueError('当前不接收新草稿；每次尝试只接收一次有效计划。')
            self.draft = validate_draft(self.session, payload, self.state['request_id'])
            write(self.directory / 'draft.json', self.draft)
            return {'ok': True, **copy.deepcopy(self.draft), 'state': 'queued',
                    'execution_deferred': True, 'detail': '已保存草稿；必须等待用户点击执行。'}

    def execute(self):
        with self.lock:
            if self.state.get('stage') != 'draft' or not self.draft or self.draft['decision'] != 'execute':
                raise ValueError('没有可执行的草稿。')
            self.check_cancel()
            payload = copy.deepcopy(self.draft)
            current = self.svc.session(payload['session_id'])
            validate_draft(current, payload, self.state['request_id'])
            self.update(stage='executing', message='用户已确认，正在将同一计划交给原 X-VLA 流程。')
            write(self.directory / 'user_execution_confirmation.json', {'time': time.time(), 'payload': payload})
            submitted = self.svc.submit_plan(payload)
            if submitted.get('ok') is not True:
                raise RuntimeError(json.dumps(submitted, ensure_ascii=False))
        deadline = time.monotonic() + 2400
        cancellation_sent = False
        while True:
            plan = self.svc.plan(payload['request_id'])
            if not plan or plan.get('session_id') != payload['session_id']:
                raise RuntimeError('未能取得本次计划状态；请勿重复执行。')
            jobs = [self.svc.job(j) for j in plan.get('job_ids', [])]
            jobs = [j for j in jobs if j and j.get('request_id') == payload['request_id'] and j.get('session_id') == payload['session_id']]
            self.update(plan=plan, jobs=jobs, evidence=self.evidence(jobs))
            if plan['state'] in TERMINAL:
                message = {'completed': '本次动作计划已结束。', 'blocked': '流程被检查器或复位验收阻断。',
                           'error': '执行发生错误；未自动重试。', 'cancelled': '后端已确认停止。'}[plan['state']]
                self.update(stage=plan['state'], message=message)
                return
            if self.cancel.is_set() or time.monotonic() > deadline:
                if not cancellation_sent:
                    result = self.svc.cancel_request(payload['request_id'], payload['session_id'])
                    if result.get('ok') is not True:
                        # Keep tracking real execution; never present an unconfirmed stop.
                        self.update(stage='stopping', message='停止尚未被后端接受，继续跟踪真实状态。')
                    else:
                        cancellation_sent = True
                        self.update(stage='stopping', message='停止请求已发送，等待后端确认；不会自动归位。')
            time.sleep(1)

    def evidence(self, jobs):
        output = []
        for job in jobs:
            directory = Path(job['run_dir'])
            for path in sorted(directory.rglob('process_events.jsonl')):
                try:
                    lines = path.read_text(encoding='utf-8').splitlines()
                except OSError:
                    continue
                for line in lines:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue  # a writer may not yet have finished the final line
                    output.append({'capability': job['capability_id'], 'event': event})
        return output

    def stop(self):
        with self.lock:
            self.cancel.set()
            if self.busy:
                self.update(message='已请求停止，等待当前调用或动作确认结束。')
            elif self.state.get('mode') == 'live':
                self.update(stage='cancelled', message='已放弃当前草稿；没有新增任务动作。')

    def unload(self):
        self.close_runtime()
        self.session = self.draft = None
        self.update(stage='cancelled', message='已释放本工作台的模型与仿真资源。历史记录保留。')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, value, status=200):
        blob = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(blob)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        console = self.server.console
        if not self.local_host():
            return self.send({'error': 'local access only'}, 403)
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path == '/':
                blob = (HERE / 'xvla_console.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
            elif path == '/api/bootstrap':
                self.send({'token': console.token, 'examples': EXAMPLES, 'scenes': catalog_data(),
                           'app': 'xvla_interactive_console_v1'})
            elif path == '/api/state':
                self.send(console.snapshot())
            elif path == '/api/export':
                self.send(console.snapshot())
            elif path.startswith('/bridge/' + console.bridge_token + '/'):
                part = path.split('/')[3:]
                with console.lock:
                    if part == ['sessions', (console.session or {}).get('session_id')]:
                        return self.send(console.session)
                    if console.draft and part == ['plans', console.draft['request_id']]:
                        return self.send({**console.draft, 'ok': True, 'state': 'queued', 'execution_deferred': True})
                self.send({'ok': False, 'reason': 'unknown_request'}, 404)
            elif path == '/media':
                source = Path(urllib.parse.parse_qs(parsed.query).get('path', [''])[0]).resolve()
                if not source.is_relative_to(RESULTS.resolve()) or source.suffix.lower() not in ('.png', '.mp4'):
                    raise ValueError('图片或视频路径不在允许的实验记录目录内。')
                size = source.stat().st_size
                start, end, status = 0, size - 1, 200
                if self.headers.get('Range'):
                    import re
                    match = re.fullmatch(r'bytes=(\d+)-(\d*)', self.headers['Range'])
                    if not match:
                        return self.send({'error': 'invalid range'}, 416)
                    start, end = int(match[1]), min(int(match[2]) if match[2] else size - 1, size - 1)
                    if start > end:
                        return self.send({'error': 'invalid range'}, 416)
                    status = 206
                self.send_response(status)
                self.send_header('Content-Type', mimetypes.guess_type(source)[0])
                self.send_header('Content-Length', str(end - start + 1))
                self.send_header('Accept-Ranges', 'bytes')
                if status == 206:
                    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.end_headers()
                with source.open('rb') as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = stream.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            else:
                self.send({'error': 'not found'}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.send({'error': str(exc)}, 400)

    def do_POST(self):
        console = self.server.console
        try:
            if not self.local_host():
                return self.send({'error': 'local access only'}, 403)
            n = int(self.headers.get('Content-Length', '0'))
            if not 0 < n <= 65536:
                raise ValueError('invalid body size')
            payload = json.loads(self.rfile.read(n))
            if self.path == '/bridge/' + console.bridge_token + '/plans':
                return self.send(console.capture_plan(payload), 202)
            if self.path == '/bridge/' + console.bridge_token + '/observe':
                with console.lock:
                    if not console.session or payload.get('session_id') != console.session['session_id']:
                        raise ValueError('unknown session')
                    return self.send(console.session)
            if self.headers.get('X-Console-Token') != console.token:
                return self.send({'error': '请刷新页面后重试。'}, 403)
            origin = self.headers.get('Origin')
            if origin and urllib.parse.urlparse(origin).netloc != self.headers.get('Host'):
                return self.send({'error': 'foreign origin'}, 403)
            if self.path == '/api/history':
                console.history(payload['id'])
            elif self.path == '/api/load':
                if console.state.get('mode') == 'live' and console.state.get('stage') in ('ready', 'draft'):
                    raise ValueError('请先停止当前尝试，再创建新场景。')
                console.start(lambda: console.load(payload.get('scene')))
            elif self.path == '/api/plan':
                request = task_text(payload.get('request'))
                with console.lock:
                    if payload.get('request_id') != console.state.get('request_id') or console.state.get('stage') != 'ready':
                        raise ValueError('请先载入场景；页面过期时请刷新。')
                    console.start(lambda: console.plan(request))
            elif self.path == '/api/execute':
                # The browser must send the exact reviewed request, never "latest".
                with console.lock:
                    if payload.get('request_id') != console.state.get('request_id'):
                        raise ValueError('页面请求已过期，请刷新核对。')
                    console.start(console.execute)
            elif self.path == '/api/stop':
                if payload.get('request_id') != console.state.get('request_id'):
                    raise ValueError('页面请求已过期，请刷新核对。')
                console.stop()
            elif self.path == '/api/unload':
                console.start(console.unload)
            else:
                return self.send({'error': 'unknown endpoint'}, 404)
            self.send({'ok': True})
        except Exception as exc:
            self.send({'error': str(exc)}, 400)

    def local_host(self):
        return self.headers.get('Host', '').split(':')[0] in ('127.0.0.1', 'localhost')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8083)
    parser.add_argument('--launch', action='store_true', help='Start the UI only in background; no model or simulator.')
    args = parser.parse_args()
    if args.launch:
        url = f'http://127.0.0.1:{args.port}'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with socket.socket() as sock:
            sock.settimeout(1)
            occupied = sock.connect_ex(('127.0.0.1', args.port)) == 0
        if occupied:
            with opener.open(url + '/api/bootstrap', timeout=3) as response:
                if json.load(response).get('app') != 'xvla_interactive_console_v1':
                    raise RuntimeError('端口被其他服务占用')
            print(url)
            return
        output = RESULTS / 'interactive-console'
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'web.log').open('ab') as stream:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--port', str(args.port)],
                                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        write(output / 'web_process.json', {'pid': proc.pid, 'url': url})
        for _ in range(30):
            try:
                with opener.open(url + '/api/bootstrap', timeout=1) as response:
                    if json.load(response).get('app') == 'xvla_interactive_console_v1':
                        print(url)
                        return
            except OSError:
                pass
            if proc.poll() is not None:
                break
            time.sleep(.25)
        raise RuntimeError('网页未就绪，请查看 interactive-console/web.log')
    # The frozen successful launch environment; does not launch its commands.
    os.environ.update(read(PREVIOUS / 'runtime/table_tidy_stove_environment.json'))
    for path in os.environ['PYTHONPATH'].split(':'):
        if path not in sys.path:
            sys.path.insert(0, path)
    console = Console()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.console = console
    console.url = f'http://127.0.0.1:{args.port}'
    print(console.url, flush=True)
    def shutdown(*_):
        console.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.serve_forever()
    finally:
        console.stop()
        console.close_runtime()
        server.server_close()


if __name__ == '__main__':
    main()
