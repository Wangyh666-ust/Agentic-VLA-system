"""Saved three-view images -> real Qwen/Hermes -> capture-only plan, zero simulation."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / 'First_Phase/scene_demo'
OUT = Path(__file__).resolve().parent
OLD = HERE / 'results/2026-10-14-qwen38-three-requests'
PROPOSAL = HERE / 'plans/20261014-partial-scope-offline.json'
SOURCES = ['planning_scope.py', 'run_agent.py', 'setup_profile.py', 'mcp_server.py',
           'service.py', 'xvla_service.py', 'xvla_skills.py', 'xvla_grounding.py', 'catalog.py',
           'planning_diagnostics.py', 'tests/test_planning_scope.py']
PROTECTED = ['joint_home.py', 'safe_exit.py', 'safe_exit_contacts.py', 'xvla_finish.py',
             'xvla_backend.py', 'finish_v1/runtime.py', 'finish_v1/control.py',
             'finish_v1/vision.py', 'xvla_subtask_review.py']


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    assert not PROPOSAL.exists()
    cases = read(OLD / 'cases.json')
    inputs = {}
    for name, spec in cases.items():
        run = next((OLD / name / 'runs').iterdir())
        directory = run / 'planning_grounding'
        if (directory / 'capture.json').exists():
            capture = read(directory / 'capture.json')
            frames = capture['checker_frames']
        else:
            # Last native settling record is the original capture instant;
            # no state reconstruction, invented pixels, or simulator launch.
            row = json.loads((run / 'native/native_steps.jsonl').read_text().splitlines()[-1])
            frames = [dict(view=v, rgb_path=str(directory / f), timestamp=row['sim_time']) for v, f in (
                ('wrist', 'cap000002_wrist.png'), ('agentview', 'cap000002_wrist_checker_agentview.png'),
                ('sideview', 'cap000002_wrist_checker_sideview.png'))]
            capture = dict(timestamp=row['sim_time'], checker_frames=frames,
                           robot=dict(finger_qpos=row['finger_qpos'], arm_qvel=row['arm_qvel']),
                           provenance='frame paths from original planning capture; time/encoder from last settling row')
        frame_hashes = {}
        for frame in frames:
            path = Path(frame['rgb_path'])
            frame_hashes[str(path.relative_to(ROOT))] = digest(path)
        inputs[name] = dict(spec=spec, capture=capture, image_sha256=frame_hashes)
    hashes = {}
    for filename in SOURCES + PROTECTED:
        source = HERE / filename
        target = OUT / 'source_snapshot' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[filename] = digest(source)
    old_hashes = read(OLD / 'source_manifest.json')
    assert all(hashes[f] == old_hashes['First_Phase/scene_demo/' + f] for f in PROTECTED)
    dump(OUT / 'inputs.json', inputs)
    dump(OUT / 'source_manifest.json', hashes)
    prior = {
        '20261014-qwen38-three-requests': '复用三条原文及各自初始三视图；整理通过、黄杯准备选择合理但缺能力说明、双罐网络失败未规划。本轮不复用旧规划作为答案。',
        '20261014-qwen38-initial-recognition': '复用Qwen3.8原生视觉接口与离线180秒传输设置；不能代替这三条原文的规划验证。',
        '20261008-skill-context': '复用只收集计划且不执行的真实Hermes工具链；保留丢弃与收纳语义边界。',
    }
    proposal = dict(schema_version=1, id='20261014-partial-scope-offline', intent='analyze_existing',
        question_key='partial-scope-offline-planning',
        purpose='验证通用部分执行与能力边界：三条旧原文+原始初态照片，经当前识别、Hermes真实规划与选择复核；不启动仿真，线上须用户另行批准。',
        prior_evidence=list(prior), history_review=[dict(experiment_id=k, disposition='reuse', reason=v) for k, v in prior.items()],
        related_search='核对当前公开catalog、初识别inspect_scene、run_agent/SOUL/MCP提示、原始plan_during_execution和capture照片；无垃圾桶/加热启动能力。',
        new_physical_actions=0,
        analysis_steps=['固定原始图片哈希与原文，公开能力由当前catalog生成，隐藏预期答案不进入模型。',
            '每项两次Qwen3.8初识别复核、一次真实Hermes CLI规划；capture-only服务只记录，无VLA/环境/动作入口。',
            'execute时复用validate_selection与verify_plan，检查实际目标绑定和部分执行范围。',
            '人工核对整理三目标、黄杯partial及缺启动说明、丢弃不偷偷入收纳篮；通过后仅请求在线许可。'],
        success_criteria='三个实际提交均符合各自原意；黄杯partial分支真实触发且保留未支持加热；execute目标身份复核通过；未启动仿真。',
        failure_criteria='不提交、图片识别/选择失败、漏报能力缺口、擅自替代意图均不通过；不以零动作评判VLA。',
        scope=dict(max_cases=3, max_attempts_per_case=1, inventory_calls_per_case=2, selection_calls_per_case=1,
                   hermes_turns_per_case=12, stop_rule='每项独立一次；保存错误，不自动重试。任一不通过不申请在线启动。'),
        configuration=dict(model='qwen3.8-max', inventory_timeout_s=180, production_inventory_timeout_s=90,
            transport_note='仅离线传输等待180秒，沿用旧离线比较；生产90秒未改，不据此证明线上网络可靠。',
            hermes_timeout_s=900, sources=hashes, helper_actions=0, simulation_actions=0,
            model_authoring='root actual model/reasoning unavailable; no subagents', cases=list(inputs)))
    dump(PROPOSAL, proposal)
    print(json.dumps(dict(prepared=True, images=sum(len(x['image_sha256']) for x in inputs.values()), physical_actions=0)), flush=True)


def run_case(name):
    from scene_demo import catalog, run_agent, setup_profile, planning_diagnostics as diag
    from scene_demo import xvla_grounding as grounding
    from scene_demo.finish_v1 import vision
    from scene_demo.planning_scope import validate_request_scope
    from scene_demo.xvla_skills import route_capabilities
    import yaml

    spec = read(OUT / 'inputs.json')[name]
    for f, sha in read(OUT / 'source_manifest.json').items():
        assert digest(HERE / f) == sha and digest(OUT / 'source_snapshot' / f) == sha
    for relative, sha in spec['image_sha256'].items():
        assert digest(ROOT / relative) == sha
    directory = OUT / name
    directory.mkdir(exist_ok=False)
    request = spec['spec']['request']
    result = dict(request=request, physical_actions=0, vla_calls=0, simulator_instances=0)

    class OfflineInventory(grounding.InventoryClient):
        # Identical production prompt/options/body. Longer offline wait and
        # failure audit only; no change to production's 90-second transport.
        def _request(self, context, image_paths, prompt):
            assert self._calls < self.max_calls
            key = self._resolve_key()
            if not key:
                raise RuntimeError('Qwen key unavailable')
            content = [dict(type='text', text=prompt)]
            for f, url in zip(context['sensor_frames'], self._load_images(image_paths), strict=True):
                content.extend([dict(type='text', text=f"Image frame={f['frame']}; view={f['view']}; timestamp={f['timestamp']}"),
                                dict(type='image_url', image_url=dict(url=url))])
            body = dict(model=self.model, temperature=0, max_tokens=1600, messages=[dict(role='user', content=content)])
            body.update(self.request_options)
            raw_body = json.dumps(body).encode('utf-8')
            self._calls += 1
            index = self._calls
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            (self.audit_dir / f'call_{index:02d}_body.json').write_bytes(raw_body)
            dump(self.audit_dir / f'call_{index:02d}_attempt.json', dict(model=self.model, timeout_s=180,
                images=[dict(path=str(p), sha256=digest(Path(p))) for p in image_paths], started_at=time.time()))
            started = time.monotonic()
            print(json.dumps(dict(case=name, stage='qwen_request', call=index, images=len(image_paths))), flush=True)
            req = urllib.request.Request(self.base_url.rstrip('/') + '/chat/completions', data=raw_body,
                headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}, method='POST')
            try:
                with urllib.request.urlopen(req, timeout=180) as response:
                    status, raw = response.status, response.read()
            except Exception as exc:
                dump(self.audit_dir / f'call_{index:02d}_error.json', dict(error_type=type(exc).__name__,
                    error=str(exc), wall_s=time.monotonic()-started))
                raise
            self.record_response(context, image_paths, prompt, raw_body, raw, index, status)
            payload = json.loads(raw)
            self.raw_inventory = vision._parse_verdict_content(vision._extract_content(payload))
            meta = dict(model=payload.get('model'), usage=payload.get('usage'), wall_s=time.monotonic()-started,
                http_status=status, request_sha256=hashlib.sha256(raw_body).hexdigest(), response_sha256=hashlib.sha256(raw).hexdigest())
            dump(self.audit_dir / f'call_{index:02d}_meta.json', meta)
            return meta

    server = None
    try:
        client = OfflineInventory(env_path='/home/yhwang/fyp/xvla_native/hermes_home/.env', max_calls=2)
        inventory, picture = grounding.inspect_scene(spec['capture'], request, directory / 'grounding', client)
        sid, rid = uuid.uuid4().hex, uuid.uuid4().hex
        scene_id = spec['spec']['scene']
        scene = catalog.SCENES[scene_id]
        public = dict(ok=True, session_id=sid, scene_id=scene_id, state='ready', scene_version=0, total_steps=0,
            storage_policy=scene['storage_policy'], description=scene['description'],
            capabilities=catalog.scene_capabilities(scene_id, backend='xvla'), scene_inventory=inventory,
            planning_image=picture, latest_png=picture,
            images=[dict(view=f['view'], image_path=f['rgb_path']) for f in spec['capture']['checker_frames']])

        class Capture(diag.FakeSceneContext):
            def _submit_plan(self, body):
                try:
                    if not isinstance(body, dict) or body.get('request_scope') is None:
                        raise ValueError('request_scope required for this planner validation')
                    validate_request_scope(body['request_scope'], body.get('decision'))
                    ids = body.get('capability_ids')
                    available = {c['capability_id']: c for c in public['capabilities']}
                    if not isinstance(ids, list) or len(set(ids)) != len(ids) or any(c not in available for c in ids):
                        raise ValueError('invalid scene capabilities')
                    if body['decision'] == 'execute':
                        if not ids:
                            raise ValueError('execute requires capabilities')
                        grounding.validate_selection(inventory, body.get('target_bindings'), [available[c] for c in ids])
                    elif ids:
                        raise ValueError('non-execute must carry no capabilities')
                except (ValueError, TypeError) as exc:
                    return 400, dict(ok=False, reason='invalid_plan', detail=str(exc)), 'invalid_plan'
                status, queued, note = super()._submit_plan(body)
                if status == 200:
                    for key in ('request_scope', 'target_bindings'):
                        queued[key] = copy.deepcopy(body.get(key))
                        self._captures[rid][-1][key] = copy.deepcopy(body.get(key))
                        self._plans[rid][key] = copy.deepcopy(body.get(key))
                return status, queued, note

        context = Capture(public, str(directory / 'transport.jsonl'))
        context.begin_case(name, rid)
        port = {'heat_yellow_mug': 8778, 'table_tidy_stove': 8779, 'discard_two_cans': 8780}[name]
        server, thread = diag.serve_capture_server(context, port=port)
        # Ephemeral isolated profile outside evidence tree; credentials never
        # appear in repository artifacts, console output, or snapshots.
        home = Path('/home/yhwang/fyp/xvla_native/offline_scope_20261014') / name
        home.mkdir(parents=True, exist_ok=False)
        shutil.copyfile('/home/yhwang/fyp/xvla_native/hermes_home/.env', home / '.env')
        (home / '.env').chmod(0o600)
        config = yaml.safe_load(setup_profile.CONFIG_YAML)
        config['mcp_servers']['scene_tools']['env'].update(SCENE_SERVICE_URL=f'http://127.0.0.1:{port}',
            SCENE_BUDGET_PER_SUBGOAL=str(spec['spec']['budget_per_subgoal']))
        (home / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True))
        (home / 'SOUL.md').write_text(setup_profile.SOUL_MD)
        dump(directory / 'profile_public.json', config)
        dump(directory / 'session_public.json', public)
        paths = [x['image_path'] for x in public['images']]
        prompt = run_agent.build_initial_prompt(public, request, rid, paths)
        assert all(x not in prompt for x in ('intent_goals', 'allowed_decisions', 'goal_options'))
        (directory / 'prompt.txt').write_text(prompt)
        print(json.dumps(dict(case=name, stage='hermes_planning')), flush=True)
        result['hermes'] = run_agent.HermesRunner(home=str(home), cwd=str(HERE))(
            prompt, picture, str(directory / 'usage.json'), 900)
        (directory / 'hermes.log').write_bytes(result['hermes'].pop('output'))
        for log in (home / 'logs').glob('*'):
            if log.is_file():
                shutil.copyfile(log, directory / ('hermes_runtime_' + log.name))
        plans = context.captures_for(rid)
        dump(directory / 'submitted_plans.json', plans)
        result.update(plans=plans, protocol_errors=context.protocol_errors, forwarded=context.forwarded)
        if len(plans) != 1 or result['hermes']['exit_code'] != 0:
            raise RuntimeError('expected one successful Hermes plan submission')
        plan = plans[0]
        if plan['decision'] == 'execute':
            result['route'] = route_capabilities(scene_id, plan['capability_ids'],
                [c['capability_id'] for c in public['capabilities'] if c.get('skill_kind') == 'native_long_horizon'])
            result['selection_review'] = grounding.verify_plan(inventory, plan['target_bindings'], request,
                [catalog.CAPABILITIES[c] for c in result['route']['executed_capability_ids']], directory,
                client=OfflineInventory(env_path='/home/yhwang/fyp/xvla_native/hermes_home/.env', max_calls=1),
                request_scope=plan['request_scope'])
        result['status'] = 'captured'
    except Exception as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc))
    finally:
        if server:
            server.shutdown(); server.server_close()
        dump(directory / 'result.json', result)
    print(json.dumps(dict(case=name, status=result['status'], physical_actions=0), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--case', choices=['table_tidy_stove', 'heat_yellow_mug', 'discard_two_cans'])
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.case:
        run_case(args.case)
    else:
        parser.error('use --prepare or --case')
