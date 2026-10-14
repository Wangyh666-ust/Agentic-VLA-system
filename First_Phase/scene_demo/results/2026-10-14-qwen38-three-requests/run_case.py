"""Use production scene service and Hermes runner with the literal user request."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path('/mnt/d/FYP')
HERE = ROOT / 'First_Phase/scene_demo'
OUT = Path(__file__).resolve().parent
PROPOSAL = HERE / 'plans/20261014-qwen38-three-requests.json'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    name = parser.parse_args().case
    value = json.loads(PROPOSAL.read_text())
    spec = value['configuration']['cases'][name]
    for relative, expected in value['configuration']['source_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
        assert hashlib.sha256((OUT / 'source_snapshot' / relative).read_bytes()).hexdigest() == expected
    directory = OUT / name
    directory.mkdir(exist_ok=False)
    check = subprocess.run([sys.executable, str(ROOT / '.agents/skills/fyp-experiment-review/scripts/review.py'),
                            'check', '--proposal', str(PROPOSAL)], capture_output=True)
    (directory / 'preflight.stdout').write_bytes(check.stdout)
    (directory / 'preflight.stderr').write_bytes(check.stderr)
    if check.returncode:
        dump(directory / 'result.json', {'status': 'preflight_failed', 'physical_actions': 0})
        return 2
    os.environ.update(SCENE_SERVICE_URL='http://127.0.0.1:8769', SCENE_BUDGET_PER_SUBGOAL=str(spec['budget_per_subgoal']),
                      SCENE_EXPECTED_WORKFLOW='persistent_scene_xvla_v3',
                      SCENE_EXPECTED_MODEL_REVISION='129e71460678b7236cee6fc9707f09d9fa0c3590', SCENE_PLANNER_MODEL='qwen3.8-max')
    import yaml
    from scene_demo import setup_profile, service, run_agent
    from scene_demo.xvla_service import XvlaSceneService
    home = Path('/home/yhwang/fyp/xvla_native/hermes_home')
    shutil.copyfile(setup_profile.SRC_ENV, home / '.env')
    (home / '.env').chmod(0o600)
    profile = yaml.safe_load(setup_profile.CONFIG_YAML)
    profile['mcp_servers']['scene_tools']['env'].update(SCENE_SERVICE_URL=os.environ['SCENE_SERVICE_URL'],
                                                       SCENE_BUDGET_PER_SUBGOAL=str(spec['budget_per_subgoal']))
    (home / 'config.yaml').write_text(yaml.safe_dump(profile, allow_unicode=True), encoding='utf-8')
    (home / 'SOUL.md').write_text(setup_profile.SOUL_MD, encoding='utf-8')
    run_agent.SERVICE_BASE = os.environ['SCENE_SERVICE_URL']
    run_agent.EXPECTED_WORKFLOW = os.environ['SCENE_EXPECTED_WORKFLOW']
    run_agent.EXPECTED_MODEL_REVISION = os.environ['SCENE_EXPECTED_MODEL_REVISION']
    svc = XvlaSceneService(str(directory / 'runs'), expected_capabilities=None, grounding_enabled=True)
    httpd = None
    try:
        svc.start()
        startup_deadline = time.monotonic() + 60
        while not svc.health().get('ready'):
            if time.monotonic() > startup_deadline or svc.health().get('worker_error'):
                raise RuntimeError('service_not_ready')
            time.sleep(.25)
        httpd = service._Server(('127.0.0.1', 8769), service._Handler)
        httpd.service = svc
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        deadline = time.monotonic() + 1800
        created = svc.create_session(spec['scene'], seed=142, init_state_index=0)
        if created.get('ok') is False or created.get('state') != 'ready':
            raise RuntimeError('session_create_failed: ' + json.dumps(created))
        session_id = created['session']['session_id'] if 'session' in created else created['session_id']
        request_id = 'qwen38-' + name
        bound = svc.bind_request_context(request_id, session_id, spec['request'])
        if bound.get('ok') is False:
            raise RuntimeError('request_bind_failed')
        config = SimpleNamespace(session_id=session_id, request=spec['request'], case_id=None, request_id=request_id,
                                 timeout=max(1, int(deadline - time.monotonic())), max_repairs=0, cancel_file=None)
        (directory / 'agent').mkdir()
        agent = run_agent.Runner(config, run_agent.ServiceClient(), run_agent.HermesRunner(home=str(home), cwd=str(HERE)),
                                run_agent.SystemClock(), str(directory / 'agent')).run()
        dump(directory / 'agent/agent_result.json', agent)
        plan = agent.get('plan') or {}
        pilots = [json.loads(p.read_text()) for p in sorted((directory / 'runs').rglob('pilot_result.json'))]
        independent = []
        for goal in spec['intent_goals']:
            try:
                raw = service.eval_goal_predicate(svc._env, goal)
                independent.append({'goal': goal, 'value': None if raw is None else bool(raw)})
            except Exception as exc:
                independent.append({'goal': goal, 'value': None, 'error_type': type(exc).__name__})
        result = {'request': spec['request'], 'scene': spec['scene'], 'agent_result': agent, 'plan': plan, 'jobs': pilots,
                  'independent_user_goals': independent, 'status': 'finished',
                  'pipeline_terminal': plan.get('state'), 'decision': plan.get('decision'),
                  'physical_user_goal_success': all(g['value'] is True for g in independent) if independent else None}
        dump(directory / 'result.json', result)
        print(json.dumps({'case': name, 'decision': result['decision'], 'state': plan.get('state'),
                          'capabilities': plan.get('capability_ids'), 'physical_user_goal_success': result['physical_user_goal_success']}, ensure_ascii=False))
        return 0
    except Exception as exc:
        dump(directory / 'result.json', {'request': spec['request'], 'scene': spec['scene'],
                                         'status': 'program_error', 'error': str(exc)})
        raise
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        svc.stop()


if __name__ == '__main__':
    raise SystemExit(main())
