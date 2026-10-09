import hashlib, json, subprocess, time
from pathlib import Path

TMP = Path(r'D:\FYP\First_Phase\tmp')
CODE = Path(r'D:\FYP\First_Phase\scene_demo')
OUTPUT = TMP / 'collision_handoff_v1'
BASELINE = Path(r'\\wsl.localhost\Ubuntu\home\yhwang\fyp\scene_demo\grasp_assist\2026-10-09-collision-probe-v1\probe.json')
PROOF = TMP / 'collision_handoff_v1_run_proof.json'
LOG = TMP / 'collision_handoff_v1_run.log'
NAMES = ('collision_grasp.py', 'collision_geometry.py', 'side_grasp.py', 'local_grasp.py', 'preparation_diagnostics.py', 'service.py', 'placement_experiments.py', 'guard_validation.py', 'paired_config_experiments.py', 'side_grasp_pilot.py', 'catalog.py')
INPUTS = (CODE / 'results/2026-10-09-side-grasp/pilot/case_01/job_01/action_sources.jsonl.gz', CODE / 'results/2026-10-09-side-grasp/pilot/case_01/job_02/action_sources.jsonl.gz', CODE / 'results/2026-10-09-side-grasp/pilot/case_01/job_02/wine_telemetry.jsonl.gz')

INPUTS = INPUTS + (BASELINE, CODE / 'results/2026-10-09-side-grasp/pilot/case_01/job_02/grasp_assist.json')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def build_argv():
    argv = ['wsl.exe', '-d', 'Ubuntu', '--exec', 'env']
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'): argv += ['-u', name]
    return argv + ['MUJOCO_GL=egl', 'LIBERO_CONFIG_PATH=/home/yhwang/fyp/libero_demo/libero_config',
                   'LD_LIBRARY_PATH=/usr/lib/wsl/lib', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
                   'PYTHONPATH=/home/yhwang/fyp/vla/lerobot/src:/mnt/d/FYP/First_Phase/scene_demo',
                   '/home/yhwang/fyp/libero_demo/venv/bin/python', '-u', '/mnt/d/FYP/First_Phase/tmp/collision_handoff_v1_run.py']

def main():
    assert not OUTPUT.exists() and not PROOF.exists() and not LOG.exists(), 'existing candidate output/proof/log; no retry'
    record = {'status': 'RUNNING', 'argv': build_argv(), 'started_epoch': time.time(), 'candidate_attempts': 1,
              'script_sha256': sha(TMP / 'collision_handoff_v1_run.py'), 'source_sha256': {name: sha(CODE / name) for name in NAMES},
              'input_sha256': {str(path): sha(path) for path in INPUTS}, 'model_calls': 'actual continuation count in final report', 'hermes_calls': 0}
    def save(): PROOF.write_text(json.dumps(record, indent=2), encoding='utf-8')
    with LOG.open('xb') as log:
        log.write((json.dumps(record['argv']) + '\n').encode()); log.flush()
        child = subprocess.Popen(record['argv'], stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        record['owned_wsl_windows_pid'] = child.pid; save(); print(json.dumps(record), flush=True)
        record['cli_exit'] = child.wait()
    record['finished_epoch'] = time.time()
    record['sources_unchanged'] = all(sha(CODE / name) == value for name, value in record['source_sha256'].items())
    record['inputs_unchanged'] = all(sha(path) == record['input_sha256'][str(path)] for path in INPUTS)
    if (OUTPUT / 'report.json').exists():
        raw = (OUTPUT / 'report.json').read_bytes(); report = json.loads(raw)
        record.update({'report_sha256': hashlib.sha256(raw).hexdigest(), 'report_bytes': len(raw),
                       'candidate_confirmed': report.get('candidate_confirmed'), 'phase': report.get('phase'), 'reason': report.get('reason'),
                       'aux_actions': report.get('aux_actions'), 'physical_prefix_steps': report.get('physical_prefix_steps'), 'placement_success': report.get('placement_success'), 'post_handoff_vla_actions': report.get('post_handoff_vla_actions'),
                       'operational_errors': report.get('operational_errors'), 'report_ok': report.get('ok')})
    record['status'] = 'CANDIDATE_SINGLE_ATTEMPT_COMPLETE' if record['cli_exit'] == 0 and record['sources_unchanged'] and record['inputs_unchanged'] else 'CANDIDATE_OPERATIONAL_ERROR'
    save(); print(json.dumps(record), flush=True)
    return record['cli_exit'] or (0 if record['status'] == 'CANDIDATE_SINGLE_ATTEMPT_COMPLETE' else 2)

if __name__ == '__main__': raise SystemExit(main())
