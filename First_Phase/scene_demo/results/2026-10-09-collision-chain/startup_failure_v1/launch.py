import hashlib
import json
from pathlib import Path
import subprocess
import time

TMP = Path(r'D:\FYP\First_Phase\tmp')
CODE = Path(r'D:\FYP\First_Phase\scene_demo')
OUT = TMP / 'collision_chain_v1'
HOME = TMP / 'collision_chain_hermes_v1'
CAL = Path(r'\\wsl.localhost\Ubuntu\home\yhwang\fyp\scene_demo\grasp_assist\2026-10-09-side-calibration-v1\calibration.json')
CONFIRM = CODE / 'results/2026-10-09-collision-approach/candidate_v2/report.json'
SOFTWARE = TMP / 'collision_chain_wrapper_acceptance.json'
PREFLIGHT = TMP / 'collision_chain_preflight.json'
PREPARE = TMP / 'collision_chain_v1_launch_prepare.json'
PROOF = TMP / 'collision_chain_v1_run_proof.json'
LOG = TMP / 'collision_chain_v1_run.log'
SOURCE_SHA = '1113abbf99c8f51953aebfea9af6f4390ec87bdd380ce85eb21efa8b84fc6354'
NAMES = ('collision_grasp_pilot.py', 'side_grasp_pilot.py', 'grasp_assist_service.py',
         'placement_experiments.py', 'service.py', 'collision_grasp.py',
         'collision_geometry.py', 'side_grasp.py', 'local_grasp.py')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def argv():
    command = ['wsl.exe', '-d', 'Ubuntu', '--exec', 'env']
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        command.extend(['-u', name])
    return command + [
        'MUJOCO_GL=egl', 'LIBERO_CONFIG_PATH=/home/yhwang/fyp/libero_demo/libero_config',
        'LD_LIBRARY_PATH=/usr/lib/wsl/lib', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
        'PYTHONPATH=/home/yhwang/fyp/vla/lerobot/src:/mnt/d/FYP/First_Phase/scene_demo',
        '/home/yhwang/fyp/libero_demo/venv/bin/python', '-u',
        '/mnt/d/FYP/First_Phase/scene_demo/collision_grasp_pilot.py',
        '--output-dir', '/mnt/d/FYP/First_Phase/tmp/collision_chain_v1',
        '--hermes-home', '/mnt/d/FYP/First_Phase/tmp/collision_chain_hermes_v1',
        '--calibration', '/home/yhwang/fyp/scene_demo/grasp_assist/2026-10-09-side-calibration-v1/calibration.json',
        '--collision-confirmation', '/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-09-collision-approach/candidate_v2/report.json']

def main():
    software = json.loads(SOFTWARE.read_text(encoding='utf-8'))
    preflight = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    prepared = json.loads(PREPARE.read_text(encoding='utf-8'))
    assert software['status'] == 'COLLISION_CHAIN_WRAPPER_SOFTWARE_PASS'
    assert preflight['ok'] is True and preflight['status'] == 'COLLISION_CHAIN_PREFLIGHT_PASS'
    assert prepared['ok'] is True and prepared['launcher_sha256'] == sha(__file__)
    assert sha(CODE/'collision_grasp_pilot.py') == SOURCE_SHA == software['source_sha256']
    hashes = {name:sha(CODE/name) for name in NAMES}
    assert hashes == prepared['source_hashes'], 'Source differs from reviewed preparation'
    assert all(hashes[name] == digest for name,digest in software['dependency_sha256_after'].items())
    inputs = {'calibration':sha(CAL), 'confirmation':sha(CONFIRM)}
    assert inputs == prepared['input_hashes'], 'Frozen input differs'
    for path in (OUT, HOME, LOG, PROOF):
        assert not path.exists(), 'Refusing existing path: '+str(path)
    commands = []
    for args in (['rev-parse','HEAD'], ['branch','--show-current']):
        command = ['git','-C',r'D:\FYP',*args]
        result = subprocess.run(command,capture_output=True,timeout=30)
        commands.append({'argv':command,'exit':result.returncode,
                         'stdout':result.stdout.decode('utf-8'),'stderr':result.stderr.decode('utf-8')})
        assert result.returncode == 0
    proof = {'status':'RUNNING', 'started_epoch':time.time(), 'attempts':1, 'no_retry':True,
             'argv':argv(), 'repo_head':commands[0]['stdout'].strip(),
             'repo_branch':commands[1]['stdout'].strip(), 'commands':commands,
             'software_proof_path':str(SOFTWARE),'software_proof_sha256':sha(SOFTWARE),
             'preflight_path':str(PREFLIGHT),'preflight_sha256':sha(PREFLIGHT),
             'launcher_sha256':sha(__file__), 'source_hashes_before':hashes,
             'input_hashes_before':inputs, 'output_dir':str(OUT),
             'hermes_home_path':str(HOME), 'grasp_module':'collision_grasp',
             'controller':'collision_grasp.LocalGraspController'}
    def save():
        PROOF.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with LOG.open('xb') as log:
        log.write((json.dumps(argv())+'\nSTART\nRAW_STDOUT_STDERR\n').encode('utf-8'))
        log.flush()
        process = subprocess.Popen(argv(),stdout=log,stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        proof['owned_wsl_windows_pid'] = process.pid
        save()
        code = process.wait()
        log.write(('\nEND CLI_EXIT='+str(code)+'\n').encode('utf-8'))
    proof.update(cli_exit=code,finished_epoch=time.time(),status='CLI_COMPLETED_EVIDENCE_PRESERVED')
    after = {name:sha(CODE/name) for name in NAMES}
    inputs_after = {'calibration':sha(CAL),'confirmation':sha(CONFIRM)}
    proof.update(source_hashes_after=after,source_unchanged=after==hashes,
                 input_hashes_after=inputs_after,inputs_unchanged=inputs_after==inputs)
    report_path = OUT/'report.json'
    proof['report_exists'] = report_path.is_file()
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        proof.update(report_path=str(report_path),report_sha256=sha(report_path),
                     report_root_keys=list(report),
                     report_summary={key:report.get(key) for key in
                                     ('status','completed','planned_cases','executed_cases','fatal_error','stage_counts')})
        proof['case_outcomes'] = [{key:entry.get(key) for key in
                                 ('case_index','model_seed','request_text','actual_order','expected_order','order_ok',
                                  'agent','plan','score','jobs','operational_errors','physical_failures')}
                                for entry in report.get('cases',[])]
    prereg = OUT/'preregistration.json'
    if prereg.is_file():
        data = json.loads(prereg.read_text(encoding='utf-8'))
        proof['preregistration'] = {'path':str(prereg),'sha256':sha(prereg),
                                   'grasp_module':data.get('grasp_module'),'controller':data.get('controller')}
        proof['collision_metadata_matches'] = (data.get('grasp_module')=='collision_grasp' and
                                               data.get('controller')=='collision_grasp.LocalGraspController')
    save()
    print(json.dumps({key:proof.get(key) for key in
                     ('cli_exit','source_unchanged','inputs_unchanged','report_path','report_summary')},ensure_ascii=False))
    return code

if __name__ == '__main__':
    raise SystemExit(main())
