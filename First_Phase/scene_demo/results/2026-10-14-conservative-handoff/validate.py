"""Save software-test output and protected controller comparisons."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[3]
base=root/'First_Phase/scene_demo'
snap=base/'results/interactive-console/20261014-130348-6b2083a7/source_snapshot'
protected=['finish_v1/control.py','finish_v1/runtime.py','finish_v1/sensors.py','finish_v1/geometry.py','joint_home.py']
hashes=[]
for name in protected:
    before=hashlib.sha256((snap/name).read_bytes()).hexdigest()
    after=hashlib.sha256((base/name).read_bytes()).hexdigest()
    hashes.append({'file':name,'before':before,'after':after,'unchanged':before==after})
(out/'protected_controls.json').write_text(json.dumps(hashes,indent=2))
for source in Path('/tmp/pytest-of-yhwang/pytest-2').glob('test_real_monitor_constructs_w*/finish/model_requests/*failure.json'):
    dest=out/'test_network_failures'/source.parents[2].name/source.name
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
tests=['test_xvla_relation_gate','test_xvla_subtask_review','test_xvla_grounding','test_xvla_process_monitor',
       'test_xvla_full_checks','test_xvla_integration','test_xvla_pipeline','test_xvla_trial_evidence']
argv=[sys.executable,'-m','pytest',*[f'scene_demo/tests/{t}.py' for t in tests],'-q']
result=subprocess.run(argv,cwd=root/'First_Phase',capture_output=True)
(out/'software.stdout').write_bytes(result.stdout);(out/'software.stderr').write_bytes(result.stderr)
(out/'software.json').write_text(json.dumps({'argv':argv,'exit_code':result.returncode,
    'initial_integration_fixture_error':'11 invalid-image HTTP400 attempts before fixture isolation; no robot actions or successful vision responses'},indent=2))
print(result.stdout.decode(),flush=True)
assert result.returncode==0 and all(h['unchanged'] for h in hashes)
