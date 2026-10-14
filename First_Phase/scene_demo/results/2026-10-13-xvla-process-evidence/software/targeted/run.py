import json,subprocess
from pathlib import Path
out=Path('/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-13-xvla-process-evidence/software/targeted/attempt3')
environment=['PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/mnt/d/FYP:/mnt/d/FYP/First_Phase:/mnt/d/FYP/First_Phase/scene_demo/finish_v1:/home/yhwang/fyp/xvla_native/LIBERO:/home/yhwang/fyp/xvla_native/upstream','LIBERO_CONFIG_PATH=/home/yhwang/fyp/xvla_native/libero_config','MUJOCO_GL=egl','PYOPENGL_PLATFORM=egl','LD_LIBRARY_PATH=/usr/lib/wsl/lib','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1']
tests=['test_xvla_process_evidence.py','test_xvla_process_monitor.py','test_xvla_finish.py','test_xvla_full_checks.py','test_xvla_integration.py','test_xvla_pipeline.py']
argv=['env']+environment+['/home/yhwang/fyp/xvla_native/sim_venv/bin/python','-m','pytest','-q','-p','no:cacheprovider']+['First_Phase/scene_demo/tests/'+t for t in tests]
(out/'argv.json').write_text(json.dumps({'cwd':'/mnt/d/FYP','argv':argv},indent=2)+'\n');r=subprocess.run(argv,cwd='/mnt/d/FYP',capture_output=True);(out/'stdout').write_bytes(r.stdout);(out/'stderr').write_bytes(r.stderr);(out/'exitcode').write_text(str(r.returncode)+'\n');print(r.stdout.decode());print(r.stderr.decode());raise SystemExit(r.returncode)
