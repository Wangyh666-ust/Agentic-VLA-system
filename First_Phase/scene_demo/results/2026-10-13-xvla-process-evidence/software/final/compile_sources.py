from pathlib import Path
files=['xvla_process_evidence.py','xvla_monitor.py','xvla_finish.py','service.py','xvla_service.py','xvla_pilot.py','tests/test_xvla_finish.py','tests/test_xvla_integration.py','tests/test_xvla_process_evidence.py','tests/test_xvla_process_monitor.py']
for name in files:
 p=Path('First_Phase/scene_demo')/name
 compile(p.read_bytes(),str(p),'exec')
 assert all(line.rstrip(b' \t')==line for line in p.read_bytes().splitlines()),name
print('10 sources compiled; real trailing space/tab lines: 0')
