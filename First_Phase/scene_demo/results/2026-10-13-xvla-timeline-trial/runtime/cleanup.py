"""Stop only the recorded campaign-owned model server after both cases exit."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

OUT = Path('/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial')
RUNTIME = OUT/'runtime'
assert (OUT/'basket_independent/runtime/worker.exitcode').exists(), 'basket worker still running'
record = json.loads((RUNTIME/'server_state.json').read_text())

def identity(pid):
    try:
        raw = Path('/proc/%d/stat'%pid).read_text()
        rest = raw[raw.rfind(')')+2:].split()
        return {'pid': pid, 'startticks': rest[19], 'state': rest[0], 'alive': rest[0] != 'Z',
                'argv': Path('/proc/%d/cmdline'%pid).read_bytes().decode().strip('\0').split('\0'),
                'cwd': os.readlink('/proc/%d/cwd'%pid)}
    except (OSError, ValueError):
        return {'pid': pid, 'alive': False}

def matches(current):
    return (record.get('owned') is True and record.get('pid') == 413 and record.get('startticks') == '474'
            and current.get('alive') is True and current.get('pid') == record['pid']
            and current.get('startticks') == record['startticks'] and current.get('argv') == record['argv']
            and current.get('cwd') == record['cwd'])

before = identity(record['pid'])
result = {'owned': record.get('owned'), 'before': before, 'recorded': record,
          'matched_identity': matches(before), 'signals': [], 'case_launches': {'microwave_mug': 1, 'basket_two': 1}}
if not before.get('alive'):
    result['reason'] = 'already_exited'
elif not matches(before):
    result['reason'] = 'identity_mismatch; no signal sent'
else:
    os.kill(record['pid'], signal.SIGTERM)
    result['signals'].append('SIGTERM')
    deadline = time.monotonic()+15
    while time.monotonic() < deadline and identity(record['pid']).get('alive'):
        time.sleep(.5)
    current = identity(record['pid'])
    if current.get('alive') and matches(current):
        os.kill(record['pid'], signal.SIGKILL)
        result['signals'].append('SIGKILL')
        deadline = time.monotonic()+5
        while time.monotonic() < deadline and identity(record['pid']).get('alive'):
            time.sleep(.5)
    result['reason'] = 'matched owned process signalled'
result['after'] = identity(record['pid'])
result['terminated'] = not result['after'].get('alive')
result['ports'] = []
for port in (8794, 8769):
    argv = ['ss', '-H', '-ltnp', 'sport = :'+str(port)]
    response = subprocess.run(argv, capture_output=True)
    result['ports'].append({'port': port, 'argv': argv, 'exitcode': response.returncode,
                           'stdout': response.stdout.decode(), 'stderr': response.stderr.decode(),
                           'listening': bool(response.stdout.strip())})
result['source_checks'] = {}
for name, directory in (('microwave_mug', OUT), ('basket_two', OUT/'basket_independent')):
    values = json.loads((directory/'runtime/freeze.after.json').read_text())
    result['source_checks'][name] = {'count': len(values), 'all_match': all(row['matches'] for row in values)}
(RUNTIME/'cleanup.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'terminated': result['terminated'], 'signals': result['signals'],
                  'ports': result['ports'], 'source_checks': result['source_checks']}))
