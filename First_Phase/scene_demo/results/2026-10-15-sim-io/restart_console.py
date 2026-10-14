"""Bounded, validated idle-console restart; GET requests only."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

OUT = Path(__file__).resolve().parent
ENTRY = '/mnt/d/FYP/First_Phase/scene_demo/xvla_console.py'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
report = {'commit': '6b6bec3f37408f51bba75572f9b38631ad694387', 'old_pid': 398,
          'scope': 'restart UI only; no model inference or robot action'}


def get(path):
    with OPENER.open('http://127.0.0.1:8083'+path, timeout=3) as response:
        return json.load(response)


def port_open():
    with socket.socket() as sock:
        sock.settimeout(.5)
        return sock.connect_ex(('127.0.0.1', 8083)) == 0


try:
    state = get('/api/state')
    report['before'] = {'busy':state.get('busy'), 'stage':state.get('stage')}
    assert state.get('busy') is False, 'console busy'
    assert state.get('stage') in ('completed','idle','history','blocked','error','cancelled'), 'unexpected stage'
    cmdline = Path('/proc/398/cmdline').read_bytes().split(b'\0')
    args = [arg.decode() for arg in cmdline if arg]
    assert ENTRY in args and '--port' in args and args[args.index('--port')+1] == '8083', 'PID identity mismatch'
    os.kill(398, signal.SIGTERM)
    report['sigterm_sent'] = True
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        if not Path('/proc/398').exists() and not port_open():
            break
        time.sleep(.25)
    assert not Path('/proc/398').exists() and not port_open(), 'old console did not exit; no forcekill'
    command = [sys.executable, ENTRY, '--launch', '--port', '8083']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    report.update(command=command, returncode=result.returncode,
                  stdout=result.stdout.decode(), stderr=result.stderr.decode())
    assert result.returncode == 0, 'launcher failed'
    deadline = time.monotonic()+15
    while True:
        try:
            bootstrap = get('/api/bootstrap')
            state = get('/api/state')
            assert bootstrap.get('app') == 'xvla_interactive_console_v1', 'wrong app'
            assert state.get('busy') is False, 'new console busy'
            assert state.get('stage') == 'idle', 'new console not idle'
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.25)
    report.update(ok=True, app=bootstrap['app'], after={'busy':state['busy'],'stage':state['stage']},
                  model_cold_expected=True)
except Exception as error:
    report.update(ok=False, error_type=type(error).__name__, error=str(error))
finally:
    (OUT/'restart_console.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
if not report.get('ok'):
    raise SystemExit(1)
