import json,os,signal,time,subprocess
from pathlib import Path
OUT=Path('/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-13-xvla-release-trial/runtime')
record=json.loads((OUT/'server_state.json').read_text())
def identity(pid):
 try:
  text=Path('/proc/%d/stat'%pid).read_text();rest=text[text.rfind(')')+2:].split()
  return {'pid':pid,'startticks':rest[19],'state':rest[0],'alive':rest[0]!='Z',
          'argv':Path('/proc/%d/cmdline'%pid).read_bytes().decode().strip('\0').split('\0'),
          'cwd':os.readlink('/proc/%d/cwd'%pid)}
 except (OSError,ValueError):return {'pid':pid,'alive':False}
def matches(current):
 return (record.get('owned') is True and record.get('pid')==409 and record.get('startticks')=='475'
         and current.get('alive') is True and current.get('pid')==record['pid']
         and current.get('startticks')==record['startticks'] and current.get('argv')==record['argv']
         and current.get('cwd')==record['cwd'])
before=identity(record['pid']);result={'owned':record.get('owned'),'before':before,'recorded':record,
    'matched_identity':matches(before),'signals':[],'basket_launches':0}
if not before.get('alive'):
 result['reason']='already_exited'
elif not matches(before):
 result['reason']='identity_mismatch; no signal sent'
else:
 os.kill(record['pid'],signal.SIGTERM);result['signals'].append('SIGTERM')
 deadline=time.monotonic()+15
 while time.monotonic()<deadline and identity(record['pid']).get('alive'):time.sleep(.5)
 current=identity(record['pid'])
 if current.get('alive') and matches(current):
  os.kill(record['pid'],signal.SIGKILL);result['signals'].append('SIGKILL')
  deadline=time.monotonic()+5
  while time.monotonic()<deadline and identity(record['pid']).get('alive'):time.sleep(.5)
 result['reason']='matched owned process signalled'
result['after']=identity(record['pid']);result['terminated']=not result['after'].get('alive')
result['ports']=[]
for port in (8794,8769):
 argv=['ss','-H','-ltnp','sport = :'+str(port)];r=subprocess.run(argv,capture_output=True)
 result['ports'].append({'port':port,'argv':argv,'exitcode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode(),'listening':bool(r.stdout.strip())})
result['freeze_after']=json.loads((OUT/'freeze.after.json').read_text())
result['all_frozen_sources_match']=all(r['matches'] for r in result['freeze_after'])
(OUT/'cleanup.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
