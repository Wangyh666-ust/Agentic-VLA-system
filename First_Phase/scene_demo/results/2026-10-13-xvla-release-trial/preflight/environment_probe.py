import hashlib,json,subprocess,os,shutil,re
from pathlib import Path
ROOT=Path('/mnt/d/FYP');OUT=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-release-trial';SNAP=OUT/'source_snapshot/external';SNAP.mkdir(parents=True,exist_ok=True)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def command(argv):
 r=subprocess.run(argv,capture_output=True);return {'argv':argv,'returncode':r.returncode,'stdout':r.stdout.decode(errors='replace'),'stderr':r.stderr.decode(errors='replace')}
model=Path('/home/yhwang/fyp/xvla_native/model/model.safetensors')
info={'model':{'path':str(model),'bytes':model.stat().st_size,'sha256':sha(model)},'python':{'model':'/home/yhwang/fyp/xvla_native/venv/bin/python','sim':'/home/yhwang/fyp/xvla_native/sim_venv/bin/python'},'credentials_source_exists':Path('/home/yhwang/.hermes/.env').is_file()}
info['git']={name:command(['git','-C',path,'rev-parse','HEAD']) for name,path in [('upstream','/home/yhwang/fyp/xvla_native/upstream'),('LIBERO','/home/yhwang/fyp/xvla_native/LIBERO')]}
info['gpu']=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'])
info['ports']=[]
for port in (8769,8794):
 status=command(['ss','-H','-ltnp','sport = :'+str(port)]);pids=sorted(set(re.findall(r'pid=(\d+)',status['stdout'])));processes=[]
 for pid in pids:
  try:
   raw=Path('/proc/'+pid+'/cmdline').read_bytes();stat=Path('/proc/'+pid+'/stat').read_text();rest=stat[stat.rfind(')')+2:].split();processes.append({'pid':int(pid),'argv':raw.decode(errors='replace').strip('\0').split('\0'),'startticks':rest[19],'exe':os.readlink('/proc/'+pid+'/exe')})
  except OSError as e:processes.append({'pid':int(pid),'read_error':str(e)})
 info['ports'].append({'port':port,'ss':status,'listening':bool(status['stdout'].strip()),'processes':processes})
external=[Path('/home/yhwang/fyp/xvla_native/upstream/deploy.py'),Path('/home/yhwang/fyp/xvla_native/upstream/evaluation/libero/libero_client.py')]
bddl=Path('/home/yhwang/fyp/xvla_native/LIBERO/libero/libero/bddl_files/libero_10')
for instruction in ('put the yellow and white mug in the microwave and close it','put both the alphabet soup and the tomato sauce in the basket'):
 files=list(bddl.glob('*'+instruction.replace(' ','_')+'.bddl'));assert len(files)==1;external.append(files[0])
info['external_sources']=[]
for source in external:
 target=SNAP/source.name;shutil.copyfile(source,target);info['external_sources'].append({'source':str(source),'snapshot':str(target.relative_to(ROOT)),'sha256':sha(source),'snapshot_sha256':sha(target)})
from scene_demo.xvla_pilot import frozen_sources
info['repo_sources']=[str(p.relative_to(ROOT)) for p in frozen_sources()]
print(json.dumps(info,ensure_ascii=False,indent=2))
