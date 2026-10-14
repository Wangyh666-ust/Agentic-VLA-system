import subprocess,json,re,hashlib
from pathlib import Path
ROOT=Path('D:/FYP');HERE=ROOT/'First_Phase/scene_demo';BASE=HERE/'results/2026-10-13-xvla-all-native/software/final';OUT=BASE/'v2'
old=json.loads((BASE/'qualification.json').read_text(encoding='utf-8'));commands={}
def run(name,argv):
 (OUT/(name+'.argv.json')).write_text(json.dumps(argv,ensure_ascii=False,indent=2),encoding='utf-8');r=subprocess.run(argv,cwd=ROOT,capture_output=True)
 (OUT/(name+'.stdout')).write_bytes(r.stdout);(OUT/(name+'.stderr')).write_bytes(r.stderr);(OUT/(name+'.exitcode')).write_text(str(r.returncode)+'\n')
 commands[name]={'argv':name+'.argv.json','stdout':name+'.stdout','stderr':name+'.stderr','exitcode':r.returncode}
 print(name,r.returncode,r.stdout[-300:].decode('utf-8',errors='replace'),flush=True);return r
argv=json.loads((BASE/'regression.argv.json').read_text(encoding='utf-8'))
contract=run('contract',argv[:-2]+['First_Phase/scene_demo/tests/test_service_contract.py'])
reg=run('regression',argv)
p=HERE/'tests/test_service_contract.py';compile(p.read_text(encoding='utf-8-sig'),str(p),'exec');violations=[i for i,line in enumerate(p.read_bytes().splitlines(),1) if line.endswith((b' ',b'\t'))]
(OUT/'source_checks.json').write_text(json.dumps({'file':str(p),'syntax_passed':True,'trailing_whitespace':violations,'passed':not violations},indent=2))
run('git_whitespace',['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--check','--','First_Phase/scene_demo/tests/test_service_contract.py'])
run('registry_validate',['D:/miniconda/python.exe','.agents/skills/fyp-experiment-review/scripts/review.py','validate'])
registry=HERE/'experiment_registry.json';index=HERE/'EXPERIMENT_INDEX.md';data=json.loads(registry.read_text(encoding='utf-8'));rows=[]
for entry in data['entries']:
 for e in entry['evidence']:
  p=ROOT/e['path'];actual=hashlib.sha256(p.read_bytes()).hexdigest();rows.append({'entry':entry['id'],'path':e['path'],'expected':e['sha256'],'actual':actual,'matches':actual==e['sha256']})
report={'entries':len(data['entries']),'evidence':len(rows),'all_match':all(r['matches'] for r in rows),'rows':rows}
(OUT/'evidence.after.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
summary=dict(old);summary['commands']={name:{**value,'argv':'../'+value['argv'],'stdout':'../'+value['stdout'],'stderr':'../'+value['stderr']} for name,value in old['commands'].items()};summary['commands'].update(commands)
summary['test_summary']=re.findall(r'[^\n]*(?:passed|failed)[^\n]*',reg.stdout.decode('utf-8',errors='replace'))[-1:]
summary['contract_test_summary']=re.findall(r'[^\n]*(?:passed|failed)[^\n]*',contract.stdout.decode('utf-8',errors='replace'))[-1:]
summary['previous_regression']={'exitcode':1,'stdout':'../regression.stdout','reason':'旧测试错误假定Smol8scene与包括7个X专用scene的15scene目录相等；产品隔离符合设计'}
summary.update(registry_entries=report['entries'],evidence_count=report['evidence'],all_old_evidence_sha_match=report['all_match'],registry_bytes_unchanged=registry.read_bytes()==(BASE/'registry.before.json').read_bytes(),index_bytes_unchanged=index.read_bytes()==(BASE/'index.before.md').read_bytes())
summary['qualified']=all(r['exitcode']==0 for r in summary['commands'].values()) and not violations and all(summary[k] for k in ('help_11_cases','source_syntax_whitespace_passed','downstream_9_byte_identical','all_old_evidence_sha_match','registry_bytes_unchanged','index_bytes_unchanged'))
summary['production_changes_for_v2']=False
(OUT/'qualification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False),flush=True)
