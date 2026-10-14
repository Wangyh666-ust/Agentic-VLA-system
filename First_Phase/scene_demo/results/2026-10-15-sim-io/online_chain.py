"""One authorized simulator-fast full-skill validation. No automatic retry."""
import hashlib,json,time,urllib.request
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[3]
scene=root/'First_Phase/scene_demo'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
base='http://127.0.0.1:8083'
def save(name,value): (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def get(path):
    with opener.open(base+path,timeout=15) as r:return json.load(r)
def post(path,body):
    request=urllib.request.Request(base+path,data=json.dumps(body).encode(),method='POST',headers={'Content-Type':'application/json','X-Console-Token':token})
    with opener.open(request,timeout=20) as r:return json.load(r)
def wait(target,limit):
    previous=None;deadline=time.monotonic()+limit
    while time.monotonic()<deadline:
        state=get('/api/state');save('chain_latest.json',state)
        key=(state.get('stage'),state.get('message'))
        if key!=previous:
            timing['stage_changes'].append({'monotonic':time.monotonic(),'elapsed':time.monotonic()-started,'stage':key[0],'message':key[1]})
            print(json.dumps({'elapsed':round(time.monotonic()-started,2),'stage':key[0],'message':key[1]},ensure_ascii=False),flush=True);previous=key
        if not state['busy']:
            if state['stage'] in target:return state
            if state['stage'] in ('error','blocked','cancelled'):raise RuntimeError(str(state.get('error_detail') or key))
        time.sleep(2)
    raise TimeoutError('No automatic retry; inspect current state')
assert json.loads((out/'chain_review.json').read_text(encoding='utf-8-sig'))['ok'] is True
assert json.loads((out/'bowl_timing.json').read_text())['completed'] is True
bowl_final=json.loads((out/'bowl_final.json').read_text(encoding='utf-8-sig'))
assert bowl_final['jobs'], 'bowl jobs missing'
for job in bowl_final['jobs']:
    pilot=json.loads((Path(job['run_dir'])/'pilot_result.json').read_text(encoding='utf-8-sig'))
    assert pilot['handoff_evidence']['ok'] is True, 'bowl handoff prerequisite failed'
assert not (out/'chain_timing.json').exists(),'one case only'
protected=json.loads((out/'protected_controls.json').read_text(encoding='utf-8-sig'))
assert all(hashlib.sha256((scene/p).read_bytes()).hexdigest()==h for p,h in protected.items())
save('chain_source_hashes.json',{p:hashlib.sha256((scene/p).read_bytes()).hexdigest() for p in ['xvla_sim_io.py','xvla_sim_finish.py','xvla_service.py','xvla_console.py','xvla_console.html']})
state=get('/api/state');assert not state['busy'] and state['stage'] in ('idle','history','completed','blocked','error','cancelled')
token=get('/api/bootstrap')['token']
started=time.monotonic();timing={'started_unix':time.time(),'request':'先把桌上的酒瓶放到酒架上，再把碗放到盘子上，最后打开灶台。','scene':'goal_table','checker_mode':'simulator','human_wait_seconds':0,'helpers':False,'completed':False,'stage_changes':[]}
save('chain_timing.json',timing)
try:
    post('/api/load',{'scene':'goal_table','checker_mode':'simulator'})
    ready=wait({'ready'},180);assert ready['checker_mode']=='simulator'
    loaded=time.monotonic();timing['load_scene_seconds']=loaded-started
    post('/api/plan',{'request':timing['request'],'request_id':ready['request_id']})
    state=wait({'draft'},700);save('chain_draft.json',state)
    planned=time.monotonic();timing['prepare_and_plan_seconds']=planned-loaded
    assert state['draft']['capability_ids']==['wine_to_rack','bowl_to_plate','stove_on']
    assert state['draft']['request_scope']['coverage']=='full'
    save('chain_timing.json',timing)
    post('/api/execute',{'request_id':state['request_id']})
    state=wait({'completed','blocked','error','cancelled'},2200)
    ended=time.monotonic();save('chain_final.json',state)
    timing.update(execution_seconds=ended-planned,total_seconds=ended-started,stage=state['stage'],completed=state['stage']=='completed',run_path=state.get('run_path'))
    print(json.dumps(timing,ensure_ascii=False),flush=True)
except BaseException as exc:
    timing.update(error=str(exc),total_seconds=time.monotonic()-started)
    raise
finally:
    timing['ended_unix']=time.time();save('chain_timing.json',timing)
