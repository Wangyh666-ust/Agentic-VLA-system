"""One authorized console experiment, with explicit stage/skill gates."""
import json,sys,time,urllib.request
from pathlib import Path
out=Path(__file__).resolve().parent
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
base='http://127.0.0.1:8083'
def get(path):
    with opener.open(base+path,timeout=10) as r:return json.load(r)
token=get('/api/bootstrap')['token']
def post(path,body):
    req=urllib.request.Request(base+path,data=json.dumps(body).encode(),method='POST',
        headers={'Content-Type':'application/json','X-Console-Token':token})
    with opener.open(req,timeout=15) as r:return json.load(r)
def save(name,value):
    (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
def wait(target,limit):
    start=time.monotonic();previous=None
    while time.monotonic()-start<limit:
        state=get('/api/state');save('online_latest.json',state)
        key=(state.get('stage'),state.get('message'))
        if key!=previous:print(json.dumps({'elapsed':round(time.monotonic()-start,1),'stage':key[0],'message':key[1]},ensure_ascii=False),flush=True);previous=key
        if not state['busy']:
            if state['stage'] in target:return state
            if state['stage'] in ('error','blocked','cancelled'):raise RuntimeError(json.dumps(state.get('error_detail') or key,ensure_ascii=False))
        time.sleep(2)
    raise TimeoutError('Console stage deadline exceeded; inspect live state before any further action')
action=sys.argv[1]
if action in ('plan','plan-loaded'):
    assert not get('/api/state')['busy']
    save('online_authorization.json',{'request':'收拾桌面，将碗和瓶子归位','user_authorization':'改完后，再在线测试一下这个任务','cases':1,'helpers':False,'finish_streak':5})
    if action=='plan':post('/api/load',{'scene':'bowl_drawer'})
    ready=wait({'ready'},180)
    started=time.time();post('/api/plan',{'request':'收拾桌面，将碗和瓶子归位','request_id':ready['request_id']})
    state=wait({'draft'},420);save('online_draft_state.json',state)
    save('online_preparation_elapsed.json',{'seconds':time.time()-started,'includes_model_loading':True})
    print(json.dumps(state['draft'],ensure_ascii=False),flush=True)
elif action=='execute':
    state=get('/api/state');draft=state['draft']
    assert not state['busy'] and state['stage']=='draft'
    assert draft['capability_ids']==['xvla_bowl_drawer_close']
    assert draft['request_scope']['coverage']=='partial'
    assert draft['request_scope']['unmet_outcomes']
    post('/api/execute',{'request_id':state['request_id']})
    state=wait({'completed','blocked','error','cancelled'},2500)
    save('online_final_state.json',state)
    print(json.dumps({'stage':state['stage'],'run_path':state.get('run_path')},ensure_ascii=False),flush=True)
else:raise ValueError('plan or execute required')
