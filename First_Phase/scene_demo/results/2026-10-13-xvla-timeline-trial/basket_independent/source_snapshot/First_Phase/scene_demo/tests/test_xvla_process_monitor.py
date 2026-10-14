"""Actual X monitor subclass with inert measured ports and fake HTTP responses."""
import copy
import json
import base64
import hashlib
from pathlib import Path
from unittest import mock
import pytest
from scene_demo import catalog, service, xvla_service
from scene_demo.finish_v1 import sensors, control, runtime
from scene_demo.finish_v1.tests.test_runtime import FakePort
from scene_demo.xvla_monitor import XvlaFinishMonitor
from scene_demo.xvla_finish import XvlaQwenClient
from scene_demo.tests.test_xvla_process_evidence import HOME,CLOSED,VIEWS,context,checks,respond

class ProcessPort(FakePort):
    def __init__(self,env,reference,output_dir,step_callback=None):
        super().__init__(env,reference,output_dir,step_callback)
        self.directory=Path(output_dir);self.directory.mkdir(parents=True,exist_ok=True)
        self.capture_count=0
    def capture(self):
        self.capture_count+=1
        frames=[]
        for view in VIEWS:
            path=self.directory/('%s_%s.png'%(self.timestamp,view));path.write_bytes(b'fake image')
            frames.append({'view':view,'rgb_path':str(path)})
        return {'timestamp':self.timestamp,'frames':frames,'robot':self.read_robot()}

def monitor(tmp_path):
    ctx=context('xvla_mugs_both',{'timestamp':0.,'robot':{'finger_qpos':HOME}})
    client=XvlaQwenClient(env_path='/nonexistent',max_calls=14)
    with mock.patch.object(sensors,'SensorPort',ProcessPort):
        result=XvlaFinishMonitor(None,{'finger_home':HOME},tmp_path,ctx,lambda f:f(),None,lambda:True,900,client)
    result.port.checker_views=VIEWS
    return result

def release(m,start=1):
    for step,fingers in [(start,CLOSED),(start+1,HOME),(start+2,HOME)]:
        m.port.timestamp=float(step);m.port.robot['finger_qpos']=list(fingers);m.record(step)

def test_event_between_intervals_and_current_only_handoff(tmp_path):
    m=monitor(tmp_path);release(m)
    assert m.release_events==1 and m.port.steps==0 and m.calls==m.release_checks==0
    before=m.port.capture_count;m.record(3);assert m.port.capture_count==before
    events=[json.loads(line) for line in (tmp_path/'process_events.jsonl').read_text().splitlines()]
    assert len(events)==1;assert events[0]['capture']['timestamp']==3.
    assert [i['view'] for i in events[0]['images']]==list(VIEWS)
    records=[]
    def request(ctx,paths,prompt):
        role=json.loads(prompt.splitlines()[-1]).get('observation_role')
        records.append((role,copy.deepcopy(ctx),list(paths)))
        values=[True,False] if role else [True,False]
        return respond(m.call_client,values,None if role else {0:['release_0001']})(ctx,paths,prompt)
    with mock.patch.object(m.call_client,'_request',side_effect=request),mock.patch.object(control,'run_finish') as finish:
        m.port.timestamp=150.;assert m.poll(150) is None;finish.assert_not_called()
    assert [r[0] for r in records]==['release_snapshot',None]
    assert records[0][1]['actuator_measurements']['timestamp']==3.
    assert records[1][1]['actuator_measurements']['timestamp']==150.
    status=m.status();assert (status['release_evidence_calls'],status['current_view_calls'],status['calls'])==(1,1,2)
    assert m.port.steps==0
    with mock.patch.object(m.call_client,'_request',side_effect=respond(m.call_client,[True,True],{0:['release_0001']})),mock.patch.object(control,'run_finish',return_value={'ok':False,'reason':'fake finish'}) as finish:
        m.port.timestamp=300.;m.poll(300);finish.assert_called_once()
    assert m.port.steps==0
    assert m.status()['calls']==3


def test_release_budget_photos_persist_without_extra_current_calls(tmp_path):
    m=monitor(tmp_path)
    for i in range(5):release(m,10*i+1)
    assert m.release_events==5
    with mock.patch.object(m.call_client,'_request',side_effect=respond(m.call_client,[False,False])):
        m.port.timestamp=150.;m.poll(150)
    assert m.release_checks==4 and m.calls==1 and m.status()['calls']==5
    rows=[json.loads(line) for line in (tmp_path/'process_checks.jsonl').read_text().splitlines()]
    assert len(rows)==5 and rows[-1]['status']=='unchecked_due_budget'
    assert m.port.steps==0


def test_helper_release_records_without_extra_actions(tmp_path):
    m=monitor(tmp_path)
    for i,fingers in enumerate((CLOSED,HOME,HOME)):
        m.port.timestamp=float(i);m.port.robot['finger_qpos']=list(fingers)
        m._on_action({'phase':'release'})
    assert m.release_events==1 and m.release_checks==0
    assert m._helper_actions==3 and m.helper_counts['release']==3 and m.port.steps==0

@pytest.mark.parametrize('problem',['missing','time','finger'])
def test_pending_source_errors_reject_no_handoff(tmp_path,problem):
    m=monitor(tmp_path);release(m);event=m.pending_releases[0]
    if problem=='missing':event['capture']['frames'].pop()
    if problem=='time':event['capture']['frames'][0]['timestamp']=9.
    if problem=='finger':event['capture']['robot']['finger_qpos']=CLOSED
    with mock.patch.object(m.call_client,'_request',side_effect=respond(m.call_client,[True,True])) as request,mock.patch.object(control,'run_finish') as finish:
        with pytest.raises(ValueError):m.poll(150)
        finish.assert_not_called()
        request.assert_not_called()
    assert m.port.steps==0


def test_service_factory_all_x_and_original_smol():
    from scene_demo.xvla_skills import NATIVE_LONG_SKILLS
    smol=object.__new__(service.SceneService);x=object.__new__(xvla_service.XvlaSceneService)
    assert smol._finish_monitor_class() is runtime.FinishMonitor
    for cid in NATIVE_LONG_SKILLS:assert x._finish_monitor_class() is XvlaFinishMonitor


def test_real_http_serializer_release_and_current(tmp_path):
    root=Path(__file__).resolve().parents[1]
    cases=json.loads((root/'results/2026-10-12-xvla-checker-grounding/cases.json').read_text())
    case=next(c for c in cases if c['name']=='white_arrived');cap=copy.deepcopy(case['capture'])
    sources=cap.get('frames',[])+cap.get('verification_frames',[]);paths=[]
    for view in VIEWS:
        frame=next(f for f in sources if f['view']==view)
        p=frame['rgb_path'].replace('D:/FYP','/mnt/d/FYP').replace('D:\\FYP','/mnt/d/FYP').replace('\\','/');frame['rgb_path']=p;paths.append(p)
    ctx=context('white_mug_left',cap);ctx['actuator_measurements']=copy.deepcopy(case['source_context']['actuator_measurements'])
    client=XvlaQwenClient(env_path='/nonexistent',max_calls=14);client.audit_dir=tmp_path/'audit';client.observe_capture(cap)
    replies=[True,None];bodies=[]
    def urlopen(request,**kwargs):
        bodies.append(json.loads(request.data));value=replies[len(bodies)-1]
        rows=checks(ctx,[True])
        if value is None:rows[0].update(current_evidence=False,evidence_event_ids=['release_software_fixture'])
        payload={'choices':[{'message':{'content':json.dumps({'checks':rows,'reason':'software fake response'})}}]}
        response=mock.Mock();response.getcode.return_value=200;response.read.return_value=json.dumps(payload).encode();return response
    with mock.patch.object(client,'_resolve_key',return_value='software-test-only'),mock.patch('urllib.request.urlopen',side_effect=urlopen) as http:
        release_result=client.analyze_release(ctx,paths,'release_software_fixture')
        current_result=client.analyze(ctx,paths)
    assert http.call_count==2 and client._calls==2
    assert release_result['visual_goal_verdict']['goal_achieved'] is True
    assert current_result['raw_visual_goal_verdict']['goal_achieved'] is True
    assert current_result['verdict']['operation_achieved'] is True
    summaries=[]
    for i,body in enumerate(bodies):
        assert len(body['messages'])==1;items=body['messages'][0]['content']
        urls=[x['image_url']['url'] for x in items if x['type']=='image_url'];assert len(urls)==(3 if i==0 else 6)
        prompt=items[0]['text'];safe=json.loads(prompt.splitlines()[-1]);assert safe.get('observation_role')==('release_snapshot' if i==0 else None)
        labels=[x['text'] for x in items[1:] if x['type']=='text'];images=[]
        for view,url,path,label in zip(list(VIEWS)*(1 if i==0 else 2),urls,paths*(1 if i==0 else 2),labels):
            data=base64.b64decode(url.split(',',1)[1]);assert data==Path(path).read_bytes();assert 'view='+view in label;assert str(cap['timestamp']) in label
            images.append({'view':view,'path':path,'sha256':hashlib.sha256(data).hexdigest()})
        summaries.append({'role':safe.get('observation_role'),'images':images,'labels':labels,'encoder':safe['actuator_measurements']})
    out=tmp_path/'serializer';out.mkdir(parents=True,exist_ok=True)
    (out/'two_request_summary.json').write_text(json.dumps({'fake_response':True,'real_http_calls':0,'environment_actions':0,'saved_source_capture_time':cap['timestamp'],'software_event_only':True,'requests':summaries},indent=2)+'\n')
    for i,body in enumerate(bodies):(out/('request_%s_prompt.txt'%i)).write_text(body['messages'][0]['content'][0]['text'],encoding='utf-8')


def test_pilot_configuration_is_only_template():
    from scene_demo import xvla_pilot
    assert 'xvla-timeline' in str(xvla_pilot.DEFAULT_OUTPUT)
    assert 'xvla-timeline' in str(xvla_pilot.DEFAULT_PROPOSAL)
    assert '20261013-xvla-process-evidence-software' in xvla_pilot.HISTORY
    assert {'xvla_monitor.py','xvla_process_evidence.py'} <= {p.name for p in xvla_pilot.frozen_sources()}
    assert XvlaQwenClient(env_path='/nonexistent').max_calls==14


def test_future_template_objective_event_constraints():
    import inspect
    from scene_demo import xvla_pilot
    source=inspect.getsource(xvla_pilot.prepare)
    for phrase in ('实测闭→开分支触发次数','实际capture组三图数量','release分类真实请求数',
                   '逐goal evidence_id/图片SHA/同步编码器','中间True/False/Null状态实际入请求',
                   '同一目标当前反证撤销','release分类0','停止依赖本机制的验证',
                   '不能拿归位成功代替','复用效果unverified','不据0复用宣称无效',
                   '预期负例0采样不强制触发'):
        assert phrase in source
    for phrase in ('所有native通用','采集','分类','遮挡复用','反证撤销'):
        assert phrase in xvla_pilot.PURPOSE
