import copy,json,tempfile,unittest,base64,hashlib,struct
from pathlib import Path
from unittest import mock
from scene_demo import catalog
from scene_demo.xvla_finish import build_skill_context,build_prompt,XvlaQwenClient
from scene_demo.finish_v1 import runtime,sensors,control
from scene_demo.finish_v1.tests.test_runtime import FakePort

class Tests(unittest.TestCase):
 def ctx(self,id='xvla_mugs_both'):
  c=build_skill_context(catalog.CAPABILITIES[id],'secret request')
  c['sensor_frames']=[{'frame':i,'view':v,'timestamp':1.0} for i,v in enumerate(XvlaQwenClient.required_views)]
  c['actuator_measurements']={'timestamp':1.0,'finger_qpos':[.04,-.04],'standard_open_finger_qpos':[.04,-.04]}
  return c
 def client(self):
  c=XvlaQwenClient(env_path='/nonexistent');c.set_check_contract(self.ctx());return c
 def reply(self,values):
  return json.dumps({'checks':[{'id':'goal_'+str(i),'achieved':v,'reason':'software fake'} for i,v in enumerate(values)],'reason':'fake','goal_achieved':True})
 def test_context(self):
  for id,last in [('xvla_basket_both','sauce_to_basket'),('xvla_mugs_both','yellow_mug_right'),('xvla_microwave_place_close','microwave_close')]:
   c=self.ctx(id);self.assertEqual(len(c['criteria']['success_checks']),2);self.assertEqual(c['current_subtask'],runtime.build_context(catalog.CAPABILITIES[last],'x')['current_subtask'])
  self.assertEqual(len(self.ctx('white_mug_left')['criteria']['success_checks']),1)
  cap=copy.deepcopy(catalog.CAPABILITIES['xvla_mugs_both']);cap['goals'].pop()
  with self.assertRaises(ValueError):build_skill_context(cap,'x')
 def test_prompt(self):
  for id,relations in [('white_mug_left',['on']),('mug_to_microwave',['in']),('xvla_microwave_place_close',['in','close'])]:
   p=build_prompt(self.ctx(id));s=json.loads(p.splitlines()[-1]);self.assertEqual([c['relation'] for c in s['success_checks']],relations)
   self.assertEqual(set(s),{'current_skill','success_checks','sensor_frames','actuator_measurements'})
   for x in ('secret request','phase','stage','predicates','contact','home','goals'):self.assertNotIn(x,json.dumps(s))
   self.assertIn('at_standard_open=true',p);self.assertTrue(s['actuator_measurements']['at_standard_open'])
 def test_verdict(self):
  c=self.client()
  for values,want in [([True,False],False),([True,None],None),([True,True],True)]:
   self.assertIs(c.parse_verdict(self.reply(values))['operation_achieved'],want);self.assertTrue(c.last_skill_verdict['reported_goal_achieved'])
  checks=json.loads(self.reply([True,True]))['checks']
  bad=[[],checks[:1],[checks[0],checks[0]]]
  missing=copy.deepcopy(checks);del missing[0]['achieved'];bad.append(missing)
  for field,value in [('id','unknown'),('achieved',1),('achieved','true'),('reason',1)]:
   item=copy.deepcopy(checks);item[0][field]=value;bad.append(item)
  for item in bad:
   with self.assertRaises(ValueError):c.parse_verdict(json.dumps({'checks':item}))
 def monitor(self,tmp,client,three=True,side=True):
  with mock.patch.object(sensors,'SensorPort',FakePort):m=runtime.FinishMonitor(None,{'finger_home':[.04,-.04]},tmp,self.ctx(),lambda f:f(),None,lambda:True,300,client)
  orig=m.port.capture
  def capture():
   c=orig();c['timestamp']=1.0
   if side:c['verification_frames']=[{'view':'sideview','rgb_path':'side.png'}]
   return c
  m.port.capture=capture
  if three:m.port.checker_views=XvlaQwenClient.required_views
  return m
 def test_poll(self):
  with tempfile.TemporaryDirectory() as tmp:
   c=self.client()
   def request(ctx,paths,prompt):
    self.assertEqual([f['view'] for f in ctx['sensor_frames']],list(c.required_views));self.assertEqual([f['timestamp'] for f in ctx['sensor_frames']],[1.0]*3)
    return {'verdict':c.parse_verdict(self.reply([True,False]))}
   with mock.patch.object(c,'_request',side_effect=request) as call,mock.patch.object(control,'run_finish') as finish:
    m=self.monitor(tmp,c);self.assertIsNone(m.poll(150));finish.assert_not_called();self.assertEqual(m.port.steps,0);call.assert_called_once()
   with mock.patch.object(c,'_request') as call:
    m=self.monitor(tmp,c,side=False)
    with self.assertRaises(RuntimeError):m._check(m.context,[])
    call.assert_not_called()
   with mock.patch.object(c,'_request',side_effect=lambda *a:{'verdict':c.parse_verdict(self.reply([True,True]))}),mock.patch.object(control,'run_finish',return_value={'ok':False,'reason':'fake'}) as finish:
    m=self.monitor(tmp,c);m.poll(150);finish.assert_called_once();self.assertEqual(m.port.steps,0)
   from scene_demo.finish_v1.simple_checker import SimpleQwenClient
   old=SimpleQwenClient(env_path='/nonexistent')
   with mock.patch.object(old,'analyze',return_value={'verdict':old.parse_verdict('{"goal_achieved":false,"reason":"fake"}')}) as call:
    m=self.monitor(tmp,old,three=False);m._check(m.context,[]);self.assertEqual([f['view'] for f in call.call_args.args[0]['sensor_frames']],['wrist','sideview'])
 def test_microwave_partial(self):
  c=self.client();c.set_check_contract(self.ctx('xvla_microwave_place_close'))
  self.assertIs(c.parse_verdict(self.reply([True,False]))['operation_achieved'],False)
  self.assertIs(c.parse_verdict(self.reply([True,None]))['operation_achieved'],None)
 def test_capture_mismatch(self):
  c=self.client();cap={'timestamp':1.0,'robot':{'finger_qpos':[.04,-.04]},'frames':[{'view':v,'rgb_path':v+'.png'} for v in c.required_views]};c.observe_capture(cap)
  with mock.patch.object(c,'_request') as request:
   for field in ('time','path','finger'):
    ctx=self.ctx();paths=[v+'.png' for v in c.required_views]
    if field=='time':ctx['sensor_frames'][0]['timestamp']=2
    if field=='path':paths[0]='wrong.png'
    if field=='finger':ctx['actuator_measurements']['finger_qpos'][0]=0
    with self.assertRaises(ValueError):c.analyze(ctx,paths)
   request.assert_not_called()
 def test_serializer(self):
  root=Path(__file__).resolve().parents[3]/'First_Phase/scene_demo'
  cases=json.loads((root/'results/2026-10-12-xvla-checker-grounding/cases.json').read_text())
  case=next(c for c in cases if c['name']=='white_arrived')
  capture=copy.deepcopy(case['capture']);ctx=self.ctx('white_mug_left');ctx['actuator_measurements']=copy.deepcopy(case['source_context']['actuator_measurements'])
  ctx['actuator_measurements']['timestamp']=capture['timestamp'];ctx['actuator_measurements']['finger_qpos']=capture['robot']['finger_qpos']
  sources=capture.get('frames',[])+capture.get('verification_frames',[]);paths=[]
  for i,v in enumerate(XvlaQwenClient.required_views):
   frame=next(f for f in sources if f['view']==v);p=frame['rgb_path'];p=p.replace('D:/FYP','/mnt/d/FYP').replace('D:\\FYP','/mnt/d/FYP').replace('\\','/');frame['rgb_path']=p;paths.append(p)
  ctx['sensor_frames']=[{'frame':i,'view':v,'timestamp':capture['timestamp']} for i,v in enumerate(XvlaQwenClient.required_views)]
  c=XvlaQwenClient(env_path='/nonexistent');c.observe_capture(capture)
  fake=json.dumps({'choices':[{'message':{'content':self.reply([True])}}]}).encode();response=mock.Mock();response.getcode.return_value=200;response.read.return_value=fake
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(c,'_resolve_key',return_value='software-test-only'),mock.patch('urllib.request.urlopen',return_value=response) as call:
   c.audit_dir=tmp;c.analyze(ctx,paths);body=json.loads(call.call_args.args[0].data);self.assertEqual(len(body['messages']),1)
   items=body['messages'][0]['content'];urls=[x['image_url']['url'] for x in items if x['type']=='image_url'];self.assertEqual(len(urls),3)
   summary=[]
   for i,(url,path) in enumerate(zip(urls,paths)):
    data=base64.b64decode(url.split(',',1)[1]);self.assertEqual(data,Path(path).read_bytes());self.assertEqual(data[:8],b'\x89PNG\r\n\x1a\n');size=struct.unpack('>II',data[16:24]);self.assertEqual(size,[(256,256),(256,256),(512,512)][i]);summary.append({'view':c.required_views[i],'path':path,'sha256':hashlib.sha256(data).hexdigest(),'size':size})
   labels=[x['text'] for x in items if x['type']=='text'][1:];self.assertEqual(len(labels),3)
   for v,label in zip(c.required_views,labels):self.assertIn('view='+v,label);self.assertIn('timestamp='+str(capture['timestamp']),label)
   out=root/'results/2026-10-13-xvla-process-evidence/software/serializer';out.mkdir(parents=True,exist_ok=True)
   (out/'serializer_summary.json').write_text(json.dumps({'fake_model_response':True,'real_model_calls':0,'sim_actions':0,'messages':1,'images':summary,'labels':labels},indent=2)+'\n')
   (out/'exact_prompt.txt').write_text(build_prompt(ctx)+'\n',encoding='utf-8')
