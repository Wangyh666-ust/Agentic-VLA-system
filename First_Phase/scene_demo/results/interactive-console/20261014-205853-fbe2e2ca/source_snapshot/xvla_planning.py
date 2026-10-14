"""Joint Hermes identity/planning, followed by one independent visual audit."""
import copy
import hashlib
import json
import os
import time
from pathlib import Path

from scene_demo import catalog
from scene_demo.planning_scope import PLANNING_SCOPE_GUIDANCE
from scene_demo.xvla_grounding import (InventoryClient, VIEWS, build_inventory_prompt,
    dump, triptych, validate_inventory, validate_selection, verify_plan)
from scene_demo.xvla_process_evidence import capture_frames
from scene_demo.xvla_subtask_evidence import digest
from scene_demo.finish_v1.vision import _parse_verdict_content

NATIVE_POLICY = '''只从本次目录选择已经训练的完整原生技能；一个技能的instruction及全部goals不可拆分、删减、改写。单物体技能也必须确为目录中的独立原生技能。
“整理/收好/归位”等宽泛需求可匹配完整收纳技能，完整入抽屉并关抽屉属于收纳的自然收尾；必须在executable_outcomes中明确包含关抽屉等完整结果。明确要求保持打开、禁止关门，或原生技能包含无关对象/改变目的地时，不能强行调用，返回澄清或仅选择其他独立原生能力。不能伪造半段技能。
加热缺少启动加热能力时，可以选择完整入炉关门作为准备，coverage=partial并说明无法启动加热。收纳篮不能无依据当垃圾桶；丢弃目的地不明需要澄清。当前场景没有瓶子归位技能就明确列未支持，但可执行碗的完整收纳。
只输出必要短句，不能把不存在的能力称作支持。所有实际所选技能的完整goals都会验收，中途子动作不意味着整项结束。'''


def signature(payload, inventory):
    keys = ('request_id','session_id','scene_version','decision','capability_ids',
            'target_bindings','request_scope')
    value = {k:payload.get(k) for k in keys}
    value['inventory'] = inventory
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def build_prompt(session, request):
    caps = [{k:c[k] for k in ('capability_id','instruction','goals')} for c in session['capabilities']]
    return (build_inventory_prompt(request) + '\n将上面的物品JSON放入inventory字段；同一次识别同时选择完整原生技能生成draft。'
        + NATIVE_POLICY + '\n' + PLANNING_SCOPE_GUIDANCE
        + '\ndraft格式：{"decision":"execute|clarify|unsupported","capability_ids":["目录中的ID，按顺序"],'
          '"target_bindings":{"所选goals每个对象或区域ID":"inventory中的object_id"},'
          '"rationale":"简短规划理由", "request_scope":{"coverage":"full|partial|none",'
          '"executable_outcomes":[],"unmet_outcomes":[],"reason":"范围理由"}}。'
          '只输出顶层inventory和draft两个字段。目标区域可绑定其所属真实装置，同一装置不同区域需在描述中区分。'
          '不要输出session/request ID、budget或额外控制参数，这些由服务端填写。\n'
        + json.dumps({'scene':session['scene_id'],'storage_policy':session.get('storage_policy',{}),
                      'native_skills':caps},ensure_ascii=False))


def hermes_once(prompt, frames, home, directory, cancel_check=None):
    from scene_demo.run_agent import HermesRunner
    home, directory = Path(home), Path(directory)
    # Use installation facts, not a guessed package API or a new environment.
    facts = sorted((home/'installs').glob('*/facts.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not facts:
        raise RuntimeError('Shared Hermes installation missing')
    venv = Path(json.loads(facts[0].read_text())['packages']['venv']['environment'])
    sdk = venv.parent/'workspace'
    if not (venv/'bin/python').is_file() or not (sdk/'run_agent.py').is_file():
        raise RuntimeError('Installed Hermes SDK paths invalid')
    content = [{'type':'text','text':prompt}]
    urls = InventoryClient()._load_images([f['rgb_path'] for f in frames])
    for i,(frame,url) in enumerate(zip(frames,urls)):
        content += [{'type':'text','text':f"Image frame={i}; view={frame['view']}; timestamp={frame['timestamp']}"},
                    {'type':'image_url','image_url':{'url':url}}]
    source, target = directory/'hermes_input.json', directory/'hermes_result.json'
    dump(source,{'content':content})
    env = dict(os.environ,HERMES_HOME=str(home),PYTHONPATH=str(sdk),HERMES_EPHEMERAL_MODE='1')
    runner = HermesRunner(home=str(home),cwd=str(sdk),cancel_check=cancel_check)
    result = runner._run_cancellable([str(venv/'bin/python'),str(Path(__file__).with_name('xvla_hermes_once.py')),
        str(source),str(target),str(sdk)],env,None,120)
    (directory/'hermes.log').write_bytes(result.get('output') or b'')
    if result['exit_code'] != 0:
        raise RuntimeError('Hermes joint planning failed: '+str(result.get('error') or 'see hermes.log'))
    raw = json.loads(target.read_text())
    return _parse_verdict_content(raw['content']), raw


def plan_from_capture(capture, request, session, request_id, directory, home, cancel_check=None, planner=None, review_client=None):
    from scene_demo.xvla_console import validate_draft
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    frames = [{**f,'timestamp':f.get('timestamp',capture['timestamp'])} for f in capture_frames(capture)]
    dump(directory/'capture.json',capture)
    prompt = build_prompt(session,request)
    (directory/'joint_prompt.txt').write_text(prompt,encoding='utf-8')
    first, meta = (planner or hermes_once)(prompt,frames,home,directory,cancel_check)
    inventory = validate_inventory(first.get('inventory'))
    inventory.update(version=1,captured_at=capture['timestamp'],source_images=[
        {'frame':i,'view':f['view'],'path':f['rgb_path'],'sha256':digest(f['rgb_path'])} for i,f in enumerate(frames)])
    dump(directory/'inventory_first.json',{'inventory':inventory,'metadata':meta})
    draft = copy.deepcopy(first.get('draft'))
    if not isinstance(draft,dict):
        raise ValueError('Hermes draft missing')
    draft.update(request_id=request_id,session_id=session['session_id'],scene_version=session['scene_version'])
    session = {**session,'scene_inventory':inventory}
    draft = validate_draft(session,draft,request_id)
    capabilities = [catalog.CAPABILITIES[cid] for cid in draft['capability_ids']]
    selection = validate_selection(inventory,draft.get('target_bindings',{}),capabilities)
    draft['target_bindings'] = selection
    if cancel_check and cancel_check():raise RuntimeError('user_cancelled')
    first_seconds = time.monotonic()-started
    review = verify_plan(inventory,selection,request,capabilities,directory,client=review_client,
                         request_scope=draft['request_scope'],joint_draft=draft)
    if not review['passed']:
        raise ValueError('Independent identity/plan review rejected draft; see selected_targets_review.json')
    inventory = review['inventory']
    inventory['reviewed_from_views'] = list(VIEWS)
    validate_draft({**session,'scene_inventory':inventory},draft,request_id)
    dump(directory/'inventory_confirmed.json',{'inventory':inventory,'metadata':review['metadata']})
    picture = directory/'planning_three_views.png';triptych(frames,picture)
    timing = {'first_call_with_startup_seconds':first_seconds,
              'independent_review_seconds':time.monotonic()-started-first_seconds,
              'total_seconds':time.monotonic()-started,'normal_model_calls':2,
              'planner':'installed Hermes SDK','model':'qwen3.8-max'}
    dump(directory/'joint_planning_timing.json',timing)
    dump(directory/'joint_draft.json',draft)
    return {'inventory':inventory,'draft':draft,'review':review,'image':str(picture),
            'signature':signature(draft,inventory),'timing':timing}


def reusable_review(session, plan):
    prepared = getattr(session,'prepared_joint_plan',None)
    if not prepared:return None
    inventory = session.scene_inventory
    if any(digest(i['path']) != i['sha256'] for i in inventory['source_images']):
        return None
    payload = {k:getattr(plan,k,None) for k in ('request_id','session_id','scene_version','decision',
               'capability_ids','target_bindings','request_scope')}
    if signature(payload,inventory) != prepared['signature']:return None
    return copy.deepcopy(prepared['review'])
