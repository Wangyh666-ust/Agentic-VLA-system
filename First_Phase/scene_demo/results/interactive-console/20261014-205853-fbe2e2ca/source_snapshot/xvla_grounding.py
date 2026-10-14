"""Three-view scene identities shared by planning and checking; no control actions."""
import copy
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from scene_demo import joint_home
from scene_demo.finish_v1.simple_checker import SimpleQwenClient
from scene_demo.finish_v1 import vision

VIEWS = ('wrist', 'agentview', 'sideview')
INVENTORY_REQUEST_TIMEOUT = 180
# Fixed workspace camera rig. These cameras are rendered temporarily; native
# policy cameras, control RGBD frames and all physical state remain unchanged.
RIG = {'agentview': ([.80, -.65, .85], [0., 0., .10]),
       'sideview': ([.80, .65, .85], [0., 0., .10])}


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def render_external(env, directory, stem):
    import mujoco
    inner = joint_home._inner_env(env)
    sim = inner.sim
    # Static arena calibration, never object poses or task predicates.
    offset = np.asarray(inner.workspace_offset, dtype=float)
    cid = sim.model.camera_name2id('sideview')
    old = (sim.model.cam_pos[cid].copy(), sim.model.cam_quat[cid].copy(), float(sim.model.cam_fovy[cid]))
    qpos, qvel, stamp = sim.data.qpos.copy(), sim.data.qvel.copy(), float(sim.data.time)
    frames = []
    try:
        for name, (eye, target) in RIG.items():
            eye, target = np.array(eye, float) + offset, np.array(target, float) + offset
            z = eye - target; z /= np.linalg.norm(z)
            x = np.cross([0., 0., 1.], z); x /= np.linalg.norm(x)
            y = np.cross(z, x)
            quat = Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()
            sim.model.cam_pos[cid] = eye
            sim.model.cam_quat[cid] = quat[[3, 0, 1, 2]]
            sim.model.cam_fovy[cid] = 55.
            # Only recompute camera/light transforms. mj_forward can normalize
            # free-joint quaternions and must not be used for a read-only image.
            mujoco.mj_camlight(sim.model._model, sim.data._data)
            pixels = np.asarray(sim.render(width=512, height=512, camera_name='sideview', depth=False))[::-1]
            path = Path(directory) / (stem + '_' + name + '.png')
            Image.fromarray(pixels.astype(np.uint8)).save(path)
            frames.append({'view': name, 'rgb_path': str(path), 'timestamp': stamp,
                           'camera_position': eye.tolist(), 'camera_target': target.tolist(),
                           'camera_fovy': 55., 'camera_role': 'checker_only'})
    finally:
        sim.model.cam_pos[cid], sim.model.cam_quat[cid], sim.model.cam_fovy[cid] = old
        mujoco.mj_camlight(sim.model._model, sim.data._data)
    if not (np.array_equal(qpos, sim.data.qpos) and np.array_equal(qvel, sim.data.qvel) and stamp == sim.data.time):
        raise RuntimeError('checker rendering changed physical state')
    return frames


def attach_checker_cameras(port):
    original = port.capture
    def capture():
        result = original()
        wrist = next(f for f in result['frames'] if f['view'] == 'wrist')
        stem = Path(wrist['rgb_path']).stem + '_checker'
        result['checker_frames'] = [copy.deepcopy(wrist)] + render_external(port.env, port.output_dir, stem)
        return result
    port.capture = capture
    return port


def triptych(frames, path):
    canvas = Image.new('RGB', (1536, 544), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, frame in enumerate(frames):
        with Image.open(frame['rgb_path']) as raw:
            picture = raw.convert('RGB'); picture.thumbnail((512, 512))
            canvas.paste(picture, (512*i+(512-picture.width)//2, 32+(512-picture.height)//2))
        draw.text((512*i+12, 8), f"frame {i}: {frame['view']}", fill='black')
    canvas.save(path)


class InventoryClient(SimpleQwenClient):
    request_options = {'enable_thinking': False, 'max_tokens': 5000,
                       'response_format': {'type': 'json_object'}}

    def __init__(self, env_path='/home/yhwang/fyp/scene_demo/hermes_home/.env',
                 base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
                 model=None, max_calls=18, request_timeout=INVENTORY_REQUEST_TIMEOUT):
        super().__init__(env_path=env_path, base_url=base_url,
                         model=model if model is not None else os.environ.get('SCENE_PLANNER_MODEL', 'qwen3.8-max'),
                         max_calls=max_calls, request_timeout=request_timeout)

    def parse_verdict(self, content):
        self.raw_inventory = vision._parse_verdict_content(content)
        return super().parse_verdict('{"goal_achieved":null,"reason":"scene identity only"}')


def validate_inventory(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get('objects'), list) or not raw['objects']:
        raise ValueError('scene inventory missing')
    ids = set()
    for obj in raw['objects']:
        oid = obj.get('object_id')
        if not isinstance(oid, str) or not oid or oid in ids:
            raise ValueError('inventory IDs must be unique')
        ids.add(oid)
        if not isinstance(obj.get('description'), str) or not obj['description'].strip():
            raise ValueError('visible description required')
        if type(obj.get('confirmed')) is not bool:
            raise ValueError('confirmation required')
        anchors = obj.get('anchors', [])
        if obj['confirmed'] and not anchors:
            raise ValueError('confirmed object requires image reference')
        seen = set()
        for ref in anchors:
            if type(ref.get('frame')) is not int or ref['frame'] not in range(3) or ref['frame'] in seen:
                raise ValueError('invalid/duplicate inventory view')
            seen.add(ref['frame'])
            box = ref.get('box')
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(x) not in (int, float) or not np.isfinite(x) or not 0 <= x <= 1000 for x in box)
                    or not (box[0] < box[2] and box[1] < box[3])):
                raise ValueError('invalid inventory box')
        views = obj.get('view_observations', [])
        if len(views) != 3 or {v.get('frame') for v in views} != {0, 1, 2}:
            raise ValueError('all three views must be inspected per object')
        if any(v.get('visibility') not in ('visible', 'partial', 'occluded', 'out_of_frame', 'uncertain') for v in views):
            raise ValueError('invalid visibility')
    return copy.deepcopy(raw)


def build_inventory_prompt(request, initial=None):
    prompt = ('根据同步三视图建立现场物体清单，供规划和后续检查共用。用户任务：' + request + '\n'
        '清单以用户任务为范围：只列实际要操作的物品、目的地/装置及可能误认成这些目标的必要干扰物。宽泛整理时列其范围内待整理物；不泛列灯、机械臂、墙面、无关背景设施。外观简写实际可见颜色、形状、把手、包装；看不清文字不要猜品牌。'
        '同一实物跨视角只分配一个object_id；不同实物不能合并。识别门/抽屉时可用装置整体加可见活动部件描述。'
        '每个物体必须逐一检查frame 0/1/2；遮挡视角如实记录，不要求三张都看得见。至少一个清晰参照且其他视角无身份冲突才confirmed=true。'
        '框采用每幅原图0..1000的[x1,y1,x2,y2]，包住真实物体，不按期望位置猜框。'
        '输出JSON：{"objects":[{"object_id":"obj_1","description":"可见特征及初始相对位置",'
        '"confirmed":true,"anchors":[{"frame":1,"box":[100,200,300,400]}],'
        '"view_observations":[{"frame":0,"visibility":"occluded","reason":"遮挡情况"},'
        '{"frame":1,"visibility":"visible","reason":"辨认依据"},{"frame":2,"visibility":"partial","reason":"对应依据"}],'
        '"confusable_with":[],"role":"target/destination/distractor/uncertain"}],'
        '"coverage":{"targets_identifiable":true,"destination_visible":true,"state_change_visible":true,"reason":"画面覆盖依据"}}。')
    if initial is None:
        return prompt
    return (prompt + '\n现在重新从三张原图审核下列初稿，初稿可能框到空桌面、混淆目标、把同一物体数两遍。'
        '逐个核对框内确有该物体及跨视角对应，错误必须修订；无法确认的物体标confirmed=false。'
        '不要因为初稿描述自洽就确认。保留相同实物ID，核对任务目标与目的地、必要干扰物；无关项可移除。描述与reason各写必要短句，不逐段重复物品位置。'
        '复核只输出差异JSON：{"reviewed_ids":["逐一核对的原ID"],"reviewed_views":[0,1,2],"updates":["有变化的完整objects项，包括必要新增对象；不变项省略"],"removed_ids":["删除的无关原ID"],"coverage":{"targets_identifiable":true,"destination_visible":true,"state_change_visible":true,"reason":"短依据"},"corrections":[{"object_id":"ID","reason":"短纠正依据"}]}。每个原ID必须明确列在reviewed_ids或removed_ids中；updates仍须三视角观察。\n初稿：'
        + json.dumps(initial, ensure_ascii=False))


def inspect_scene(capture, request, directory, client=None):
    from scene_demo.xvla_process_evidence import capture_frames
    frames = capture_frames(capture)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    dump(directory / 'capture.json', capture)
    client = client or InventoryClient(max_calls=2)
    client.audit_dir = directory / 'model_requests'
    ctx = {'sensor_frames': [{'frame': i, 'view': f['view'], 'timestamp': capture['timestamp']} for i, f in enumerate(frames)]}
    paths = [f['rgb_path'] for f in frames]
    first_meta = client._request(ctx, paths, build_inventory_prompt(request))
    first = validate_inventory(client.raw_inventory)
    dump(directory / 'inventory_first.json', {'inventory': first, 'metadata': first_meta})
    review_prompt = build_inventory_prompt(request, first)
    review_meta = client._request(ctx, paths, review_prompt)
    final = apply_inventory_review(first, client.raw_inventory)
    final.update(version=1, captured_at=capture['timestamp'], reviewed_from_views=list(VIEWS),
                 source_images=[{'frame': i, 'view': f['view'], 'path': f['rgb_path']} for i, f in enumerate(frames)])
    from scene_demo.xvla_subtask_evidence import digest
    for image in final['source_images']: image['sha256'] = digest(image['path'])
    dump(directory / 'inventory_confirmed.json', {'inventory': final, 'metadata': review_meta})
    dump(directory / 'capture.json', capture)
    picture = directory / 'planning_three_views.png'; triptych(frames, picture)
    return final, str(picture)


def apply_inventory_review(first, raw):
    """Compose an audited delta without dropping unchanged original records."""
    if 'objects' in raw:  # Full replies from existing clients remain compatible.
        return validate_inventory(raw)
    if raw.get('reviewed_views') != [0,1,2]:
        raise ValueError('inventory review must inspect three synchronized views')
    old={obj['object_id']:copy.deepcopy(obj) for obj in first['objects']}
    reviewed=raw.get('reviewed_ids');removed=raw.get('removed_ids');updates=raw.get('updates')
    if not all(isinstance(v,list) for v in (reviewed,removed,updates)):
        raise ValueError('inventory review delta lists missing')
    if len(set(reviewed))!=len(reviewed) or len(set(removed))!=len(removed) or set(reviewed)&set(removed):
        raise ValueError('duplicate/conflicting inventory review IDs')
    if set(reviewed)|set(removed)!=set(old):
        raise ValueError('inventory review must account for every original object')
    for oid in removed:old.pop(oid)
    seen=set()
    for obj in updates:
        oid=obj.get('object_id')
        if not isinstance(oid,str) or oid in seen or oid in removed:
            raise ValueError('invalid inventory update ID')
        seen.add(oid);old[oid]=copy.deepcopy(obj)
    result={**copy.deepcopy(first),'objects':list(old.values()),'coverage':raw.get('coverage'),
            'corrections':raw.get('corrections',[])}
    return validate_inventory(result)


def validate_selection(inventory, selection, capabilities):
    required = {str(g[i]) for cap in capabilities for g in cap['goals'] for i in range(1, len(g))}
    if not isinstance(selection, dict) or not required.issubset(selection):
        raise ValueError('bind every goal target and destination to inventory object_id')
    objects = {o['object_id']: o for o in inventory['objects']}
    for key in required:
        if selection[key] not in objects or objects[selection[key]]['confirmed'] is not True:
            raise ValueError('selected identity not visually confirmed: ' + key)
    # Region IDs can refer to different parts of the same destination appliance;
    # distinct manipulated objects, however, must never share one identity.
    moved = {g[1] for c in capabilities for g in c['goals'] if len(g) == 3}
    if len({selection[k] for k in moved}) != len(moved):
        raise ValueError('distinct manipulated objects share one identity')
    return {key: selection[key] for key in sorted(required)}


def seed_checker(client, inventory, selection, capture, context, verified_review=None):
    from scene_demo.xvla_process_evidence import capture_frames
    client.set_check_contract(context)
    store = client.store
    if store.initial_review_done: return
    # This is the shared pre-execution source, with its own synchronized encoder.
    initial_context = copy.deepcopy(context)
    initial_context['sensor_frames'] = [{'frame': i, 'view': v, 'timestamp': capture['timestamp']} for i, v in enumerate(VIEWS)]
    initial_context['actuator_measurements'].update(timestamp=capture['timestamp'],
        finger_qpos=capture['robot']['finger_qpos'], max_abs_arm_qvel=max(map(abs, capture['robot']['arm_qvel'])))
    item = store.observe(initial_context, [f['rgb_path'] for f in capture_frames(capture)], 'planning_initial')
    objects = {o['object_id']: o for o in inventory['objects']}
    required = {g['target_id'] for g in client._check_definitions.values()} | {g['destination_id'] for g in client._check_definitions.values() if g.get('destination_id')}
    for target in required:
        obj = objects[selection[target]]
        store.bindings[target] = {'target_id': target, 'object_id': obj['object_id'],
            'features': obj['description'], 'anchors': [{'capture_id': item['capture_id'], 'view': VIEWS[a['frame']], 'box': a['box']} for a in obj['anchors']],
            'distractors': [], 'reviewed_at': capture['timestamp'], 'review_id': 'planning_inventory_v1'}
    # Inventory establishes parent identities, not their requested subregions.
    store.initial_review_done = False; store.revision += 1
    store.append('planning_bindings', {'selection': selection, 'inventory': inventory, 'bindings': store.bindings})
    client._reference = item
    if verified_review is not None:
        from scene_demo.xvla_subtask_evidence import digest, goal_key
        source=[digest(f['rgb_path']) for f in capture_frames(capture)]
        required_goals={goal_key(g) for g in client._check_definitions.values()}
        if (verified_review.get('passed') is not True or verified_review.get('selection')!=selection
                or source!=verified_review.get('image_sha256')
                or not required_goals.issubset(set(verified_review.get('goal_keys',[])))):
            raise ValueError('combined grounding review does not match task or original images')
        checks={r['target_id']:r for r in verified_review['response']['checks']}
        for target in required:
            checked=checks[target]
            store.bindings[target].update(features=checked['features'],
                anchors=[{'capture_id':item['capture_id'],'view':VIEWS[a['frame']],'box':a['box']} for a in checked['anchors']],
                review_id='combined_plan_grounding')
        required_regions={g['destination_id'] for g in client._check_definitions.values()
                          if g['relation'] in ('in','on','right_of')}
        for region in verified_review['response']['regions']:
            if region['destination_id'] in required_regions:
                store.regions[region['destination_id']]={**copy.deepcopy(region),'reviewed_at':capture['timestamp'],
                    'anchors':[{'capture_id':item['capture_id'],'view':VIEWS[a['frame']],'box':a['box']} for a in region['anchors']],
                    'request_sha256':verified_review['metadata']['request_sha256'],
                    'response_sha256':verified_review['metadata']['response_sha256']}
        if not required_regions.issubset(store.regions):
            raise ValueError('combined grounding review missing requested regions')
        store.initial_review_done=True;store.revision+=1
        store.append('combined_grounding',verified_review)


def verify_plan(inventory, selection, request, capabilities, directory, client=None, request_scope=None, joint_draft=None):
    """Recheck the selected real objects against the request, before any motion."""
    client = client or InventoryClient(max_calls=1)
    client.audit_dir = Path(directory) / 'selection_requests'
    ctx = {'sensor_frames': [{'frame': i, 'view': v, 'timestamp': inventory['captured_at']} for i, v in enumerate(VIEWS)]}
    paths = [i['path'] for i in inventory['source_images']]
    from scene_demo.xvla_finish import build_skill_context
    from scene_demo.xvla_subtask_evidence import goal_key, digest
    from scene_demo.xvla_subtask_review import XvlaSubtaskReviewClient
    goals=[g for cap in capabilities for g in build_skill_context(cap,request)['criteria']['success_checks']]
    requested_regions={g['destination_id'] for g in goals if g['relation'] in ('in','on','right_of')}
    selected_ids=set(selection.values())
    needed=set(selected_ids)
    for obj in inventory['objects']:
        if obj['object_id'] in selected_ids:needed.update(obj.get('confusable_with',[]))
    selected_inventory={**inventory,'objects':[o for o in inventory['objects'] if o['object_id'] in needed]}
    prompt = ('一次独立执行前审核：同时核对规划对象身份和任务指定目标区域，供检查器直接复用初始审核。根据原始三视图核对；不能只因为编号合法或描述自洽就通过。'
        'request_scope若声明部分执行，核对所选准备步骤或子目标的对象/目的地是否确实推进原需求；未支持的后续步骤不是本次身份检查的额外完成条件。不能接受改变原意的替代操作。'
        '逐个核对selection目标是否确为所指实物；两件操作目标不可同一实物。每个放置goal的destination_id必须另给regions，定位指定隔间/层/承载面/相对区域，不能只框整个装置。具体边界不清则confirmed=false，不猜。'
        '允许某个视角遮挡，但必须逐一看完三个视角；存在冲突、框到空桌面或语义不符则false，不确定null。'
        '框为每幅原图0..1000归一化坐标。描述各写必要短句，只输出JSON：{"checks":[{"target_id":"selection中的键","matches_request":true或false或null,"features":"可见外观","anchors":[{"frame":0,"box":[0,0,100,100]}],"reason":"原图短依据"}],"regions":[{"destination_id":"放置goal的目的地ID","confirmed":true或false,"description":"具体区域","boundary":"实际可见边沿/隔板/参照","anchors":[{"frame":1,"box":[0,0,100,100]}]}]}。门/开关无需放置区域。\n'
        + json.dumps({'request': request, 'request_scope': request_scope, 'selection': selection, 'inventory': selected_inventory,'goals':goals,
                      'skills': [{'instruction': c['instruction'], 'goals': c['goals']} for c in capabilities]}, ensure_ascii=False))
    if joint_draft is not None:
        from scene_demo.xvla_planning import NATIVE_POLICY
        prompt += ('\n同时独立复核整份物品初稿和计划范围。' + NATIVE_POLICY
            + '\n在同一个JSON中保留checks/regions，增加plan_approved布尔值、plan_reason短句、reviewed_views:[0,1,2]。'
              'plan_approved确认所选完整技能、顺序及coverage/未支持结果与原需求相符；缺少能力不能冒充已支持。'
              '再增加inventory_review，使用差异格式：reviewed_ids列出所有审核的原物品ID，removed_ids列删除ID，updates列有修改的完整物品项（不变不重复），coverage和corrections。'
              '每个原物品必须在reviewed_ids或removed_ids中。核对框内实物与跨视角身份；选中物品认错时必须更新描述/框并重新核对checks。'
              '无法确认所选目标则checks不通过，不能靠初稿自洽认可。无可执行技能的计划仍需审核物品及能力缺口。\n'
            + json.dumps({'draft':joint_draft,'full_inventory':inventory},ensure_ascii=False))
    meta = client._request(ctx, paths, prompt)
    raw = client.raw_inventory
    checks = raw.get('checks', [])
    passed = (isinstance(checks, list) and len(checks) == len(selection)
              and {r.get('target_id') for r in checks} == set(selection)
              and all(r.get('matches_request') is True and isinstance(r.get('reason'), str) and r['reason'].strip() for r in checks))
    validator=XvlaSubtaskReviewClient(env_path='/none')
    try:
        for row in checks:
            if not isinstance(row.get('features'),str) or not row['features'].strip():raise ValueError('visible identity features required')
            if not validator._refs(row.get('anchors'),{0,1,2}):raise ValueError('identity image reference required')
        regions=raw.get('regions',[])
        if not isinstance(regions,list) or len(regions)!=len(requested_regions) or {r['destination_id'] for r in regions}!=requested_regions:
            raise ValueError('all selected placement regions required')
        for row in regions:
            if row.get('confirmed') is not True:raise ValueError('selected region unknown')
            if any(not isinstance(row.get(f),str) or not row[f].strip() for f in ('description','boundary')):raise ValueError('region geometry description required')
            if not validator._refs(row.get('anchors'),{0,1,2}):raise ValueError('region image reference required')
    except (ValueError,TypeError,KeyError):passed=False
    result = {'passed': passed, 'response': raw, 'metadata': meta, 'selection': selection,
              'image_sha256':[digest(p) for p in paths],'goal_keys':list({goal_key(g) for g in goals})}
    if joint_draft is not None:
        delta = raw.get('inventory_review', {})
        delta['reviewed_views'] = raw.get('reviewed_views')
        updated = apply_inventory_review(inventory, delta)
        validate_selection(updated, selection, capabilities)
        result['inventory'] = updated
        result['passed'] = passed and raw.get('plan_approved') is True and bool(raw.get('plan_reason'))
    dump(Path(directory) / 'selected_targets_review.json', result)
    return result
