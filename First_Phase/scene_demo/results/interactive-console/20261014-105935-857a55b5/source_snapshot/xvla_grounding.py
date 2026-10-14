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
    request_options = {'enable_thinking': True, 'thinking_budget': 2048, 'max_tokens': 6500}

    def __init__(self, env_path='/home/yhwang/fyp/scene_demo/hermes_home/.env',
                 base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
                 model=None, max_calls=18):
        super().__init__(env_path=env_path, base_url=base_url,
                         model=model if model is not None else os.environ.get('SCENE_PLANNER_MODEL', 'qwen3.8-max'),
                         max_calls=max_calls)

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


def inspect_scene(capture, request, directory, client=None):
    from scene_demo.xvla_process_evidence import capture_frames
    frames = capture_frames(capture)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    client = client or InventoryClient(max_calls=2)
    client.audit_dir = directory / 'model_requests'
    ctx = {'sensor_frames': [{'frame': i, 'view': f['view'], 'timestamp': capture['timestamp']} for i, f in enumerate(frames)]}
    paths = [f['rgb_path'] for f in frames]
    prompt = ('根据同步三视图建立现场物体清单，供规划和后续检查共用。用户任务：' + request + '\n'
        '先辨认桌面可见物品、容器和装置，包括目标和易混淆物。外观只写实际可见颜色、形状、把手、包装；看不清文字不要猜品牌。'
        '同一实物跨视角只分配一个object_id；不同实物不能合并。识别门/抽屉时可用装置整体加可见活动部件描述。'
        '每个物体必须逐一检查frame 0/1/2；遮挡视角如实记录，不要求三张都看得见。至少一个清晰参照且其他视角无身份冲突才confirmed=true。'
        '框采用每幅原图0..1000的[x1,y1,x2,y2]，包住真实物体，不按期望位置猜框。'
        '输出JSON：{"objects":[{"object_id":"obj_1","description":"可见特征及初始相对位置",'
        '"confirmed":true,"anchors":[{"frame":1,"box":[100,200,300,400]}],'
        '"view_observations":[{"frame":0,"visibility":"occluded","reason":"遮挡情况"},'
        '{"frame":1,"visibility":"visible","reason":"辨认依据"},{"frame":2,"visibility":"partial","reason":"对应依据"}],'
        '"confusable_with":[],"role":"target/destination/distractor/uncertain"}],'
        '"coverage":{"targets_identifiable":true,"destination_visible":true,"state_change_visible":true,"reason":"画面覆盖依据"}}。')
    first_meta = client._request(ctx, paths, prompt)
    first = validate_inventory(client.raw_inventory)
    dump(directory / 'inventory_first.json', {'inventory': first, 'metadata': first_meta})
    review_prompt = (prompt + '\n现在重新从三张原图审核下列初稿，初稿可能框到空桌面、混淆目标、把同一物体数两遍。'
        '逐个核对框内确有该物体及跨视角对应，错误必须修订；无法确认的物体标confirmed=false。'
        '不要因为初稿描述自洽就确认。保留相同实物的ID，可修订描述/框，解释corrections。输出完整相同结构。\n初稿：'
        + json.dumps(first, ensure_ascii=False))
    review_meta = client._request(ctx, paths, review_prompt)
    final = validate_inventory(client.raw_inventory)
    final.update(version=1, captured_at=capture['timestamp'], reviewed_from_views=list(VIEWS),
                 source_images=[{'frame': i, 'view': f['view'], 'path': f['rgb_path']} for i, f in enumerate(frames)])
    from scene_demo.xvla_subtask_evidence import digest
    for image in final['source_images']: image['sha256'] = digest(image['path'])
    dump(directory / 'inventory_confirmed.json', {'inventory': final, 'metadata': review_meta})
    dump(directory / 'capture.json', capture)
    picture = directory / 'planning_three_views.png'; triptych(frames, picture)
    return final, str(picture)


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


def seed_checker(client, inventory, selection, capture, context):
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
    store.initial_review_done = True; store.revision += 1
    store.append('planning_bindings', {'selection': selection, 'inventory': inventory, 'bindings': store.bindings})
    client._reference = item


def verify_plan(inventory, selection, request, capabilities, directory, client=None, request_scope=None):
    """Recheck the selected real objects against the request, before any motion."""
    client = client or InventoryClient(max_calls=1)
    client.audit_dir = Path(directory) / 'selection_requests'
    ctx = {'sensor_frames': [{'frame': i, 'view': v, 'timestamp': inventory['captured_at']} for i, v in enumerate(VIEWS)]}
    paths = [i['path'] for i in inventory['source_images']]
    prompt = ('执行前审核规划选择与原始三视图。核对用户要求实际对应的物品及目的地；不能只因为编号合法或描述自洽就通过。'
        'request_scope若声明部分执行，核对所选准备步骤或子目标的对象/目的地是否确实推进原需求；未支持的后续步骤不是本次身份检查的额外完成条件。不能接受改变原意的替代操作。'
        '逐个核对selection的目标在原图中是否确实为用户所指的实物；两件目标不可是同一实物；目的地部件可绑定装置整体。'
        '允许某个视角遮挡，但必须逐一看完三个视角；存在冲突、框到空桌面或语义不符则false，不确定null。'
        '只输出JSON：{"checks":[{"target_id":"selection中的键","matches_request":true或false或null,"reason":"原图依据"}]}。\n'
        + json.dumps({'request': request, 'request_scope': request_scope, 'selection': selection, 'inventory': inventory,
                      'skills': [{'instruction': c['instruction'], 'goals': c['goals']} for c in capabilities]}, ensure_ascii=False))
    meta = client._request(ctx, paths, prompt)
    raw = client.raw_inventory
    checks = raw.get('checks', [])
    passed = (isinstance(checks, list) and len(checks) == len(selection)
              and {r.get('target_id') for r in checks} == set(selection)
              and all(r.get('matches_request') is True and isinstance(r.get('reason'), str) and r['reason'].strip() for r in checks))
    result = {'passed': passed, 'response': raw, 'metadata': meta, 'selection': selection}
    dump(Path(directory) / 'selected_targets_review.json', result)
    return result
