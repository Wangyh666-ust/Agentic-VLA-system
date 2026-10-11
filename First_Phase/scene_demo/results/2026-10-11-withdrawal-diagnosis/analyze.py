"""Offline geometry diagnosis. No task environment, reset, step or model calls."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import mujoco
import numpy as np
from robosuite.models.grippers import PandaGripper
from robosuite.models.robots import Panda

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'First_Phase'))
from scene_demo.finish_v1.geometry import clouds, mesh_surface_distance, shape_distance

SCENE = ROOT / 'First_Phase/scene_demo'
CAMERA_FIX = np.diag([1., -1., -1., 1.])
BASE = np.array([-.66, 0., .912])
robot = Panda(idn=0)
gripper = PandaGripper(idn=0)
robot.add_gripper(gripper)
xml = robot.get_xml()
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'robot0_eye_in_hand')
site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'gripper0_grip_site')


def forward(q, fingers):
    data.qpos[:7] = q
    data.qpos[7:9] = fingers
    mujoco.mj_forward(model, data)


def camera_pose():
    t = np.eye(4)
    t[:3, :3] = data.cam_xmat[camera].reshape(3, 3)
    t[:3, 3] = data.cam_xpos[camera] + BASE
    return t @ CAMERA_FIX


def bounds(gid):
    kind = model.geom_type[gid]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gid]
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        return v.min(0), v.max(0)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return -model.geom_size[gid], model.geom_size[gid]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        h = np.full(3, model.geom_size[gid][0])
        return -h, h
    raise ValueError(f'unsupported shape {kind}')


def shape(gid):
    lo, hi = bounds(gid)
    rotation = data.geom_xmat[gid].reshape(3, 3)
    center = rotation @ ((lo + hi) / 2) + data.geom_xpos[gid] + BASE
    return {'center': center, 'rotation': rotation, 'half_extents': (hi - lo) / 2}


def obstacle_points(frames):
    # Same visual CAD surface mask as the saved sensors.py implementation.
    points = clouds(frames)
    mask = np.zeros(len(points), dtype=bool)
    for name in list(robot.visual_geoms) + list(gripper.visual_geoms):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        lo, hi = bounds(gid)
        local = (points - data.geom_xpos[gid] - BASE) @ data.geom_xmat[gid].reshape(3, 3)
        candidates = np.all((local >= lo - .003) & (local <= hi + .003), axis=1) & ~mask
        if not candidates.any():
            continue
        loc = local[candidates]
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            faces = model.mesh_face[model.mesh_faceadr[mid]:model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
            distance = mesh_surface_distance(loc, vertices, faces)
        elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            distance = abs(shape_distance(loc, {'center': np.zeros(3), 'rotation': np.eye(3), 'half_extents': model.geom_size[gid]}))
        else:
            distance = abs(np.linalg.norm(loc, axis=1) - model.geom_size[gid][0])
        mask[np.where(candidates)[0][distance <= .003]] = True
    return points[~mask], {'input': len(points), 'self_mask_removed': int(mask.sum())}


output = {'physical_actions': 0, 'model_requests': 0,
          'method': 'Static Panda FK checked against saved MountedPanda wrist camera poses; saved depth clouds and native MuJoCo geom distance. No full scene or dynamics replay.',
          'robot_xml_sha256': hashlib.sha256(xml.encode()).hexdigest(),
          'cases': {}, 'home_suffix_comparison': {}}
campaign = SCENE / 'results/2026-10-11-assisted-combos'
for name in ('wine_to_rack', 'stove_on'):
    path = next(campaign.rglob('cap_02_' + name + '_*'))
    finish = json.loads((path / 'pilot_result.json').read_text())['finish']
    rows = [json.loads(line) for line in (path / 'finish_actions.jsonl').read_text().splitlines()]
    movement = [row for row in rows if row['phase'] == 'movement']
    release = [row for row in rows if row['phase'] == 'release']
    final = movement[-1]
    q = np.array(final['arm_qpos'])
    fingers = np.array(final['finger_qpos'])
    target = np.array(finish['plan']['waypoints'][1])
    segments = int(np.ceil(max(abs(target - q)) / .003))
    candidate = q + (target - q) / segments
    camera_errors = []
    for index, screen in enumerate(finish['movement_screens']):
        row = movement[10 * (index + 1) - 1]
        forward(row['arm_qpos'], row['finger_qpos'])
        saved = np.array(next(frame for frame in screen['frames'] if frame['view'] == 'wrist')['T_world_camera'])
        camera_errors.append(float(np.max(abs(camera_pose() - saved))))
    forward(finish['plan']['waypoints'][0], release[-1]['finger_qpos'])
    start_xyz = data.site_xpos[site].copy() + BASE
    forward(q, fingers)
    actual_xyz = data.site_xpos[site].copy() + BASE
    contact = json.loads((path / 'independent_finish.jsonl').read_text().splitlines()[-1])['contacts'][0]
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, contact['geom2_name'])
    rotation = data.geom_xmat[gid].reshape(3, 3).copy()
    local_contact = (np.array(contact['pos']) - data.geom_xpos[gid] - BASE) @ rotation
    contact_deltas = {}
    for label, qq in (('next_interpolation', candidate), ('old_first_target', target)):
        forward(qq, fingers)
        moved = data.geom_xmat[gid].reshape(3, 3) @ local_contact + data.geom_xpos[gid] + BASE
        delta = moved - np.array(contact['pos'])
        contact_deltas[label] = {'delta_mm': (1000 * delta).tolist(),
                                 'outward_mm': 1000 * float(np.array(contact['outward_normal']) @ delta)}
    result = {'source': str(path.relative_to(ROOT)), 'reason': finish['reason'],
              'counts': finish['counts'], 'waypoint': finish['movement_progress'],
              'camera_max_matrix_errors': camera_errors,
              'eef_executed_delta_mm': (1000 * (actual_xyz - start_xyz)).tolist(),
              'contact': contact, 'contact_point_counterfactual': contact_deltas}
    assert max(camera_errors) < 1e-4, 'static model camera cross-check failed'
    if name == 'stove_on':
        # An additional pose checks the model independently of the blocked screen.
        forward(finish['plan']['waypoints'][0], release[-1]['finger_qpos'])
        saved = json.loads((path / 'finish/sensors/cap000107_wrist.json').read_text())['T']
        result['second_pose_camera_max_matrix_error'] = float(np.max(abs(camera_pose() - np.array(saved))))
        assert result['second_pose_camera_max_matrix_error'] < 1e-4
        forward(q, fingers)
        points, stats = obstacle_points(finish['movement_screens'][0]['frames'])
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'gripper0_finger1_collision')
        distance = shape_distance(points, shape(gid))
        point = points[int(np.argmin(distance))]
        result['blocking_point_world'] = point.tolist()
        result['cloud'] = stats
        tree = ET.fromstring(xml)
        ET.SubElement(tree.find('worldbody'), 'geom', {
            'name': 'depth_point_probe', 'type': 'sphere', 'size': '0.000001',
            'pos': ' '.join(str(x) for x in point - BASE)})
        probe_model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))
        probe_data = mujoco.MjData(probe_model)
        finger_id = mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_GEOM, 'gripper0_finger1_collision')
        point_id = mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_GEOM, 'depth_point_probe')
        result['point_geometry_checks'] = {}
        for label, qq in (('current', q), ('next_interpolation', candidate), ('old_first_target', target)):
            forward(qq, fingers)
            probe_data.qpos[:7] = qq
            probe_data.qpos[7:9] = fingers
            mujoco.mj_forward(probe_model, probe_data)
            result['point_geometry_checks'][label] = {
                'obb_min_clearance_m': float(np.min(shape_distance(points, shape(gid)))),
                'native_collision_shape_probe_distance_m': float(mujoco.mj_geomDistance(probe_model, probe_data, finger_id, point_id, 1., None))}
        saved_diag = finish['movement_screens'][0]['diagnostic']
        assert abs(result['point_geometry_checks']['current']['obb_min_clearance_m'] - saved_diag['initial_clearance']) < 1e-7
        assert abs(result['point_geometry_checks']['next_interpolation']['obb_min_clearance_m'] - saved_diag['current_clearance']) < 1e-7
    output['cases'][name] = result

live = (SCENE / 'finish_v1/control.py').read_text()
marker = "        points = jh.joint_waypoints(actualq, ref['homeq'])"
for archive in ('source_snapshot_atomic.zip', 'source_snapshot_repair.zip'):
    with zipfile.ZipFile(SCENE / 'results/2026-10-11-goal-finish/evidence' / archive) as z:
        old = z.read('First_Phase/scene_demo/finish_v1/control.py').decode().replace('\r\n', '\n')
    equal = live[live.index(marker):].strip() == old[old.index(marker):].strip()
    assert equal, 'successful home execution suffix changed'
    output['home_suffix_comparison'][archive] = equal
(Path(__file__).parent / 'diagnosis.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'ok': True, 'physical_actions': 0, 'model_requests': 0,
                  'home_suffix_unchanged': True, 'output': str(Path(__file__).parent / 'diagnosis.json')}, ensure_ascii=False))
