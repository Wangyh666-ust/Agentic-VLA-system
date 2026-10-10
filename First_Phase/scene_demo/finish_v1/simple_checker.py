"""User's two-question checker; no simulator object truth enters the model."""
import json
import hashlib
from pathlib import Path
from . import vision


def build_prompt(context):
    task = context['current_subtask']
    safe = {
        'user_request': context['user_request'],
        'current_subtask': {k: task.get(k) for k in ('operation', 'target', 'destination')},
        'desired_state': context.get('criteria', {}).get('desired_state'),
        'sensor_frames': context['sensor_frames'],
        'actuator_measurements': context['actuator_measurements'],
    }
    return '\n'.join([
        '你是机器人子任务检查器。只回答两组问题，依据所给照片和同步夹爪编码器。',
        '1. 子任务目的是否达成？放置：指定物品是否已经放在指定目的地，由那里承托；仅悬在上方不算。架子上允许平放或斜放，除非用户明确要求竖立。入篮：指定物品是否已经在篮子内部。开关灶台：当前炉圈/旋钮是否达到要求；红色发热炉圈表示开火，不要求看到之前旋转动作。',
        '2. 夹爪是否仍夹持或接触目标/局部支撑面？结合图像和编码器。夹爪闭合不证明持物，夹爪未张开不能否定物品已经放到位。照片中物品已搭在架面而手还靠近瓶颈时，放置目的可以已达成，手部尚未离开。',
        '两组独立判断：第一组不要求松手、退离、归位。看不清返回null；不要把遮挡编成悬空或接触。编码器只说明开合状态，不说明支撑关系。',
        '为兼容现有接口，只输出一个JSON，键必须恰为：target_visible,destination_visible,at_destination,supported,released,stable,operation_achieved,clear_of_target,boxes,evidence,unknown_reasons。',
        '第一组：target_visible/destination_visible表示看见指定物体/目的地；at_destination表示已到指定位置；supported表示由指定位置承托；operation_achieved表示当前子任务目的达成。',
        '第二组：released表示手指已不持握/接触目标；clear_of_target表示手部已与目标及其局部支撑面分离。两者根据照片判断，编码器闭合本身不能断言仍持物。',
        '所有上述状态用true/false/null。stable固定为null（本检查不额外要求视觉稳定性）。turn任务的destination_visible/at_destination/supported可为null。',
        'boxes仅用于已有退离路径定位，不作为任务成功条件：以最新照片逐视角输出target/destination/gripper各自归一化[x1,y1,x2,y2]或null；看不清可为空字典。不得跨视角复制坐标。',
        'evidence和unknown_reasons是简短字符串数组。说明第一组与第二组各自画面依据。',
        json.dumps(safe, ensure_ascii=False),
    ])


class SimpleQwenClient(vision.QwenClient):
    simple_handoff = True

    def record_response(self, context, image_paths, prompt, request_bytes, raw, index, status):
        directory = Path(self.audit_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'call_{index:02d}_response.json').write_bytes(raw)
        record = {'prompt': prompt, 'context': context, 'http_status': status,
                  'request_sha256': hashlib.sha256(request_bytes).hexdigest(),
                  'images': [{'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in image_paths]}
        (directory / f'call_{index:02d}_request.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def parse_verdict(self, content):
        raw = vision._parse_verdict_content(content)
        # Missing properties remain unknown; do not manufacture a positive goal.
        verdict = {key: raw.get(key) for key in vision._BOOL_FIELDS}
        verdict['stable'] = None
        boxes = raw.get('boxes') or {}
        verdict['boxes'] = {view: {key: item.get(key) for key in ('target', 'destination', 'gripper')}
                            for view, item in boxes.items() if isinstance(item, dict)}
        verdict['evidence'] = raw.get('evidence', [])
        verdict['unknown_reasons'] = raw.get('unknown_reasons', [])
        return verdict

    def analyze(self, context, image_paths):
        prompt = build_prompt(context)
        result = self._request(context, image_paths, prompt)
        result['prompt_text'] = prompt
        result['checker'] = 'simple_goal_contact_v1'
        return result


def goal_achieved(context, verdict):
    if verdict.get('operation_achieved') is not True:
        return False
    if context['current_subtask']['operation'] == 'place':
        return verdict.get('at_destination') is True and verdict.get('supported') is True
    return True


def reconcile(context, verdict, robot, reference):
    out = vision.validate_verdict(verdict)
    measurements = context['actuator_measurements']
    latest = max(f['timestamp'] for f in context['sensor_frames'])
    if abs(measurements['timestamp'] - latest) > 1e-9:
        raise ValueError('encoder and image timestamps differ')
    for key, values in (('finger_qpos', robot['finger_qpos']),
                        ('standard_open_finger_qpos', reference['finger_home'])):
        if len(values) != 2 or any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(measurements[key], values)):
            raise ValueError('encoder/reference mismatch')
    opened = max(abs(float(a) - float(b)) for a, b in zip(robot['finger_qpos'], reference['finger_home'])) <= .002
    if not opened:
        out['released'] = False
        out['clear_of_target'] = False
        out['evidence'].append('夹爪尚未达到标准开口，需要发出开爪命令；不据此推断物品未到位或仍被持有。')
    return out


def decide(context, verdict):
    achieved = verdict.get('operation_achieved')
    if achieved is None:
        return vision._decision('unknown', 'subtask purpose unknown')
    if not goal_achieved(context, verdict):
        return vision._decision('incomplete', 'subtask purpose not achieved')
    if verdict.get('released') is not True:
        return vision._decision('needs_release', 'purpose achieved; open fingers', allow_release=True)
    if verdict.get('clear_of_target') is not True:
        return vision._decision('needs_retreat', 'purpose achieved; withdraw hand', allow_retreat=True)
    return vision._decision('complete', 'purpose achieved and hand separated')
