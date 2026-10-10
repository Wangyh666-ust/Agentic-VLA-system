"""User's two-question checker; no simulator object truth enters the model."""
import json
import hashlib
import re
from pathlib import Path
from . import vision


def build_prompt(context):
    task = context['current_subtask']
    def display_name(identifier):
        if identifier is None:
            return None
        return re.sub(r'_\d+(?=_|$)', '', identifier).replace('_', ' ')
    from scene_demo import catalog
    descriptions = [cap['instruction'] for cap in catalog.CAPABILITIES.values()
                    if cap.get('object_id') == task.get('target') and cap.get('target_id') == task.get('destination')
                    and (task['operation'] != 'turn' or cap['goals'][0][0] == ('turnoff' if context.get('criteria',{}).get('desired_state') == 'off' else 'turnon'))]
    description = context.get('criteria',{}).get('task_description') or (descriptions[0] if descriptions else
                   f"{task['operation']} {display_name(task.get('target'))} at {display_name(task.get('destination'))}")
    safe = {
        'user_request': context['user_request'],
        'current_subtask': {'operation': task['operation'], 'description': description},
        'desired_state': context.get('criteria', {}).get('desired_state'),
        'sensor_frames': context['sensor_frames'],
        'actuator_measurements': context['actuator_measurements'],
    }
    return '\n'.join([
        '你是机器人子任务检查器。根据当前照片和同步夹爪编码器，只回答两个问题。',
        '1. goal_achieved：子任务目的是否达成？放架任务，只问指定物品是否已在架子上；放盘任务，只问碗是否已在盘子上；入篮任务，只问指定物品是否已在篮内；开关任务，只问要求的开关状态是否已达到。不要自行寻找任务没有要求的另一个目的地或更精确的摆放位置。如果描述要求在架子上，观察又说物品在架子上，就不能又因没有到另一个位置而否决。这个判断用来触发收尾，仍夹持或还没退离不否决第一问；松手退离由收尾器执行。不单独检查接触、承托、姿态。',
        '2. holding_or_contact：夹爪是否仍夹持或接触当前目标？看手指与目标的实际位置关系，并结合编码器。腕部镜头下方或照片中央出现物品，不代表物品被夹持。闭合不独自证明持物；达到标准开口也不独自证明手已离开目标。阶段标签不是动作已完成的证据。',
        '两问独立判断。看不清返回null，不猜测。开火看当前炉圈或旋钮状态，红色发热炉圈表示已开火。',
        '只核对current_subtask.description这一个子任务的自然语言目的；不要添加该描述没有要求的目标区域、姿态或条件。',
        '只输出JSON：{"goal_achieved":true或false或null,"holding_or_contact":true或false或null,"reason":"简短画面依据"}。',
        json.dumps(safe, ensure_ascii=False),
    ])


class SimpleQwenClient(vision.QwenClient):
    simple_handoff = True
    request_options = {'enable_thinking': True, 'thinking_budget': 1024}

    def record_response(self, context, image_paths, prompt, request_bytes, raw, index, status):
        directory = Path(self.audit_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'call_{index:02d}_response.json').write_bytes(raw)
        record = {'prompt': prompt, 'context': context, 'http_status': status,
                  'request_options': self.request_options,
                  'request_sha256': hashlib.sha256(request_bytes).hexdigest(),
                  'images': [{'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in image_paths]}
        (directory / f'call_{index:02d}_request.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def parse_verdict(self, content):
        raw = vision._parse_verdict_content(content)
        goal, contact = raw.get('goal_achieved'), raw.get('holding_or_contact')
        for name, value in [('goal_achieved', goal), ('holding_or_contact', contact)]:
            if value is not None and type(value) is not bool:
                raise ValueError(name + ' must be bool or null')
        reason = raw.get('reason', '')
        if not isinstance(reason, str):
            raise ValueError('reason must be string')
        # Legacy interface aliases, NOT separate model observations. No support gate.
        verdict = {key: None for key in vision._BOOL_FIELDS}
        verdict.update(operation_achieved=goal, at_destination=goal,
                       released=None if contact is None else not contact,
                       clear_of_target=None if contact is None else not contact,
                       boxes={}, evidence=[reason], unknown_reasons=[])
        self.last_two_question_verdict = raw
        return verdict

    def analyze(self, context, image_paths):
        prompt = build_prompt(context)
        result = self._request(context, image_paths, prompt)
        result['prompt_text'] = prompt
        result['checker'] = 'two_question_goal_contact_v2'
        result['two_question_verdict'] = self.last_two_question_verdict
        return result


def goal_achieved(context, verdict):
    return verdict.get('operation_achieved') is True


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
