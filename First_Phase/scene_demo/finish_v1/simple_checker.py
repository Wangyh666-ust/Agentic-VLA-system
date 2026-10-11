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
        # Phase labels describe controller bookkeeping, not measured completion.
        # In particular, never send a stale post_execution label to the model.
        'actuator_measurements': {
            key: context['actuator_measurements'][key]
            for key in ('timestamp', 'finger_qpos', 'max_abs_arm_qvel', 'standard_open_finger_qpos')
            if key in context['actuator_measurements']
        },
    }
    return '\n'.join([
        '你是机器人子任务的结束状态检查器。',
        '你的任务目标：VLA在执行某些子任务时，可能已经达到任务要求的位置或操作状态，但遗漏最后的松手、解除接触和退离动作。你负责根据当前子任务和实际观测，判断是否已经达到可以停止VLA、交给收尾器执行这些末段动作的条件。',
        '输入包括：当前子任务描述、当前相机图像，以及与图像同步的夹爪编码器数据。',
        '两张图来自同一采样时刻：wrist是夹爪上的第一人称视角，sideview是外部第三人称视角。请结合两张图判断物品与目的地的位置关系；夹爪开合不能证明物品已经到位。',
        '1. goal_achieved：当前子任务要求的物品位置关系或操作状态是否已经达到，从而可以结束VLA执行、触发收尾？只检查current_subtask.description明确要求的目标。不要额外要求支撑、特定姿态或更精确的位置。不要要求已经松手、解除接触、退离或归位：这些正是随后由收尾器补全的动作。仍然夹持或接触目标，不否定已经达到收尾触发条件。',
        '2. holding_or_contact：夹爪是否仍夹持或接触当前操作对象？结合图像与同步夹爪编码器判断。编码器开合状态不能单独证明持物或分离。',
        '根据实际观测判断，不接受计划中“已经完成”的自述。无法判断时返回null，不猜测。',
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
