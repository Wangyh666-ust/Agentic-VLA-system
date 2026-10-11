"""Visual task goal + authoritative synchronized gripper opening feedback."""
import json
import hashlib
import math
import re
from pathlib import Path
from . import vision

CHECKER_NAME = 'visual_goal_encoder_v4'
OPEN_TOLERANCE_M = .002


def encoder_observation(context):
    """Describe measured opening, never a commanded opening or visual overlap."""
    measurements = context['actuator_measurements']
    fingers = vision._finite_pair(measurements['finger_qpos'], 'finger_qpos')
    reference = vision._finite_pair(measurements['standard_open_finger_qpos'], 'standard_open_finger_qpos')
    timestamps = [measurements['timestamp']] + [f['timestamp'] for f in context['sensor_frames']]
    if len(timestamps) < 2 or any(isinstance(t, bool) or not isinstance(t, (int, float))
                                   or not math.isfinite(t) for t in timestamps):
        raise ValueError('encoder/image timestamps must be finite numbers')
    if any(abs(t - timestamps[0]) > 1e-9 for t in timestamps[1:]):
        raise ValueError('encoder and image timestamps differ')
    error = max(abs(a - b) for a, b in zip(fingers, reference))
    return {'timestamp': timestamps[0], 'finger_qpos': fingers,
            'standard_open_finger_qpos': reference, 'position_unit': 'm',
            'standard_open_tolerance_m': OPEN_TOLERANCE_M,
            'max_standard_open_error_mm': error * 1000,
            'at_standard_open': error <= OPEN_TOLERANCE_M}


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
        'actuator_measurements': encoder_observation(context),
    }
    category = re.sub(r'_\d+(?=_|$)', '', task.get('target', ''))
    if task['operation'] == 'turn' and category in catalog.OBJECT_VISUAL_STATE_DEFINITIONS:
        safe['visual_state_definition'] = catalog.OBJECT_VISUAL_STATE_DEFINITIONS[category]
    return '\n'.join([
        '你是机器人子任务的结束状态检查器。',
        '你的任务目标：根据当前子任务和同步观测，判断任务目标是否达到、是否可以移交收尾。VLA在个别任务末段可能遗漏最后的松手和退离动作；这仅说明检查器的用途，不表示本次目标已经达到，也不表示本次仍在夹持。',
        '输入包括：当前子任务描述、当前相机图像，以及与图像同步的夹爪编码器数据。',
        '两张图来自同一采样时刻：wrist是夹爪上的第一人称视角，sideview是外部第三人称视角。视觉只负责判断当前子任务要求的物品位置关系或操作状态。',
        'goal_achieved：只检查current_subtask.description明确要求的目标，不增加任务未要求的支撑、姿态或更精确的位置条件。不要要求已经松手、解除接触、退离或归位；这些由目标达到后的收尾器执行。是否仍夹持不否定目标已达到。',
        '夹持状态由程序根据同步实测编码器决定：at_standard_open=true表示夹爪已达到标准开口，按已松手、未夹持处理。遮挡、二维投影重叠或视觉中的接近不能推翻此读数，也不能另报仍夹持。at_standard_open=false表示需要开爪，不单凭闭合证明夹有物品。编码器开口不能证明物品已到位，也不能证明退离路径安全；不要混入goal_achieved判断。',
        '只依据当前实际画面判断目标，不接受计划、检查时机或阶段名称作为完成证据。无法判断目标时返回null，不猜测。reason只描述目标的位置关系或操作状态依据。',
        '如提供visual_state_definition，它只解释该场景资产的视觉标志含义，不声明当前状态；根据实际画面判断是否满足desired_state。',
        '只输出JSON：{"goal_achieved":true或false或null,"reason":"简短目标画面依据"}。',
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
        goal = raw.get('goal_achieved')
        if goal is not None and type(goal) is not bool:
            raise ValueError('goal_achieved must be bool or null')
        reason = raw.get('reason', '')
        if not isinstance(reason, str):
            raise ValueError('reason must be string')
        # Keep historical responses intact, but never turn visual contact into
        # release or clearance evidence. Reconcile supplies measured opening.
        verdict = {key: None for key in vision._BOOL_FIELDS}
        verdict.update(operation_achieved=goal, at_destination=goal,
                       boxes={}, evidence=[reason], unknown_reasons=[])
        self.last_two_question_verdict = raw
        return verdict

    def analyze(self, context, image_paths):
        prompt = build_prompt(context)
        result = self._request(context, image_paths, prompt)
        result['prompt_text'] = prompt
        result['checker'] = CHECKER_NAME
        result['visual_goal_verdict'] = self.last_two_question_verdict
        # Historical report readers retain this alias; it is raw, not fused.
        result['two_question_verdict'] = self.last_two_question_verdict
        return result


def goal_achieved(context, verdict):
    return verdict.get('operation_achieved') is True


def reconcile(context, verdict, robot, reference):
    out = vision.validate_verdict(verdict)
    measurements = context['actuator_measurements']
    encoder = encoder_observation(context)
    for key, values in (('finger_qpos', robot['finger_qpos']),
                        ('standard_open_finger_qpos', reference['finger_home'])):
        values = vision._finite_pair(list(values), key)
        if any(abs(a - b) > 1e-9 for a, b in zip(measurements[key], values)):
            raise ValueError('encoder/reference mismatch')
    out['released'] = encoder['at_standard_open']
    # Physical hand clearance belongs to the retreat/path controller. Opening
    # fingers must not fabricate clearance, nor allow visual overlap to veto it.
    out['clear_of_target'] = None
    if out['released']:
        out['evidence'].append('同步实测编码器已达到标准开口，按已松手、未夹持处理；遮挡或投影重叠不改变此判断。')
    else:
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
    # This is checker acceptance, not completion of retreat/home. Runtime always
    # runs the atomic finishing controller, even for this initial state.
    return vision._decision('complete', 'purpose achieved and encoder confirms open; controller verifies retreat/home')
