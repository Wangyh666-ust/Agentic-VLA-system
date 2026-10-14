"""Opt-in grounded visual checker; never changes the online checker."""
import copy
import json
import math
import re
from pathlib import Path
from .finish_v1 import vision
from .finish_v1.simple_checker import SimpleQwenClient, encoder_observation

IDENTITIES = {
 'flat_stove':'方形单头炉面与圆形加热圈，旋钮用于开关。',
 'microwave':'带前门和炉腔的箱式微波炉。',
 'wine_bottle':'深绿近黑色长颈玻璃酒瓶，瓶身有白色标签；不是木架支柱或其他小盒子。',
 'alphabet_soup':'圆柱形金属汤罐，银灰罐盖，标签含蓝色Alphabet Soup字样和橙色字母汤图案；不是橙色果汁纸盒，也不是红白Milk纸盒或红绿Tomato Sauce罐。',
 'porcelain_mug':'白色或光照下浅灰色的瓷杯，杯壁有细小点状纹理和白色把手；不是红白花纹咖啡杯，也不是黄白杯。',
 'white_yellow_mug':'浅色杯身带黄色区域、黄色把手的黄白杯；不是灰黑色花纹杯或红白花纹咖啡杯。',
 'akita_black_bowl':'深灰黑色带斑点的圆碗；不是盘子或盒子。'}


def task_definition(context):
    from . import catalog
    task = context['current_subtask']
    desired = context.get('criteria', {}).get('desired_state')
    state_relation = {'on':'turnon','off':'turnoff','open':'open','closed':'close'}.get(desired)
    caps = [c for c in catalog.CAPABILITIES.values() if c.get('object_id') == task.get('target')
            and c.get('target_id') == task.get('destination')
            and (task['operation'] == 'place' or c['goals'][0][0] == state_relation)]
    definitions = {(c['instruction'], c['goals'][0][0]) for c in caps}
    if len(definitions) != 1:
        raise ValueError('missing or ambiguous catalog task definition')
    description, relation = definitions.pop()
    category = re.sub(r'_\d+(?=_|$)', '', task['target'])
    destination = task.get('destination') or ''
    destination_definition = None
    if destination == 'wine_rack_1_top_region':
        destination_definition = '两层木质条板架的上层板面；on rack要求放在上层板面，不要求放进封闭内部。'
    elif 'basket' in destination:
        destination_definition = '编织篮内部。'
    elif 'plate' in destination:
        destination_definition = '盘子顶面；保留当前任务的left/right定义，不以相机屏幕左右固定标记替代。'
    elif 'microwave' in destination:
        destination_definition = '微波炉炉腔内部；放入杯子不附加门关闭条件。'
    return {'task_description': description, 'goal_relation': relation, 'target_category': category,
            'target_visual_identity': IDENTITIES.get(category, catalog.OBJECT_VISUAL_IDENTITY_DEFINITIONS.get(category)),
            'destination_definition': destination_definition, 'desired_state': desired,
            'visual_state_definition': catalog.OBJECT_VISUAL_STATE_DEFINITIONS.get(category)}


def build_prompt(context):
    safe = task_definition(context)
    safe['sensor_frames'] = context['sensor_frames']
    safe['actuator_measurements'] = encoder_observation(context)
    return '\n'.join([
      '只判断当前单项任务的指定目标。静态外观定义不声明当前位置。',
      '先识别target_visual_identity对应物品；其他相似物不能替代。身份无法辨认时target_matches_definition=null，目标关系也返回null。',
      'on只判断物品在目的地顶面，不要求进入内部；in判断进入容器，不附加后续关门。turnon/turnoff/open/close按静态状态定义判断。',
      '不要求已经开爪、退离或归位才允许移交。同步编码器只确定开口，不证明物品到位或是否持物。',
      '三图按标签为wrist、agentview、sideview，同一采样时刻。一个明确视角足够；其他视角遮挡、看不清或显示不同物品不是反证。',
      'false仅用于已识别指定物品且画面明确证明它不满足目标关系；身份或关系未辨认返回null，不用其他物品位置推断false。',
      '仅输出唯一JSONschema：{"target_matches_definition":true/false/null,"goal_achieved":true/false/null,"reason":"指定目标外观及当前位置/关系的画面依据"}。',
      json.dumps(safe, ensure_ascii=False)])


class GroundedQwenClient(SimpleQwenClient):
    def observe_capture(self, capture: dict) -> None:
        timestamp = capture['timestamp']
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
            raise ValueError('invalid capture timestamp')
        fingers = vision._finite_pair(capture['robot']['finger_qpos'], 'finger_qpos')
        frames = capture['frames'] + capture.get('verification_frames', [])
        by_view = {}
        for frame in frames:
            if frame['view'] in by_view and by_view[frame['view']] != frame['rgb_path']:
                raise ValueError('ambiguous capture view')
            by_view[frame['view']] = frame['rgb_path']
        if not all(k in by_view for k in ['wrist','agentview','sideview']):
            raise ValueError('capture requires three views')
        stored = {'timestamp': timestamp, 'views': by_view, 'finger_qpos': fingers}
        if not hasattr(self, '_captures'):
            self._captures = {}
        for path in by_view.values():
            self._captures[str(path)] = stored

    def analyze(self, context: dict, image_paths: list[str]) -> dict:
        wrist = [str(path) for path, frame in zip(image_paths, context['sensor_frames']) if frame['view'] == 'wrist']
        if len(wrist) != 1 or wrist[0] not in getattr(self, '_captures', {}):
            raise ValueError('original wrist path has no capture')
        capture = self._captures[wrist[0]]
        if len(image_paths) != len(context['sensor_frames']) or any(self._captures.get(str(p)) is not capture for p in image_paths):
            raise ValueError('original paths differ from same capture')
        encoder_observation(context)
        if any(f['timestamp'] != capture['timestamp'] for f in context['sensor_frames']) or context['actuator_measurements']['timestamp'] != capture['timestamp']:
            raise ValueError('capture/image/encoder timestamp mismatch')
        if any(abs(a-b) > 1e-9 for a,b in zip(context['actuator_measurements']['finger_qpos'], capture['finger_qpos'])):
            raise ValueError('capture finger mismatch')
        if any(capture['views'].get(frame['view']) != str(path)
               for path, frame in zip(image_paths, context['sensor_frames'])):
            raise ValueError('original camera label/path mismatch')
        self._source_context = copy.deepcopy(context)
        self._source_image_paths = list(image_paths)
        actual = copy.deepcopy(context)
        actual['sensor_frames'] = [{'frame':i,'timestamp':capture['timestamp'],'view':view} for i,view in enumerate(['wrist','agentview','sideview'])]
        paths = [capture['views'][view] for view in ['wrist','agentview','sideview']]
        prompt = build_prompt(actual)
        result = self._request(actual, paths, prompt)
        result.update(prompt_text=prompt, checker='xvla_grounded_candidate_v1',
                      visual_goal_verdict={'goal_achieved':self.last_normalized_goal,'reason':self.last_two_question_verdict['reason']},
                      two_question_verdict=self.last_two_question_verdict)
        return result

    def parse_verdict(self, content: str) -> dict:
        raw = vision._parse_verdict_content(content)
        if set(raw) != {'target_matches_definition','goal_achieved','reason'}:
            raise ValueError('response must use unique grounded schema')
        for key in ['target_matches_definition','goal_achieved']:
            if raw[key] is not None and type(raw[key]) is not bool:
                raise ValueError(key+' must be bool or null')
        if not isinstance(raw['reason'], str):
            raise ValueError('reason must be string')
        goal = raw['goal_achieved'] if raw['target_matches_definition'] is True else None
        self.last_two_question_verdict = raw
        self.last_normalized_goal = goal
        verdict = {k:None for k in vision._BOOL_FIELDS}
        verdict.update(at_destination=goal, operation_achieved=goal, boxes={}, evidence=[raw['reason']], unknown_reasons=[])
        return verdict

    def record_response(self, context, image_paths, prompt, request_bytes, raw, index, status):
        super().record_response(context,image_paths,prompt,request_bytes,raw,index,status)
        path = Path(self.audit_dir) / f'call_{index:02d}_request.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        record.update(source_context=self._source_context, source_image_paths=self._source_image_paths)
        path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
