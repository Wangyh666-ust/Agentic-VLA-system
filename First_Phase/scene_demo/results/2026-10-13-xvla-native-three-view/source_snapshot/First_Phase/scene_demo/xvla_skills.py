"""Registered native X-VLA skills and conservative interface routing."""
from __future__ import annotations
import copy

NATIVE_SKILL_GUIDANCE = """[native long-horizon skills]
Prefer one call to a listed native long-horizon skill only when all requested actions, objects, destinations and their order exactly match it. Otherwise split into the listed public executable capabilities. Never add closing a door, turning on a stove, or extra objects merely to use a skill. Intermediate sub-actions do not end the skill and do not start task-boundary resets. Candidate skills and Hermes integration have not yet been physically tested; do not exaggerate reliability.
"""

NATIVE_LONG_SKILLS = {
    'xvla_basket_both': {
        'instruction': 'put both the alphabet soup and the tomato sauce in the basket',
        'goals': [['in','alphabet_soup_1','basket_1_contain_region'],['in','tomato_sauce_1','basket_1_contain_region']],
        'scene_ids': ['basket_two'], 'native_task': {'suite':'libero_10','task_id':0},
        'component_capability_ids': ['soup_to_basket','sauce_to_basket'],
    },
    'xvla_mugs_both': {
        'instruction': 'put the white mug on the left plate and put the yellow and white mug on the right plate',
        'goals': [['on','porcelain_mug_1','plate_1'],['on','white_yellow_mug_1','plate_2']],
        'scene_ids': ['mugs_two'], 'native_task': {'suite':'libero_10','task_id':4},
        'component_capability_ids': ['white_mug_left','yellow_mug_right'],
    },
    'xvla_microwave_place_close': {
        'instruction': 'put the yellow and white mug in the microwave and close it',
        'goals': [['in','white_yellow_mug_1','microwave_1_heating_region'],['close','microwave_1']],
        'scene_ids': ['microwave_mug'], 'native_task': {'suite':'libero_10','task_id':9},
        'component_capability_ids': ['mug_to_microwave','microwave_close'],
    },
}
for _skill in NATIVE_LONG_SKILLS.values():
    _skill.update(backends=['xvla'], skill_kind='native_long_horizon', audit_only=False,
                  exclusive_target=False, object_id=None, target_id=None,
                  evidence='X-VLA native benchmark evidence exists; Hermes composite integration not yet physically verified')
del _skill


def route_capabilities(scene_id: str, capability_ids: list[str], available_native_ids: list[str] | None = None) -> dict:
    available = set(NATIVE_LONG_SKILLS if available_native_ids is None else available_native_ids)
    requested = copy.deepcopy(capability_ids)
    executed, replacements = [], []
    index = 0
    while index < len(requested):
        for identifier, skill in NATIVE_LONG_SKILLS.items():
            components = skill['component_capability_ids']
            if (identifier in available and scene_id in skill['scene_ids']
                    and requested[index:index + len(components)] == components):
                executed.append(identifier)
                replacements.append({'native_capability_id': identifier,
                    'component_capability_ids': copy.deepcopy(components), 'start_index': index})
                index += len(components)
                break
        else:
            executed.append(requested[index])
            index += 1
    return copy.deepcopy({'requested_capability_ids': requested,
        'executed_capability_ids': executed, 'replacements': replacements})
