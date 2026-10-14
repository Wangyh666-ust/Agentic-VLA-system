"""Registered native X-VLA skills and conservative interface routing."""
from __future__ import annotations
import copy

NATIVE_SKILL_GUIDANCE = """[native long-horizon skills]
Prefer one call to a listed native long-horizon skill only when all actions, objects, destinations and their order in the explicitly selected executable scope exactly match it. This scope may be a useful supported part of a larger request; declare its unsupported remainder separately, never claim full coverage. Otherwise split into the listed public executable capabilities. Never add closing a door, turning on a stove, or extra objects merely to use a skill; a preparation step must actually advance the user's goal. Intermediate sub-actions do not end the skill and do not start task-boundary resets. Candidate skills and Hermes integration have not yet been physically tested; do not exaggerate reliability.
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
# The recipes describe supported decomposition, not guaranteed policy timing.
_NEW_TASKS = [
 (1,'basket_cheese_butter','xvla_basket_cheese_butter','put both the cream cheese box and the butter in the basket',
  [('xvla_t01_cheese_to_basket','put the cream cheese box in the basket',['in','cream_cheese_1','basket_1_contain_region']),('xvla_t01_butter_to_basket','put the butter in the basket',['in','butter_1','basket_1_contain_region'])],None),
 (2,'stove_moka','xvla_stove_moka','turn on the stove and put the moka pot on it',
  [('xvla_t02_stove_on','turn on the stove',['turnon','flat_stove_1']),('xvla_t02_moka_to_stove','put the moka pot on the stove',['on','moka_pot_1','flat_stove_1_cook_region'])],None),
 (3,'bowl_drawer','xvla_bowl_drawer_close','put the black bowl in the bottom drawer of the cabinet and close it',
  [('xvla_t03_bowl_to_drawer','put the black bowl in the bottom drawer of the cabinet',['in','akita_black_bowl_1','white_cabinet_1_bottom_region']),('xvla_t03_drawer_close','close the bottom drawer of the cabinet',['close','white_cabinet_1_bottom_region'])],
  [['close','white_cabinet_1_bottom_region'],['in','akita_black_bowl_1','white_cabinet_1_bottom_region']]),
 (5,'book_caddy','xvla_book_caddy','pick up the book and place it in the back compartment of the caddy',
  [('xvla_t05_book_to_caddy','put the book in the back compartment of the caddy',['in','black_book_1','desk_caddy_1_back_contain_region'])],None),
 (6,'mug_pudding','xvla_mug_pudding','put the white mug on the plate and put the chocolate pudding to the right of the plate',
  [('xvla_t06_mug_to_plate','put the white mug on the plate',['on','porcelain_mug_1','plate_1']),('xvla_t06_pudding_right','put the chocolate pudding to the right of the plate',['on','chocolate_pudding_1','living_room_table_plate_right_region'])],None),
 (7,'basket_soup_cheese','xvla_basket_soup_cheese','put both the alphabet soup and the cream cheese box in the basket',
  [('xvla_t07_soup_to_basket','put the alphabet soup in the basket',['in','alphabet_soup_1','basket_1_contain_region']),('xvla_t07_cheese_to_basket','put the cream cheese box in the basket',['in','cream_cheese_1','basket_1_contain_region'])],None),
 (8,'moka_two','xvla_moka_both','put both moka pots on the stove',
  [('xvla_t08_moka1_to_stove','put one moka pot on the stove',['on','moka_pot_1','flat_stove_1_cook_region']),('xvla_t08_moka2_to_stove','put the other moka pot on the stove',['on','moka_pot_2','flat_stove_1_cook_region'])],
  [['on','moka_pot_1','flat_stove_1_cook_region'],['on','moka_pot_2','flat_stove_1_cook_region'],['turnon','flat_stove_1']]),
]
NATIVE_SCENES = {}
NATIVE_ATOMIC_CAPABILITIES = {}
for _task, _scene, _native, _instruction, _components, _goals in _NEW_TASKS:
    NATIVE_SCENES[_scene] = dict(label=_instruction, suite='libero_10', task_id=_task,
        variant='original', patches=[], backends=['xvla'], description=_instruction,
        storage_policy={g[1]:g[2] for _,_,g in _components if len(g)==3})
    for _cid, _language, _goal in _components:
        NATIVE_ATOMIC_CAPABILITIES[_cid] = dict(instruction=_language, goals=[copy.deepcopy(_goal)],
            scene_ids=[_scene], backends=['xvla'], skill_kind='atomic_candidate',
            audit_only=False, exclusive_target=len(_goal)==3, object_id=_goal[1],
            target_id=_goal[2] if len(_goal)==3 else None,
            evidence='Candidate atomic prompt; untested integration, not evidence of native single-task training')
    NATIVE_LONG_SKILLS[_native] = dict(instruction=_instruction,
        goals=copy.deepcopy(_goals if _goals is not None else [g for _,_,g in _components]),
        scene_ids=[_scene], native_task={'suite':'libero_10','task_id':_task},
        component_capability_ids=[cid for cid,_,_ in _components])
NATIVE_ATOMIC_CAPABILITIES['xvla_t06_pudding_right'].update(
    visual_relation='right_of', destination_definition='盘子右侧的桌面区域，不是盘面或盘子内部')
NATIVE_LONG_SKILLS['xvla_moka_both'].update(
    preserved_checks=[{'instruction':'keep the stove on','goals':[['turnon','flat_stove_1']],
                       'object_id':'flat_stove_1','target_id':None}],
    required_initial_goals=[['turnon','flat_stove_1']],
    precondition_description='原生双壶任务要求灶台已经开启并保持开启；此技能不提供自动点火。')
NATIVE_SKILL_GUIDANCE += """Preserved conditions are not extra actions. Native AND goals constrain the final state and do not prove internal action order. If a requested internal order differs or is unsupported, retain atomic calls. New atomic prompts are candidates, not verified native single-task training. Match every object, destination and operation.
"""

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
