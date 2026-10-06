"""Fixed persistent-scene capability catalog (standard library only).

The catalog describes the seven fixed scenes and the SmolVLA-backed
capabilities that may run inside them. It deliberately carries no hidden case
expectations and no success rates: every ``evidence`` string is a known
evidence tag captured from an earlier run or the plain ``candidate`` marker.
"""

from __future__ import annotations

import copy

SCENES: dict[str, dict] = {
    "goal_table": {
        "label": "Original goal table",
        "suite": "libero_goal",
        "task_id": 8,
        "variant": "original",
        "patches": [],
        "storage_policy": {
            "akita_black_bowl_1": "plate_1",
            "wine_bottle_1": "wine_rack_1_top_region",
        },
        "description": (
            "Shared original bowl, wine and stove scene; no trash bin exists "
            "here."
        ),
    },
    "goal_table_shifted": {
        "label": "Shifted goal table",
        "suite": "libero_goal",
        "task_id": 8,
        "variant": "layout_shift",
        "patches": [
            {"kind": "offset", "object_id": "akita_black_bowl_1", "xyz": [0.06, 0, 0]},
            {"kind": "offset", "object_id": "wine_bottle_1", "xyz": [-0.06, 0, 0]},
        ],
        "storage_policy": {
            "akita_black_bowl_1": "plate_1",
            "wine_bottle_1": "wine_rack_1_top_region",
        },
        "description": (
            "Experimental copy of the goal table whose bowl and wine bottle "
            "layout is deliberately changed along x."
        ),
    },
    "basket_two": {
        "label": "Two cans in basket",
        "suite": "libero_10",
        "task_id": 0,
        "variant": "original",
        "patches": [],
        "storage_policy": {
            "alphabet_soup_1": "basket_1_contain_region",
            "tomato_sauce_1": "basket_1_contain_region",
        },
        "description": (
            "Shared original basket scene holding the alphabet soup can and "
            "the tomato sauce can."
        ),
    },
    "mugs_two": {
        "label": "Two mugs, two plates",
        "suite": "libero_10",
        "task_id": 4,
        "variant": "original",
        "patches": [],
        "storage_policy": {
            "porcelain_mug_1": "plate_1",
            "white_yellow_mug_1": "plate_2",
        },
        "description": (
            "Shared original two-mug scene where both plates start empty."
        ),
    },
    "mugs_left_occupied": {
        "label": "Left plate occupied",
        "suite": "libero_10",
        "task_id": 4,
        "variant": "occupied",
        "patches": [
            {"kind": "place_on", "object_id": "red_coffee_mug_1", "target_id": "plate_1"},
        ],
        "storage_policy": {},
        "description": (
            "Two-mug scene pre-populated with the red coffee mug already on "
            "the left plate."
        ),
    },
    "mugs_right_occupied": {
        "label": "Right plate occupied",
        "suite": "libero_10",
        "task_id": 4,
        "variant": "occupied",
        "patches": [
            {"kind": "place_on", "object_id": "white_yellow_mug_1", "target_id": "plate_2"},
        ],
        "storage_policy": {},
        "description": (
            "Two-mug scene pre-populated with the yellow and white mug already "
            "on the right plate."
        ),
    },
    "mugs_both_occupied": {
        "label": "Both plates occupied",
        "suite": "libero_10",
        "task_id": 4,
        "variant": "occupied",
        "patches": [
            {"kind": "place_on", "object_id": "red_coffee_mug_1", "target_id": "plate_1"},
            {"kind": "place_on", "object_id": "white_yellow_mug_1", "target_id": "plate_2"},
        ],
        "storage_policy": {},
        "description": (
            "Two-mug scene pre-populated with both plates already occupied."
        ),
    },
}

CAPABILITIES: dict[str, dict] = {
    "bowl_to_plate": {
        "instruction": "put the bowl on the plate",
        "goals": [["on", "akita_black_bowl_1", "plate_1"]],
        "object_id": "akita_black_bowl_1",
        "target_id": "plate_1",
        "scene_ids": ["goal_table", "goal_table_shifted"],
        "evidence": "v0.1.0 successful original episode",
        "audit_only": False,
        "exclusive_target": True,
    },
    "wine_to_rack": {
        "instruction": "put the wine bottle on the rack",
        "goals": [["on", "wine_bottle_1", "wine_rack_1_top_region"]],
        "object_id": "wine_bottle_1",
        "target_id": "wine_rack_1_top_region",
        "scene_ids": ["goal_table", "goal_table_shifted"],
        "evidence": "v0.1.0 successful original episode",
        "audit_only": False,
        "exclusive_target": True,
    },
    "stove_on": {
        "instruction": "turn on the stove",
        "goals": [["turnon", "flat_stove_1"]],
        "object_id": "flat_stove_1",
        "target_id": None,
        "scene_ids": ["goal_table", "goal_table_shifted"],
        "evidence": "v0.1.0 successful original episode",
        "audit_only": False,
        "exclusive_target": False,
    },
    "soup_to_basket": {
        "instruction": "put the alphabet soup in the basket",
        "goals": [["in", "alphabet_soup_1", "basket_1_contain_region"]],
        "object_id": "alphabet_soup_1",
        "target_id": "basket_1_contain_region",
        "scene_ids": ["basket_two"],
        "evidence": (
            "candidate in this scene; previous successful single task is a "
            "different scene"
        ),
        "audit_only": False,
        "exclusive_target": False,
    },
    "sauce_to_basket": {
        "instruction": "put the tomato sauce in the basket",
        "goals": [["in", "tomato_sauce_1", "basket_1_contain_region"]],
        "object_id": "tomato_sauce_1",
        "target_id": "basket_1_contain_region",
        "scene_ids": ["basket_two"],
        "evidence": "candidate",
        "audit_only": False,
        "exclusive_target": False,
    },
    "white_mug_left": {
        "instruction": "put the white mug on the left plate",
        "goals": [["on", "porcelain_mug_1", "plate_1"]],
        "object_id": "porcelain_mug_1",
        "target_id": "plate_1",
        "scene_ids": [
            "mugs_two",
            "mugs_left_occupied",
            "mugs_right_occupied",
            "mugs_both_occupied",
        ],
        "evidence": "candidate",
        "audit_only": False,
        "exclusive_target": True,
    },
    "white_mug_right": {
        "instruction": "put the white mug on the right plate",
        "goals": [["on", "porcelain_mug_1", "plate_2"]],
        "object_id": "porcelain_mug_1",
        "target_id": "plate_2",
        "scene_ids": [
            "mugs_two",
            "mugs_left_occupied",
            "mugs_right_occupied",
            "mugs_both_occupied",
        ],
        "evidence": "unverified alternate destination; candidate",
        "audit_only": False,
        "exclusive_target": True,
    },
    "yellow_mug_right": {
        "instruction": "put the yellow and white mug on the right plate",
        "goals": [["on", "white_yellow_mug_1", "plate_2"]],
        "object_id": "white_yellow_mug_1",
        "target_id": "plate_2",
        "scene_ids": [
            "mugs_two",
            "mugs_left_occupied",
            "mugs_right_occupied",
            "mugs_both_occupied",
        ],
        "evidence": "candidate",
        "audit_only": False,
        "exclusive_target": True,
    },
    "table_both": {
        "instruction": (
            "put the bowl on the plate and put the wine bottle on the rack"
        ),
        "goals": [
            ["on", "akita_black_bowl_1", "plate_1"],
            ["on", "wine_bottle_1", "wine_rack_1_top_region"],
        ],
        "object_id": None,
        "target_id": None,
        "scene_ids": ["goal_table", "goal_table_shifted"],
        "evidence": "candidate composite",
        "audit_only": True,
        "exclusive_target": False,
    },
    "basket_both": {
        "instruction": "put both the alphabet soup and the tomato sauce in the basket",
        "goals": [
            ["in", "alphabet_soup_1", "basket_1_contain_region"],
            ["in", "tomato_sauce_1", "basket_1_contain_region"],
        ],
        "object_id": None,
        "target_id": None,
        "scene_ids": ["basket_two"],
        "evidence": "candidate composite",
        "audit_only": True,
        "exclusive_target": False,
    },
    "mugs_both": {
        "instruction": (
            "put the white mug on the left plate and put the yellow and white "
            "mug on the right plate"
        ),
        "goals": [
            ["on", "porcelain_mug_1", "plate_1"],
            ["on", "white_yellow_mug_1", "plate_2"],
        ],
        "object_id": None,
        "target_id": None,
        "scene_ids": ["mugs_two"],
        "evidence": "candidate composite",
        "audit_only": True,
        "exclusive_target": False,
    },
}


def scene_capabilities(scene_id: str, include_audit: bool = False) -> list[dict]:
    """Return deep copies of the capabilities usable in ``scene_id``.

    Audit-only composite capabilities are excluded unless ``include_audit`` is
    true. Unknown scenes raise ``ValueError``. Every returned dict gains a
    ``capability_id`` field and is a deep copy, so caller mutations can never
    reach the catalog itself.
    """

    if scene_id not in SCENES:
        raise ValueError("unknown scene_id: %r" % (scene_id,))

    result: list[dict] = []
    for capability_id, capability in CAPABILITIES.items():
        if scene_id not in capability["scene_ids"]:
            continue
        if capability["audit_only"] and not include_audit:
            continue
        entry = copy.deepcopy(capability)
        entry["capability_id"] = capability_id
        result.append(entry)
    return result


def deduplicate_goals(capability_ids: list[str]) -> list[list[str]]:
    """Union the goals of ``capability_ids`` with first occurrence winning.

    Repeated goal predicates collapse to their first occurrence. Unknown
    capability ids raise ``ValueError``.
    """

    seen: set[str] = set()
    merged: list[list[str]] = []
    for capability_id in capability_ids:
        if capability_id not in CAPABILITIES:
            raise ValueError("unknown capability_id: %r" % (capability_id,))
        for goal in CAPABILITIES[capability_id]["goals"]:
            key = goal_key(goal)
            if key in seen:
                continue
            seen.add(key)
            merged.append(list(goal))
    return merged


def goal_key(goal: list[str]) -> str:
    """Canonical string key for a single goal predicate."""

    return "|".join(goal)
