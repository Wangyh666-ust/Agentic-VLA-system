"""All active visual roles use the user's selected model, never a silent fallback."""
import pytest
from scene_demo import setup_profile
from scene_demo.finish_v1.vision import QwenClient
from scene_demo.finish_v1.simple_checker import SimpleQwenClient
from scene_demo.xvla_grounding import InventoryClient
from scene_demo.xvla_finish import XvlaQwenClient
from scene_demo.xvla_event_checker import XvlaEventQwenClient
from scene_demo.xvla_subtask_review import XvlaSubtaskReviewClient


@pytest.mark.parametrize('client', [QwenClient, SimpleQwenClient, InventoryClient,
                                  XvlaQwenClient, XvlaEventQwenClient, XvlaSubtaskReviewClient])
def test_every_role_uses_qwen38(client):
    assert client().model == 'qwen3.8-max'
    with pytest.raises(ValueError, match='qwen3.8-max'):
        client(model='qwen3-vl-plus')


def test_hermes_profile_matches_visual_roles():
    assert setup_profile.PLANNER_MODEL == QwenClient().model
    assert 'default: qwen3.8-max' in setup_profile.CONFIG_YAML


def test_stale_inventory_environment_cannot_reactivate_old_model(monkeypatch):
    monkeypatch.setenv('SCENE_PLANNER_MODEL', 'qwen3-vl-plus')
    with pytest.raises(ValueError, match='qwen3.8-max'):
        InventoryClient()
