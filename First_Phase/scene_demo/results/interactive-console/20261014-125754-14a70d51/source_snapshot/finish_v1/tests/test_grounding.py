import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scene_demo.finish_v1 import vision

BASE = {name: True for name in vision._BOOL_FIELDS}
BASE.update(boxes={}, evidence=[], unknown_reasons=[])

def ctx(**kw):
    result = {
        "user_request": "Place the bowl on the plate.",
        "current_subtask": {"operation": "place", "target": "akita_black_bowl_1", "destination": "plate_1", "phase": "confirm"},
        "criteria": {"required": ["bowl on plate"]},
    }
    result.update(kw)
    return result

class MockResp:
    def read(self):
        return json.dumps({"choices": [{"message": {"content": json.dumps(BASE)}}], "usage": {"t": 1}, "model": "m"}).encode("utf-8")
    def getcode(self):
        return 200

class GroundingTests(unittest.TestCase):
    def test_prompt_grounding_and_no_private_truth(self):
        prompt = vision.build_prompt(ctx(gold="SECRET_GOLD", native="SECRET_NATIVE", sim_truth="SECRET_SIM", plan={"success": "SECRET_PLAN"}))
        for text in ("bowl", "plate", "round burner ring", "current instant", "outer rim"):
            self.assertIn(text, prompt)
        for secret in ("SECRET_GOLD", "SECRET_NATIVE", "SECRET_SIM", "SECRET_PLAN"):
            self.assertNotIn(secret, prompt)

    def test_real_client_labels_interleave_matching_urls(self):
        with tempfile.TemporaryDirectory() as directory:
            data = [b"\x89PNG\r\n\x1a\nfirst", b"\x89PNG\r\n\x1a\nsecond"]
            paths = [Path(directory, f"{index}.png") for index in range(2)]
            for path, raw in zip(paths, data):
                path.write_bytes(raw)
            frames = [{"frame": 0, "view": "agentview", "timestamp": 1.0}, {"frame": 1, "view": "wrist", "timestamp": 2.0}]
            with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}), mock.patch("scene_demo.finish_v1.vision.urllib.request.urlopen", return_value=MockResp()) as opener:
                client = vision.QwenClient(env_path=Path(directory, "absent.env"))
                metadata = client.analyze(ctx(sensor_frames=frames), [str(path) for path in paths])
            request = opener.call_args[0][0]
            content = json.loads(request.data)["messages"][0]["content"]
            self.assertEqual(len(content), 5)
            self.assertEqual([item["type"] for item in content], ["text", "text", "image_url", "text", "image_url"])
            self.assertEqual(content[1]["text"], "Image frame=0; view=agentview; timestamp=1.0")
            self.assertEqual(content[3]["text"], "Image frame=1; view=wrist; timestamp=2.0")
            self.assertEqual(content[2]["image_url"]["url"], "data:image/png;base64," + base64.b64encode(data[0]).decode("ascii"))
            self.assertEqual(content[4]["image_url"]["url"], "data:image/png;base64," + base64.b64encode(data[1]).decode("ascii"))
            self.assertEqual(metadata["request_sha256"], hashlib.sha256(request.data).hexdigest())

    def test_mismatched_frames_rejected_before_http(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory, f"{index}.png") for index in range(2)]
            for path in paths:
                path.write_bytes(b"\x89PNG\r\n\x1a\n")
            frame = {"frame": 0, "view": "agentview", "timestamp": 1.0}
            cases = [[], [frame], [{"frame": 1, "view": "agentview", "timestamp": 1.0}, {"frame": 0, "view": "wrist", "timestamp": 1.0}]]
            with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}), mock.patch("scene_demo.finish_v1.vision.urllib.request.urlopen") as opener:
                for frames in cases:
                    with self.subTest(frames=frames):
                        with self.assertRaises(ValueError):
                            vision.QwenClient(env_path=Path(directory, "absent.env")).analyze(ctx(sensor_frames=frames), [str(path) for path in paths])
                self.assertEqual(opener.call_count, 0)

    def test_no_frame_metadata_compatible_and_schema_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "image.png")
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
            with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}), mock.patch("scene_demo.finish_v1.vision.urllib.request.urlopen", return_value=MockResp()) as opener:
                metadata = vision.QwenClient(env_path=Path(directory, "absent.env")).analyze(ctx(), [str(path)])
            content = json.loads(opener.call_args[0][0].data)["messages"][0]["content"]
            self.assertEqual(len(content), 3)
            self.assertEqual(content[1]["text"], "Image frame=0; view=unspecified; timestamp=unspecified")
            self.assertEqual(metadata["verdict"], BASE)
            self.assertEqual(set(metadata["verdict"]), set(vision._BOOL_FIELDS) | {"boxes", "evidence", "unknown_reasons"})

    def test_unknown_release_not_complete(self):
        verdict = dict(BASE, released=None)
        self.assertNotEqual(vision.decide(ctx(), verdict)["state"], "complete")

    def test_short_window_stability_and_local_clearance_prompt(self):
        prompt = vision.build_prompt(ctx(sensor_frames=[{"frame": 0, "view": "agentview", "timestamp": 1.0}]))
        for text in ("short-window", "minimum duration", "local separation"):
            self.assertIn(text, prompt)
        self.assertEqual(vision.decide(ctx(), dict(BASE, stable=None))["state"], "unknown")

    def test_standard_open_encoder_context_optional_and_validated(self):
        measurements = {"stage": "confirm", "finger_qpos": [0.003, -0.006], "max_abs_arm_qvel": 0.0}
        calibrated = dict(measurements, standard_open_finger_qpos=[0.04, -0.04])
        clean = vision._validate_actuator_measurements(calibrated)
        self.assertEqual(clean["standard_open_finger_qpos"], [0.04, -0.04])
        self.assertTrue(all(type(x) is float for x in clean["standard_open_finger_qpos"]))
        prompt = vision.build_prompt(ctx(actuator_measurements=calibrated))
        self.assertIn("calibrated standard-open pair", prompt)
        self.assertIn("encoder veto", prompt)
        self.assertIn("0.002", prompt)
        for bad in ([0.04], [float("nan"), -0.04], [True, -0.04]):
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, TypeError)):
                    vision._validate_actuator_measurements(dict(measurements, standard_open_finger_qpos=bad))
        self.assertEqual(set(vision._validate_actuator_measurements(measurements)), set(measurements))
        self.assertIsNone(vision._validate_actuator_measurements(None))

if __name__ == "__main__":
    unittest.main()
