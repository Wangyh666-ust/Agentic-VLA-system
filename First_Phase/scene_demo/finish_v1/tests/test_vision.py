import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.environ.get("FYP_PATH", "D:/FYP/First_Phase"))
from scene_demo.finish_v1 import vision  # noqa: E402


BASE = {
    "target_visible": True, "destination_visible": True, "at_destination": True,
    "supported": True, "released": True, "stable": True,
    "operation_achieved": True, "clear_of_target": True,
    "boxes": {}, "evidence": [], "unknown_reasons": [],
}


def ctx(op="place", **kw):
    sub = {"operation": op, "target": "cup", "phase": "do"}
    if op == "place":
        sub["destination"] = "tray"
    else:
        sub["destination"] = None
    sub.update(kw.pop("subtask", {}))
    base = {
        "user_request": "Put the cup on the tray.",
        "current_subtask": sub,
        "criteria": {"required": ["cup on tray"]},
    }
    base.update(kw)
    return base


def verdict(**kw):
    v = dict(BASE)
    v.update(kw)
    return v


class MockResp:
    def __init__(self, body, status=200):
        self._body = body.encode("utf-8") if isinstance(body, str) else body
        self._status = status
    def read(self):
        return self._body
    def getcode(self):
        return self._status
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class PromptTests(unittest.TestCase):
    def test_excludes_hidden_fields(self):
        c = ctx()
        c["plan"] = {"success": True}
        c["case_id"] = "g1"
        c["gold"] = 1
        c["sim_truth"] = {"x": 1}
        c["current_subtask"]["extra"] = "nope"
        p = vision.build_prompt(c)
        for bad in ("plan", "success", "case_id", "gold", "sim_truth", "extra"):
            self.assertNotIn('"' + bad + '":', p)
        self.assertIn("cup", p)

    def test_validates_context(self):
        with self.assertRaises(TypeError):
            vision.build_prompt(None)
        with self.assertRaises(ValueError):
            vision.build_prompt({"user_request": "", "current_subtask": {"operation": "place", "target": "a", "destination": "b", "phase": "c"}, "criteria": {"a": 1}})
        with self.assertRaises(ValueError):
            vision.build_prompt({"user_request": "x", "current_subtask": {"operation": "fly", "target": "a", "phase": "c"}, "criteria": {"a": 1}})

    def test_frames_strip_extra(self):
        c = ctx(sensor_frames=[{"frame": 1, "view": "front", "timestamp": 1.0, "secret": "x"}])
        p = vision.build_prompt(c)
        self.assertIn("front", p)
        self.assertNotIn("secret", p)


class VerdictTests(unittest.TestCase):
    def test_null_and_bool_rejections(self):
        v = verdict(target_visible=None)
        out = vision.validate_verdict(v)
        self.assertIsNone(out["target_visible"])
        for bad in (0, 1, "true"):
            with self.assertRaises((TypeError, ValueError)):
                vision.validate_verdict(verdict(stable=bad))
        with self.assertRaises(ValueError):
            vision.validate_verdict(verdict(confidence=0.9))

    def test_boxes(self):
        v = verdict(boxes={"cam": {"target": [0.1, 0.1, 0.2, 0.2], "destination": None, "gripper": None}})
        self.assertEqual(vision.validate_verdict(v)["boxes"]["cam"]["target"], [0.1, 0.1, 0.2, 0.2])
        with self.assertRaises(ValueError):
            vision.validate_verdict(verdict(boxes={"cam": {"target": [1.0, 0.0, 0.0, 1.0], "destination": None, "gripper": None}}))


class DecideTests(unittest.TestCase):
    def test_unknown_denies(self):
        d = vision.decide(ctx(), verdict(target_visible=None))
        self.assertEqual(d["state"], "unknown")
        self.assertFalse(d["allow_release"] or d["allow_retreat"])

    def test_supported_held_needs_release(self):
        d = vision.decide(ctx(), verdict(released=False, supported=True, stable=True))
        self.assertEqual(d["state"], "needs_release")
        self.assertTrue(d["allow_release"])
        d2 = vision.decide(ctx(), verdict(released=False, supported=True, stable=False))
        self.assertTrue(d2["allow_release"] is False)

    def test_unsupported_blocks_release(self):
        d = vision.decide(ctx(), verdict(released=False, supported=False))
        self.assertEqual(d["state"], "incomplete")
        self.assertFalse(d["allow_release"])

    def test_place_complete_ignores_operation_achieved(self):
        for val in (None, False):
            d = vision.decide(ctx(), verdict(operation_achieved=val))
            self.assertEqual(d["state"], "complete")

    def test_turn_needs_retreat(self):
        d = vision.decide(ctx(op="turn"), verdict(clear_of_target=False))
        self.assertEqual(d["state"], "needs_retreat")
        self.assertTrue(d["allow_retreat"])


class ParseTests(unittest.TestCase):
    def test_naked_and_fence(self):
        raw = json.dumps(BASE)
        self.assertEqual(vision.validate_verdict(vision._parse_verdict_content(raw))["stable"], True)
        fenced = "```json\n" + raw + "\n```"
        self.assertEqual(vision.validate_verdict(vision._parse_verdict_content(fenced))["stable"], True)

    def test_reject_prose(self):
        raw = json.dumps(BASE)
        for bad in ("hello " + raw, raw + " bye", "```json\n" + raw + "\n``` extra"):
            with self.assertRaises(ValueError):
                vision._parse_verdict_content(bad)


class QwenTests(unittest.TestCase):
    def _resp(self, body):
        return MockResp(json.dumps({"choices": [{"message": {"content": json.dumps(body)}}], "usage": {"t": 1}, "model": "m"}))

    def test_success(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "a.png")
            p.write_bytes(b"\x89PNG\r\n\x1a\n")
            with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "secret"}, clear=False), \
                 mock.patch("scene_demo.finish_v1.vision.urllib.request.urlopen", return_value=self._resp(BASE)) as u:
                c = vision.QwenClient(env_path=Path(d, "nope.env"), max_calls=2)
                out = c.analyze(ctx(), [str(p)])
                req = u.call_args[0][0]
                body = req.data.decode("utf-8")
                self.assertIn("data:image/png;base64,", body)
                self.assertNotIn("secret", body)
                self.assertEqual(out["verdict"]["stable"], True)
                self.assertEqual(out["usage"], {"t": 1})
                self.assertEqual(out["call_index"], 1)
                self.assertEqual(out["request_sha256"], hashlib.sha256(req.data).hexdigest())

    def test_budget_charge_on_http_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "a.png")
            p.write_bytes(b"\x89PNG\r\n\x1a\n")
            err = urllib.error.HTTPError("u", 500, "e", {}, io.BytesIO(b""))
            with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "secret"}, clear=False), \
                 mock.patch("scene_demo.finish_v1.vision.urllib.request.urlopen", side_effect=err) as u:
                c = vision.QwenClient(env_path=Path(d, "nope.env"), max_calls=1)
                with self.assertRaises(RuntimeError):
                    c.analyze(ctx(), [str(p)])
                with self.assertRaises(RuntimeError):
                    c.analyze(ctx(), [str(p)])
                self.assertEqual(u.call_count, 1)

    def test_bad_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "a.png")
            p.write_bytes(b"x")
            bad = {"choices": [{"message": {"content": json.dumps({"nope": 1})}}]}
            with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "secret"}, clear=False), \
                 mock.patch("scene_demo.finish_v1.vision.urllib.request.urlopen", return_value=MockResp(json.dumps(bad))):
                c = vision.QwenClient(env_path=Path(d, "nope.env"), max_calls=2)
                with self.assertRaises(RuntimeError):
                    c.analyze(ctx(), [str(p)])


class ImportSafetyTests(unittest.TestCase):
    def test_constructor_no_io(self):
        def boom(*a, **k):
            raise AssertionError("io")
        with mock.patch.object(Path, "read_bytes", boom), mock.patch.object(Path, "read_text", boom), mock.patch.dict(os.environ, {}, clear=True), mock.patch("scene_demo.finish_v1.vision.urllib.request.urlopen", boom):
            c = vision.QwenClient(env_path="/nope/.env", max_calls=0)
            with self.assertRaises(RuntimeError):
                c.analyze(ctx(), ["x.png"])


class ParserFinalTests(unittest.TestCase):
    def test_naked_and_fenced_success(self):
        self.assertEqual(vision._parse_verdict_content('{"a": 1}'), {'a': 1})
        self.assertEqual(vision._parse_verdict_content('```json\n{"a": 1}\n```'), {'a': 1})
        self.assertEqual(vision._parse_verdict_content('```\n{"a": 1}\n```'), {'a': 1})

    def test_leading_trailing_prose_rejected(self):
        with self.assertRaises(ValueError):
            vision._parse_verdict_content('Here: {"a": 1}')
        with self.assertRaises(ValueError):
            vision._parse_verdict_content('{"a": 1} thanks')

    def test_multiple_fences_rejected(self):
        with self.assertRaises(ValueError):
            vision._parse_verdict_content('```json\n{"a": 1}\n```\n```\n{"b": 2}\n```')

    def test_non_dict_rejected(self):
        with self.assertRaises(ValueError):
            vision._parse_verdict_content('[1, 2, 3]')
        with self.assertRaises(ValueError):
            vision._parse_verdict_content('"hello"')

if __name__ == "__main__":
    unittest.main()
