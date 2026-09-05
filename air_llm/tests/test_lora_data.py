"""Dataset loading for streamed LoRA — no GPU or torch required."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

_AIRLLM_DIR = Path(__file__).resolve().parents[1] / "airllm"

if "airllm" not in sys.modules:
    _pkg = types.ModuleType("airllm")
    _pkg.__path__ = [str(_AIRLLM_DIR)]
    sys.modules["airllm"] = _pkg

from airllm.lora_data import iter_train_items, load_records, record_from_obj


class TestRecordFromObj(unittest.TestCase):
    def test_text(self):
        rec = record_from_obj({"text": "hello world"})
        self.assertEqual(rec["text"], "hello world")
        self.assertIsNone(rec["mask_prefix"])

    def test_prompt_completion(self):
        rec = record_from_obj({"prompt": "Q?", "completion": "A."})
        self.assertEqual(rec["text"], "Q?\nA.")
        self.assertEqual(rec["mask_prefix"], "Q?")

    def test_alpaca(self):
        rec = record_from_obj({
            "instruction": "Translate",
            "input": "hola",
            "output": "hello",
        })
        self.assertEqual(rec["text"], "Translate\nhola\nhello")
        self.assertEqual(rec["mask_prefix"], "Translate\nhola")

    def test_messages(self):
        rec = record_from_obj({
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        })
        self.assertIn("user: Hi", rec["text"])
        self.assertTrue(rec["text"].endswith("Hello"))
        self.assertTrue(rec["mask_prefix"].endswith("assistant: "))


class TestLoadRecords(unittest.TestCase):
    def test_jsonl_and_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "a.jsonl"
            jsonl.write_text(
                json.dumps({"text": "one"}) + "\n"
                + json.dumps({"prompt": "Q", "completion": "A"}) + "\n",
                encoding="utf-8",
            )
            recs = load_records(jsonl)
            self.assertEqual(len(recs), 2)
            self.assertEqual(recs[0]["text"], "one")

            txt = Path(tmp) / "b.txt"
            txt.write_text("first block\n\nsecond block\n", encoding="utf-8")
            recs = load_records(txt)
            self.assertEqual([r["text"] for r in recs], ["first block", "second block"])

    def test_json_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text(json.dumps([{"text": "x"}, "y"]), encoding="utf-8")
            recs = load_records(path)
            self.assertEqual([r["text"] for r in recs], ["x", "y"])


class TestIterTrainItems(unittest.TestCase):
    def test_epochs_and_step_cap(self):
        recs = [{"text": "a"}, {"text": "b"}]
        items = list(iter_train_items(recs, epochs=2))
        self.assertEqual(len(items), 4)
        capped = list(iter_train_items(recs, epochs=9, steps=3))
        self.assertEqual([s for s, _, _ in capped], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
