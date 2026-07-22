import unittest
from types import SimpleNamespace

import torch

from ..airllm.server import _generate_chat
from ..airllm.server import create_app


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "prompt"

    def __call__(self, prompt, return_tensors=None, return_attention_mask=None):
        return {
            "input_ids": torch.tensor([[10, 11]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, generated, skip_special_tokens=False):
        return "answer"


class FakeModel:
    tokenizer = FakeTokenizer()
    device = torch.device("cpu")

    def generate(self, **kwargs):
        return SimpleNamespace(sequences=torch.tensor([[10, 11, 12, 13, 14]]))


class TestServer(unittest.TestCase):
    def test_generates_only_completion_tokens(self):
        text, prompt_tokens, completion_tokens = _generate_chat(
            FakeModel(),
            [{"role": "user", "content": "hello"}],
            max_tokens=3,
            temperature=0,
            top_p=1,
        )
        self.assertEqual(text, "answer")
        self.assertEqual(prompt_tokens, 2)
        self.assertEqual(completion_tokens, 3)

    def test_rejects_non_text_messages(self):
        with self.assertRaises(ValueError):
            _generate_chat(
                FakeModel(),
                [{"role": "user", "content": [{"type": "image"}]}],
                max_tokens=3,
                temperature=0,
                top_p=1,
            )

    def test_openai_compatible_chat_route(self):
        from fastapi.testclient import TestClient

        client = TestClient(create_app(FakeModel(), "fake-model"))
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 3,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["content"], "answer")
        self.assertEqual(payload["usage"]["total_tokens"], 5)
