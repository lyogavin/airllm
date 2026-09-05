"""Load a tiny SFT file for streamed LoRA (JSONL / JSON / TXT)."""

from __future__ import annotations

import json
from pathlib import Path


def record_from_obj(obj) -> dict:
    """Turn one JSON object (or a raw string) into ``{text, mask_prefix}``."""
    if isinstance(obj, str):
        text = obj.strip()
        if not text:
            raise ValueError("empty string example")
        return {"text": text, "mask_prefix": None}
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object or string, got {type(obj).__name__}")

    if obj.get("text"):
        return {"text": str(obj["text"]), "mask_prefix": None}

    prompt = obj.get("prompt") or obj.get("instruction") or ""
    extra = obj.get("input") or ""
    completion = obj.get("completion") or obj.get("output") or obj.get("response") or ""
    if prompt or completion:
        prefix_parts = [p for p in (str(prompt).strip(), str(extra).strip()) if p]
        prefix = "\n".join(prefix_parts)
        completion = str(completion)
        if prefix and completion:
            sep = "" if prefix.endswith(("\n", " ")) else "\n"
            text = f"{prefix}{sep}{completion}"
        else:
            text = prefix or completion
        if not text.strip():
            raise ValueError("prompt/completion example is empty")
        return {"text": text, "mask_prefix": prefix or None}

    messages = obj.get("messages")
    if isinstance(messages, list) and messages:
        lines = []
        last_assistant = None
        for msg in messages:
            role = (msg.get("role") or "user").strip()
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
            if role == "assistant":
                last_assistant = content
        if not lines:
            raise ValueError("messages example is empty")
        text = "\n".join(lines)
        prefix = None
        if last_assistant and text.endswith(last_assistant):
            prefix = text[: -len(last_assistant)]
        return {"text": text, "mask_prefix": prefix}

    raise ValueError(
        "each example needs `text`, or `prompt`/`completion`, "
        "or `instruction`/`output`, or `messages`"
    )


def _load_json_blob(raw: str):
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "rows"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    raise ValueError("JSON file must be an object, a list, or {\"data\": [...]}")


def load_records(path) -> list:
    """Load training examples from ``.jsonl``, ``.json``, or ``.txt``.

    TXT is one example per blank-line-separated block (the whole file if there
    are no blank lines).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")
    records = []
    if suffix == ".txt":
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        for block in blocks:
            records.append({"text": block, "mask_prefix": None})
    elif suffix == ".json":
        for i, obj in enumerate(_load_json_blob(text)):
            try:
                records.append(record_from_obj(obj))
            except ValueError as e:
                raise ValueError(f"{p}:{i}: {e}") from e
    else:
        # .jsonl and anything else: one JSON object per non-empty line
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(record_from_obj(json.loads(line)))
            except ValueError as e:
                raise ValueError(f"{p}:{i}: {e}") from e
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{i}: invalid JSON ({e.msg})") from e
    if not records:
        raise ValueError(f"no examples in {p}")
    return records


def encode_record(tokenizer, record, seq_len, device):
    """Tokenize one example. Prompt tokens are labeled ``-100`` when masked."""
    encoded = tokenizer(
        record["text"],
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    labels = input_ids.clone()
    prefix = record.get("mask_prefix")
    if prefix:
        pref = tokenizer(
            prefix,
            add_special_tokens=True,
            truncation=True,
            max_length=seq_len,
        )
        plen = len(pref["input_ids"])
        plen = min(plen, max(input_ids.shape[1] - 1, 0))
        if plen:
            labels[:, :plen] = -100
    return input_ids, attention_mask, labels


def iter_train_items(records, epochs, steps=None):
    """Yield ``(step, epoch, record)`` over ``epochs`` passes, optional step cap."""
    step = 0
    for epoch in range(epochs):
        for rec in records:
            if steps is not None and step >= steps:
                return
            yield step, epoch, rec
            step += 1
