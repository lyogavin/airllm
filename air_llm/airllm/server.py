"""Minimal non-streaming OpenAI-compatible chat server for AirLLM."""

import argparse
import threading
import time
import uuid

import torch

from .auto_model import AutoModel


def _generate_chat(model, messages, max_tokens, temperature, top_p):
    if not messages or any(not isinstance(message.get("content"), str) for message in messages):
        raise ValueError("AirLLM's chat server accepts non-empty text-only messages.")

    tokenizer = model.tokenizer
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    device = getattr(model, "device", torch.device("cuda:0"))
    inputs = {name: value.to(device) for name, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "return_dict_in_generate": True,
    }
    if temperature > 0:
        generation_kwargs.update(temperature=temperature, top_p=top_p)

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)
    sequence = output.sequences[0]
    generated = sequence[prompt_tokens:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, prompt_tokens, int(generated.shape[-1])


def create_app(model, served_model_name):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError('Install the server dependencies with `pip install "airllm[server]"`.') from exc

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str | None = None
        messages: list[ChatMessage]
        max_tokens: int = Field(default=128, ge=1)
        temperature: float = Field(default=0.0, ge=0.0)
        top_p: float = Field(default=1.0, gt=0.0, le=1.0)
        stream: bool = False

    app = FastAPI(title="AirLLM OpenAI-compatible server")
    inference_lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": served_model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming responses are not implemented.")
        if request.model not in (None, served_model_name):
            raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")
        try:
            with inference_lock:
                text, prompt_tokens, completion_tokens = _generate_chat(
                    model,
                    [message.model_dump() for message in request.messages],
                    request.max_tokens,
                    request.temperature,
                    request.top_p,
                )
        except (ValueError, NotImplementedError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        created = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offload-dir")
    parser.add_argument("--compression", choices=("4bit", "8bit"))
    parser.add_argument("--delete-original", action="store_true")
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Load one layer at a time; recommended when two streamed layers would approach system RAM.",
    )
    parser.add_argument("--hf-token")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError('Install the server dependencies with `pip install "airllm[server]"`.') from exc

    model = AutoModel.from_pretrained(
        args.model,
        device=args.device,
        cache_dir=args.cache_dir,
        layer_shards_saving_path=args.offload_dir,
        compression=args.compression,
        prefetching=not args.no_prefetch,
        delete_original=args.delete_original,
        hf_token=args.hf_token,
    )
    uvicorn.run(create_app(model, args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
