import os
import threading
import traceback
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from webui.config import ENV_FILE, MAX_INPUT_TOKENS, PORT, current_favicon_path
from webui.downloads import (
    active_download,
    download_job_snapshot,
    parse_patterns,
    check_hf_repo_compatibility,
    normalize_hf_repo_id,
    resolve_hf_download_dir,
    start_hf_download_job,
)
from webui.model_catalog import discover_models, list_candidate_directories
from webui.model_runtime import (
    apply_settings,
    active_load_job,
    get_runtime_field,
    load_model,
    load_job_snapshot,
    looks_like_local_path,
    normalize_optional,
    persist_runtime_to_env,
    persist_runtime_to_env_if_changed,
    resolve_requested_source,
    request_stop_load_job,
    start_load_job,
    runtime_state,
)
from webui.schemas import (
    GenerateRequest,
    GenerateResponse,
    HFDownloadRequest,
    LoadRequest,
    SettingsRequest,
)
from webui.ui import render_ui_html

app = FastAPI(title="AirLLM API", version="1.0.0")
_autoload_lock = threading.Lock()
_autoload_attempted_source = None
_autoload_last_error = None
_autoload_job_id = None

def _coerce_local_path_input(model_id: Optional[str], model_path: Optional[str]):
    normalized_model_id = normalize_optional(model_id)
    normalized_model_path = normalize_optional(model_path)
    if normalized_model_path:
        return normalized_model_id, normalized_model_path
    if normalized_model_id and looks_like_local_path(normalized_model_id):
        return None, normalized_model_id
    return normalized_model_id, normalized_model_path


def _maybe_autoload_configured_local_model():
    global _autoload_attempted_source, _autoload_last_error, _autoload_job_id

    state = runtime_state()
    if state.get("model_loaded"):
        return {"status": "already_loaded"}

    resolved = resolve_requested_source()
    model_source = resolved.get("model_source")
    local_expected = bool(resolved.get("model_path") or resolved.get("model_base_dir"))
    if not local_expected:
        return {"status": "skipped_remote"}
    if not model_source or not os.path.isdir(model_source):
        return {"status": "skipped_missing_local", "model_source": model_source}

    # Keep this non-blocking: only enqueue/load-job state checks here.
    with _autoload_lock:
        if _autoload_job_id:
            snapshot = load_job_snapshot(_autoload_job_id)
            if snapshot:
                status = snapshot.get("status")
                if status in ("queued", "loading"):
                    return {
                        "status": "already_loading",
                        "job_id": _autoload_job_id,
                        "model_source": model_source,
                    }
                if status == "failed":
                    _autoload_last_error = snapshot.get("error") or snapshot.get("message")
                elif status == "completed":
                    _autoload_last_error = None

        if _autoload_attempted_source == model_source and _autoload_last_error is not None:
            return {
                "status": "skipped_previous_failure",
                "model_source": model_source,
                "error": _autoload_last_error,
            }

        _autoload_attempted_source = model_source
        _autoload_last_error = None

    try:
        job_id = start_load_job(
            {
                "model_id": resolved.get("model_id"),
                "model_path": resolved.get("model_path"),
                "model_base_dir": resolved.get("model_base_dir"),
                "device": state.get("device"),
                "force_reload": False,
                "persist_env": False,
            }
        )
        with _autoload_lock:
            _autoload_job_id = job_id
        return {"status": "started", "job_id": job_id, "model_source": model_source}
    except Exception as exc:
        err = str(exc)
        with _autoload_lock:
            _autoload_last_error = err
        print(f"[autoloader] Failed to enqueue configured model load: {err}")
        traceback.print_exc()
        return {"status": "failed", "model_source": model_source, "error": err}

@app.on_event("startup")
def startup_autoload():
    threading.Thread(target=_maybe_autoload_configured_local_model, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def ui():
    html = render_ui_html(runtime_state())
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    favicon_path = current_favicon_path()
    if favicon_path and favicon_path.exists():
        return FileResponse(favicon_path, headers={"Cache-Control": "no-cache"})
    raise HTTPException(status_code=404, detail="favicon not found")


@app.get("/favicon", include_in_schema=False)
def favicon_any():
    favicon_path = current_favicon_path()
    if favicon_path and favicon_path.exists():
        return FileResponse(favicon_path, headers={"Cache-Control": "no-cache"})
    raise HTTPException(status_code=404, detail="favicon not found")


@app.get("/health")
def health():
    airllm_module = None
    try:
        import airllm as _airllm_pkg
        airllm_module = getattr(_airllm_pkg, "__file__", None)
    except Exception:
        airllm_module = None
    return {
        "status": "ok",
        **runtime_state(),
        "app_file": __file__,
        "airllm_module": airllm_module,
    }


@app.post("/model/resolve")
def resolve_model(request: Optional[SettingsRequest] = Body(default=None)):
    payload = request or SettingsRequest()
    model_id, model_path = _coerce_local_path_input(payload.model_id, payload.model_path)
    resolved = resolve_requested_source(
        model_id=model_id,
        model_path=model_path,
        model_base_dir=payload.model_base_dir,
    )
    local_expected = bool(resolved["model_path"] or resolved["model_base_dir"])
    exists = os.path.isdir(resolved["model_source"]) if local_expected else None
    return {
        **resolved,
        "local_expected": local_expected,
        "exists": exists,
    }


@app.post("/settings")
def update_settings(request: Optional[SettingsRequest] = Body(default=None)):
    payload = request or SettingsRequest()
    model_id, model_path = _coerce_local_path_input(payload.model_id, payload.model_path)
    apply_settings(
        model_id=model_id,
        model_path=model_path,
        model_base_dir=payload.model_base_dir,
        device=payload.device,
    )
    if payload.persist_env:
        persist_runtime_to_env()

    return {
        "status": "settings_updated",
        **runtime_state(),
        "persist_env": payload.persist_env,
        "env_file": str(ENV_FILE),
    }


@app.get("/models")
def list_models(base_dir: Optional[str] = None):
    target_base_dir = normalize_optional(base_dir) or get_runtime_field("model_base_dir")
    if not target_base_dir:
        return {"base_dir": None, "models": [], "unsupported_models": [], "directories": [], "scan_errors": []}

    if not os.path.isdir(target_base_dir):
        return {
            "base_dir": target_base_dir,
            "models": [],
            "unsupported_models": [],
            "directories": [],
            "scan_errors": [f"Base directory does not exist or is inaccessible: {target_base_dir}"],
        }

    models, unsupported_models, scan_errors = discover_models(target_base_dir)
    directories = list_candidate_directories(target_base_dir)
    return {
        "base_dir": target_base_dir,
        "models": models,
        "unsupported_models": unsupported_models,
        "directories": directories,
        "scan_errors": scan_errors,
    }


@app.get("/hf/compatibility")
def hf_compatibility(model_id: str, revision: Optional[str] = None):
    try:
        return check_hf_repo_compatibility(model_id=model_id, revision=revision)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Compatibility check failed: {exc}") from exc

@app.post("/hf/download")
def hf_download(request: HFDownloadRequest):
    model_id = normalize_hf_repo_id(request.model_id)
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required.")

    target_base_dir = normalize_optional(request.base_dir) or get_runtime_field("model_base_dir")
    if not target_base_dir:
        raise HTTPException(
            status_code=400,
            detail="No base directory set. Provide base_dir or configure AIRLLM_MODEL_BASE_DIR first.",
        )

    try:
        os.makedirs(target_base_dir, exist_ok=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to create/access base_dir: {exc}") from exc

    try:
        target_dir = resolve_hf_download_dir(target_base_dir, model_id, request.subdir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = {
        "model_id": model_id,
        "base_dir": str(Path(target_base_dir).resolve()),
        "target_dir": str(target_dir),
        "revision": normalize_optional(request.revision),
        "allow_patterns": parse_patterns(request.allow_patterns),
        "ignore_patterns": parse_patterns(request.ignore_patterns),
        "set_as_active_model": request.set_as_active_model,
        "persist_env": request.persist_env,
    }
    job_id = start_hf_download_job(payload)
    return {
        "status": "started",
        "job_id": job_id,
        "downloaded_model_id": model_id,
        "downloaded_path": str(target_dir),
        "base_dir": payload["base_dir"],
        "set_as_active_model": request.set_as_active_model,
        "persist_env": request.persist_env,
        "env_file": str(ENV_FILE),
    }


@app.get("/hf/download/active")
def hf_download_active():
    return active_download()


@app.get("/hf/download/{job_id}")
def hf_download_status(job_id: str):
    job = download_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"No download job found for id {job_id}")
    return job


@app.post("/load/start")
def preload_model_start(request: Optional[LoadRequest] = Body(default=None)):
    payload = request or LoadRequest()
    model_id, model_path = _coerce_local_path_input(payload.model_id, payload.model_path)
    job_id = start_load_job(
        {
            "model_id": model_id,
            "model_path": model_path,
            "model_base_dir": payload.model_base_dir,
            "device": payload.device,
            "force_reload": payload.force_reload,
            "persist_env": payload.persist_env,
        }
    )
    return {
        "status": "started",
        "job_id": job_id,
    }


@app.get("/load/active")
def load_active():
    return active_load_job()



@app.post("/load/stop")
def load_stop(job_id: Optional[str] = Body(default=None, embed=True)):
    return request_stop_load_job(job_id)

@app.get("/load/{job_id}")
def load_status(job_id: str):
    job = load_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"No load job found for id {job_id}")
    return job


@app.post("/load")
def preload_model(request: Optional[LoadRequest] = Body(default=None)):
    payload = request or LoadRequest()
    model_id, model_path = _coerce_local_path_input(payload.model_id, payload.model_path)
    try:
        load_model(
            model_id=model_id,
            model_path=model_path,
            model_base_dir=payload.model_base_dir,
            device=payload.device,
            force_reload=payload.force_reload,
        )

        persisted_env = False
        if payload.persist_env:
            persist_runtime_to_env()
            persisted_env = True
        else:
            persisted_env = persist_runtime_to_env_if_changed()

        return {
            "status": "loaded",
            **runtime_state(),
            "persist_env_requested": payload.persist_env,
            "persisted_env": persisted_env,
            "env_file": str(ENV_FILE),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc

def _is_airllm_generation_compat_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    signatures = (
        "cannot unpack non-iterable nonetype object",
        "position_embeddings",
        "_is_stateful",
        "nonetype object has no attribute 'shape'",
    )
    return any(signature in message for signature in signatures)


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    def _build_inputs(current_model, target_device: str):
        tokenized = current_model.tokenizer(
            [request.prompt],
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            padding=False,
        )

        input_ids_local = tokenized["input_ids"].to(target_device)
        attention_mask_local = tokenized.get("attention_mask")
        if attention_mask_local is not None:
            attention_mask_local = attention_mask_local.to(target_device)

        kwargs_local = {
            "max_new_tokens": request.max_new_tokens,
            "use_cache": False,
        }
        if attention_mask_local is not None:
            kwargs_local["attention_mask"] = attention_mask_local

        eos_token_id_local = getattr(current_model.tokenizer, "eos_token_id", None)
        pad_token_id_local = getattr(current_model.tokenizer, "pad_token_id", None)
        if eos_token_id_local is not None:
            kwargs_local.setdefault("eos_token_id", eos_token_id_local)
            kwargs_local.setdefault("pad_token_id", pad_token_id_local or eos_token_id_local)

        if request.do_sample:
            kwargs_local.update(
                {
                    "do_sample": True,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                }
            )
        return input_ids_local, kwargs_local

    def _decode_output(current_model, output_value):
        if hasattr(output_value, "sequences"):
            sequence_value = output_value.sequences[0]
        elif isinstance(output_value, tuple):
            sequence_value = output_value[0]
        else:
            sequence_value = output_value

        if getattr(sequence_value, "ndim", 0) > 1:
            sequence_value = sequence_value[0]

        full_text_local = current_model.tokenizer.decode(sequence_value, skip_special_tokens=True)
        generated_local = (
            full_text_local[len(request.prompt) :].lstrip()
            if full_text_local.startswith(request.prompt)
            else full_text_local
        )
        return full_text_local, generated_local

    try:
        model = load_model()
        runtime = runtime_state()
        target_device = runtime.get("device") or "cpu"

        if runtime.get("runtime_backend") in ("llama_cpp", "llama_server") and hasattr(model, "generate_text"):
            generated = model.generate_text(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            )
            full_text = request.prompt + generated
            runtime = runtime_state()
            return GenerateResponse(
                model_id=runtime["model_id"],
                model_source=runtime["model_source"],
                device=runtime["device"],
                runtime_backend=runtime.get("runtime_backend"),
                runtime_note=runtime.get("runtime_note"),
                text=full_text,
                generated_text=generated,
            )

        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.use_cache = False

        try:
            input_ids, generate_kwargs = _build_inputs(model, target_device)
            output = model.generate(input_ids, **generate_kwargs)
        except Exception as primary_exc:
            if runtime.get("runtime_backend") == "airllm" and _is_airllm_generation_compat_error(primary_exc):
                print("[generate] AirLLM generation incompatibility detected; retrying with Transformers runtime fallback.")
                traceback.print_exc()

                model = load_model(force_reload=True, runtime_backend="transformers")
                runtime = runtime_state()
                target_device = runtime.get("device") or "cpu"
                input_ids, generate_kwargs = _build_inputs(model, target_device)
                output = model.generate(input_ids, **generate_kwargs)
            else:
                raise

        full_text, generated = _decode_output(model, output)

        runtime = runtime_state()
        return GenerateResponse(
            model_id=runtime["model_id"],
            model_source=runtime["model_source"],
            device=runtime["device"],
            runtime_backend=runtime.get("runtime_backend"),
            runtime_note=runtime.get("runtime_note"),
            text=full_text,
            generated_text=generated,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {type(exc).__name__}: {exc}",
        ) from exc

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
























