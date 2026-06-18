import json
import os
import threading
import time
import uuid
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download

from .config import ENV_FILE, HF_TOKEN
from .model_catalog import resolve_airllm_runtime_from_architectures
from .model_runtime import apply_download_result, normalize_optional, persist_runtime_to_env

_download_jobs = {}
_download_lock = threading.Lock()


def _new_download_job(model_id: str, base_dir: str, target_dir: str):
    job_id = str(uuid.uuid4())
    with _download_lock:
        _download_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "model_id": model_id,
            "base_dir": base_dir,
            "target_dir": target_dir,
            "progress": 0,
            "current": 0,
            "total": None,
            "current_file": None,
            "logs": [f"Job created for {model_id}"],
            "error": None,
            "started_at": time.time(),
            "finished_at": None,
        }
    return job_id


def _job_update(job_id: str, **fields):
    with _download_lock:
        job = _download_jobs.get(job_id)
        if not job:
            return
        job.update(fields)


def _job_log(job_id: str, message: str):
    with _download_lock:
        job = _download_jobs.get(job_id)
        if not job:
            return
        logs = job.get("logs", [])
        logs.append(message)
        if len(logs) > 300:
            logs = logs[-300:]
        job["logs"] = logs


def normalize_hf_repo_id(model_id: Optional[str]) -> Optional[str]:
    model_id = normalize_optional(model_id)
    if not model_id:
        return None

    # Accept accidental Windows separators and normalize to HF repo syntax.
    normalized = model_id.replace("\\", "/")
    normalized = "/".join(part.strip() for part in normalized.split("/") if part.strip())
    return normalized or None


def _get_transformers_support():
    info = {
        "available": False,
        "version": "unknown",
        "supported_model_types": set(),
    }
    try:
        import transformers
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        info["available"] = True
        info["version"] = getattr(transformers, "__version__", "unknown")
        info["supported_model_types"] = set(CONFIG_MAPPING.keys())
    except Exception:
        pass
    return info


def check_hf_repo_compatibility(model_id: str, revision: Optional[str] = None):
    normalized_model_id = normalize_hf_repo_id(model_id)
    if not normalized_model_id:
        raise ValueError("model_id is required.")

    normalized_revision = normalize_optional(revision)
    api = HfApi(token=HF_TOKEN)
    repo_files = list(api.list_repo_files(repo_id=normalized_model_id, revision=normalized_revision))
    repo_files = [path_value for path_value in repo_files if not path_value.endswith("/")]

    has_config = "config.json" in repo_files
    has_tokenizer = any(
        candidate in repo_files
        for candidate in (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "spiece.model",
        )
    )
    has_weights = any(path_value.endswith(".safetensors") or path_value.endswith(".bin") for path_value in repo_files)
    has_index = (
        "model.safetensors.index.json" in repo_files
        or "pytorch_model.bin.index.json" in repo_files
    )

    config_error = None
    config_transformers_version = None
    model_type = None
    architectures = []
    if has_config:
        try:
            config_path = hf_hub_download(
                repo_id=normalized_model_id,
                filename="config.json",
                revision=normalized_revision,
                token=HF_TOKEN,
            )
            config_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
            model_type = normalize_optional(str(config_data.get("model_type") or ""))
            config_transformers_version = normalize_optional(str(config_data.get("transformers_version") or ""))
            raw_architectures = config_data.get("architectures")
            if isinstance(raw_architectures, list):
                architectures = [str(item) for item in raw_architectures if normalize_optional(str(item or ""))]
        except Exception as exc:
            config_error = str(exc)

    transformers_support = _get_transformers_support()
    transformers_model_type_supported = True
    if model_type and transformers_support["available"]:
        transformers_model_type_supported = model_type in transformers_support["supported_model_types"]

    airllm_architecture_supported, airllm_runtime_class, airllm_reason = resolve_airllm_runtime_from_architectures(
        architectures
    )

    reasons = []
    warnings = []

    if not has_config:
        reasons.append("Missing config.json in repository.")
    if config_error:
        reasons.append(f"Unable to parse config.json: {config_error}")
    if not has_weights:
        reasons.append("No .safetensors or .bin weight files found.")
    if has_config and not model_type:
        reasons.append("config.json exists but has no model_type.")
    if model_type and transformers_support["available"] and not transformers_model_type_supported:
        reasons.append(
            f"model_type={model_type} is not supported by installed transformers {transformers_support['version']}."
        )

    # Unsupported optimization should not block usage; it is reported as a warning.
    if not airllm_architecture_supported:
        warnings.append(
            airllm_reason or "Architecture is not mapped to an optimized AirLLM runtime; standard Transformers runtime will be used."
        )

    if not has_index:
        warnings.append(
            "No sharded weight index found. AirLLM may still work if weights are non-sharded and index generation succeeds."
        )
    if not has_tokenizer:
        warnings.append("No tokenizer files detected (generation may fail even if model loads).")
    if not transformers_support["available"]:
        warnings.append("Unable to inspect local transformers model registry.")

    supported = len(reasons) == 0
    airllm_optimized = bool(supported and airllm_architecture_supported)

    if supported:
        status = "compatible_with_warnings" if warnings else "compatible"
        runtime_mode = "airllm" if airllm_optimized else "transformers"
    else:
        status = "incompatible"
        runtime_mode = None

    return {
        "status": status,
        "supported": supported,
        "airllm_optimized": airllm_optimized,
        "runtime_mode": runtime_mode,
        "model_id": normalized_model_id,
        "revision": normalized_revision,
        "transformers_version": transformers_support["version"],
        "config_transformers_version": config_transformers_version,
        "model_type": model_type,
        "architectures": architectures,
        "airllm_runtime_class": airllm_runtime_class,
        "reasons": reasons,
        "warnings": warnings,
        "file_checks": {
            "has_config": has_config,
            "has_weights": has_weights,
            "has_index": has_index,
            "has_tokenizer": has_tokenizer,
            "repo_file_count": len(repo_files),
        },
    }


def parse_patterns(patterns: Optional[str]):
    patterns = normalize_optional(patterns)
    if not patterns:
        return None
    values = []
    for chunk in patterns.replace("\n", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(chunk)
    return values or None


def resolve_hf_download_dir(base_dir: str, model_id: str, subdir: Optional[str]):
    model_id = normalize_hf_repo_id(model_id)
    if not model_id:
        raise ValueError("model_id is required.")
    base = Path(base_dir)
    if subdir:
        rel = Path(subdir.strip())
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("subdir must be a relative path under the base directory.")
        return (base / rel).resolve()
    return (base / Path(*model_id.split("/"))).resolve()


def _run_hf_download_job(job_id: str, payload: dict):
    model_id = normalize_hf_repo_id(payload.get("model_id")) or payload.get("model_id")
    base_dir = payload["base_dir"]
    target_dir = payload["target_dir"]
    revision = payload.get("revision")
    allow_patterns = payload.get("allow_patterns")
    ignore_patterns = payload.get("ignore_patterns")
    set_as_active_model = payload.get("set_as_active_model", True)
    persist_env = payload.get("persist_env", True)

    _job_update(job_id, status="downloading")
    _job_log(job_id, f"Starting download for {model_id}")
    _job_log(job_id, f"Target directory: {target_dir}")
    _job_update(job_id, progress=1, current=0, total=None, current_file="Resolving repository file list")

    def _matches_any(path_value: str, patterns):
        if not patterns:
            return False
        return any(fnmatch(path_value, pattern) for pattern in patterns)

    try:
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        api = HfApi(token=HF_TOKEN)
        repo_files = list(api.list_repo_files(repo_id=model_id, revision=revision))
        repo_files = [path_value for path_value in repo_files if not path_value.endswith("/")]

        if allow_patterns:
            repo_files = [path_value for path_value in repo_files if _matches_any(path_value, allow_patterns)]
        if ignore_patterns:
            repo_files = [path_value for path_value in repo_files if not _matches_any(path_value, ignore_patterns)]

        total_files = len(repo_files)
        if total_files == 0:
            raise RuntimeError("No files matched the selected patterns for download.")

        _job_update(job_id, total=total_files, current=0, progress=1)
        _job_log(job_id, f"Resolved {total_files} files to download.")

        done = 0
        last_log = 0.0
        for repo_file in repo_files:
            progress = max(1, int((done / total_files) * 100))
            _job_update(
                job_id,
                current=done,
                total=total_files,
                progress=progress,
                current_file=repo_file,
            )
            hf_hub_download(
                repo_id=model_id,
                filename=repo_file,
                revision=revision,
                local_dir=target_dir,
                token=HF_TOKEN,
            )
            done += 1
            progress = int((done / total_files) * 100)
            _job_update(job_id, current=done, total=total_files, progress=progress, current_file=repo_file)
            now = time.time()
            if now - last_log >= 0.5 or done == total_files:
                _job_log(job_id, f"Download progress: {done}/{total_files} files")
                last_log = now
    except Exception as exc:
        _job_log(job_id, f"Hugging Face download failed: {exc}")
        _job_update(job_id, status="failed", error=str(exc), finished_at=time.time())
        return

    _job_log(job_id, "Download finished successfully.")
    apply_download_result(
        model_id=model_id,
        base_dir=base_dir,
        target_dir=target_dir,
        set_as_active_model=set_as_active_model,
    )

    if persist_env:
        persist_runtime_to_env()
        _job_log(job_id, f"Persisted settings to {ENV_FILE}")

    _job_update(
        job_id,
        status="completed",
        progress=100,
        error=None,
        finished_at=time.time(),
        set_as_active_model=set_as_active_model,
        persist_env=persist_env,
    )


def start_hf_download_job(payload: dict):
    job_id = _new_download_job(
        model_id=payload["model_id"],
        base_dir=payload["base_dir"],
        target_dir=payload["target_dir"],
    )
    thread = threading.Thread(target=_run_hf_download_job, args=(job_id, payload), daemon=True)
    thread.start()
    return job_id


def active_download():
    with _download_lock:
        active = [job for job in _download_jobs.values() if job.get("status") in ("queued", "downloading")]
        if not active:
            return {"job_id": None, "status": "idle"}
        active.sort(key=lambda job: job.get("started_at", 0), reverse=True)
        job = active[0]
        return {
            "job_id": job.get("job_id"),
            "status": job.get("status"),
            "model_id": job.get("model_id"),
            "target_dir": job.get("target_dir"),
            "started_at": job.get("started_at"),
        }


def download_job_snapshot(job_id: str):
    with _download_lock:
        job = _download_jobs.get(job_id)
        if not job:
            return None
        return dict(job)
