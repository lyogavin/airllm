import json
import os
from pathlib import Path


_TRANSFORMERS_SUPPORT = None


def _get_transformers_support():
    global _TRANSFORMERS_SUPPORT
    if _TRANSFORMERS_SUPPORT is not None:
        return _TRANSFORMERS_SUPPORT

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

    _TRANSFORMERS_SUPPORT = info
    return info


def _is_model_type_supported(model_type: str):
    info = _get_transformers_support()
    if not model_type or not info["available"]:
        return True, info["version"]
    if model_type in info["supported_model_types"]:
        return True, info["version"]
    return False, info["version"]


def resolve_airllm_runtime_from_architectures(architectures):
    if not architectures:
        return False, None, "No architectures entry found in config.json."

    first_arch = str(architectures[0])
    if "Qwen2ForCausalLM" in first_arch:
        return True, "AirLLMQWen2", None
    if "QWen" in first_arch:
        return True, "AirLLMQWen", None
    if "Baichuan" in first_arch:
        return True, "AirLLMBaichuan", None
    if "ChatGLM" in first_arch:
        return True, "AirLLMChatGLM", None
    if "InternLM" in first_arch:
        return True, "AirLLMInternLM", None
    if "Mistral" in first_arch:
        return True, "AirLLMMistral", None
    if "Mixtral" in first_arch:
        return True, "AirLLMMixtral", None
    if "Llama" in first_arch:
        return True, "AirLLMLlama2", None

    return (
        False,
        None,
        f"Architecture {first_arch} is not mapped to an optimized AirLLM runtime in this build.",
    )


def inspect_local_model_dir_details(root_path: Path, files, exhaustive: bool = False):
    file_set = set(files)
    has_gguf = any(name.lower().endswith(".gguf") for name in files)
    has_config = "config.json" in file_set
    has_airllm_index = (
        "pytorch_model.bin.index.json" in file_set
        or "model.safetensors.index.json" in file_set
    )
    has_weights = (
        "pytorch_model.bin" in file_set
        or "pytorch_model.bin.index.json" in file_set
        or "model.safetensors.index.json" in file_set
        or any(name.endswith(".safetensors") for name in files)
    )

    details = {
        "selectable": False,
        "reason": None,
        "warnings": [],
        "model_type": None,
        "architectures": [],
        "runtime_mode": None,
        "airllm_optimized": False,
        "airllm_runtime_class": None,
        "optimization_note": None,
        "has_config": has_config,
        "has_weights": has_weights,
        "has_airllm_index": has_airllm_index,
    }

    model_type = None
    architectures = []
    config_error = None
    if has_config:
        config_path = root_path / "config.json"
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
            model_type = config_data.get("model_type")
            raw_architectures = config_data.get("architectures")
            if isinstance(raw_architectures, list):
                architectures = [str(item) for item in raw_architectures if str(item).strip()]
        except Exception as exc:
            config_error = f"Invalid config.json: {exc}"

    details["model_type"] = model_type
    details["architectures"] = architectures

    if has_gguf and not (has_config and has_weights and model_type):
        details["reason"] = "GGUF directory (not a Transformers checkpoint)."
        return details
    if has_config and config_error:
        details["reason"] = config_error
        return details
    if has_config and not model_type:
        details["reason"] = "config.json exists but has no model_type."
        return details
    if has_config and model_type:
        supported_model_type, transformers_version = _is_model_type_supported(model_type)
        if not supported_model_type:
            details["reason"] = (
                f"model_type={model_type} is not supported by installed transformers {transformers_version}."
            )
            return details
    if has_config and model_type and not has_weights:
        details["reason"] = f"model_type={model_type} but no PyTorch/safetensors weights found."
        return details

    if has_config and model_type and has_weights:
        details["selectable"] = True
        details["runtime_mode"] = "transformers"
        details["optimization_note"] = "Will use standard Transformers runtime."

        airllm_supported, airllm_runtime_class, airllm_reason = resolve_airllm_runtime_from_architectures(architectures)
        details["airllm_optimized"] = bool(airllm_supported)
        details["airllm_runtime_class"] = airllm_runtime_class
        if airllm_supported:
            details["runtime_mode"] = "airllm"
            details["optimization_note"] = "AirLLM optimization available."
            if not has_airllm_index:
                details["warnings"].append(
                    "No sharded index file found. Index generation will be attempted before optimized load."
                )
        else:
            details["runtime_mode"] = "transformers"
            details["optimization_note"] = (
                airllm_reason or "AirLLM optimization is unavailable; standard Transformers runtime will be used."
            )
        return details

    if exhaustive:
        details["reason"] = "No compatible Transformers config.json + weights found."
    return details


def inspect_local_model_dir(root_path: Path, files, exhaustive: bool = False):
    details = inspect_local_model_dir_details(root_path, files, exhaustive=exhaustive)
    return details["selectable"], details["reason"]


def validate_local_model_source(model_source: str):
    source_path = Path(model_source)
    if source_path.is_file() and source_path.suffix.lower() == ".gguf":
        raise ValueError(
            f"Selected model is GGUF ({model_source}). AirLLM expects a Hugging Face Transformers checkpoint directory."
        )
    if not source_path.is_dir():
        return None

    files = [entry.name for entry in source_path.iterdir() if entry.is_file()]
    details = inspect_local_model_dir_details(source_path, files, exhaustive=True)
    if details["selectable"]:
        return details

    reason = details["reason"]
    if reason and "GGUF" in reason:
        raise ValueError(
            f"Selected model path is GGUF-based: {model_source}. Choose a Transformers model folder with config.json + model weights."
        )
    if reason and "no model_type" in reason:
        raise ValueError(
            f"Selected model path is not loadable by Transformers: {model_source} ({reason})"
        )
    if reason and "not supported by installed transformers" in reason:
        raise ValueError(
            f"Selected model requires a newer Transformers architecture than this app runtime: {model_source} ({reason})"
        )
    if reason and "No compatible Transformers config.json + weights found" in reason:
        raise ValueError(
            f"Selected local model path is not a usable Transformers checkpoint: {model_source} ({reason})"
        )
    if reason:
        raise ValueError(
            f"Selected local model path is not compatible: {model_source} ({reason})"
        )
    return None


def discover_models(base_dir: str, max_depth: int = 4, limit: int = 500):
    base_path = Path(base_dir)
    if not base_path.exists() or not base_path.is_dir():
        return [], [], [f"Base directory not found: {base_dir}"]

    models = []
    unsupported_models = []
    errors = []

    def _onerror(err):
        errors.append(str(err))

    for root, dirs, files in os.walk(base_path, onerror=_onerror):
        root_path = Path(root)
        rel_path = root_path.relative_to(base_path)
        depth = len(rel_path.parts)
        if depth > max_depth:
            dirs[:] = []
            continue

        details = inspect_local_model_dir_details(root_path, files)
        if details["selectable"]:
            label = "." if str(rel_path) == "." else str(rel_path).replace("\\", "/")
            models.append(
                {
                    "label": label,
                    "path": str(root_path),
                    "model_type": details.get("model_type"),
                    "runtime_mode": details.get("runtime_mode"),
                    "airllm_optimized": bool(details.get("airllm_optimized")),
                    "airllm_runtime_class": details.get("airllm_runtime_class"),
                    "optimization_note": details.get("optimization_note"),
                    "warnings": details.get("warnings") or [],
                }
            )
            dirs[:] = []
        elif details.get("reason"):
            label = "." if str(rel_path) == "." else str(rel_path).replace("\\", "/")
            unsupported_models.append({"label": label, "path": str(root_path), "reason": details["reason"]})

        if len(models) >= limit:
            break

    models.sort(key=lambda item: item["label"].lower())
    unsupported_models.sort(key=lambda item: item["label"].lower())
    return models, unsupported_models, errors


def list_candidate_directories(base_dir: str, max_depth: int = 2, limit: int = 300):
    base_path = Path(base_dir)
    if not base_path.exists() or not base_path.is_dir():
        return []

    directories = []
    for root, dirs, _ in os.walk(base_path):
        root_path = Path(root)
        rel_path = root_path.relative_to(base_path)
        depth = len(rel_path.parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        if depth > 0:
            label = str(rel_path).replace("\\", "/")
            directories.append({"label": label, "path": str(root_path)})
        if len(directories) >= limit:
            break

    directories.sort(key=lambda item: item["label"].lower())
    return directories
