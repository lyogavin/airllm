import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from huggingface_hub import constants as hf_constants
from transformers import AutoConfig

# Curated registry of friendly aliases (similar to Ollama model tags)
MODEL_ALIASES: Dict[str, str] = {
    # Llama 3 / 3.1 / 3.2 / 3.3
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3:8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3:70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama3.1": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.1:8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.1:70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "llama3.1:405b": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.3": "meta-llama/Llama-3.3-70B-Instruct",
    "llama3.3:70b": "meta-llama/Llama-3.3-70B-Instruct",

    # DeepSeek
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek-r1:70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-r1:32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-r1:14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1:8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-r1:7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1:1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",

    # Qwen 2.5
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5:32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5:72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen2.5-coder:7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder:32b": "Qwen/Qwen2.5-Coder-32B-Instruct",

    # Mistral & Mixtral
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral:8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral:8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",

    # Kimi / Moonshot
    "kimi-k3": "MoonshotAI/Kimi-K3",

    # Popular benchmark / open models
    "platypus2:70b": "garage-bAInd/Platypus2-70B-instruct",
}


def resolve_model_name(name_or_alias: str) -> str:
    """Resolve a friendly alias (e.g. 'llama3:70b') to HuggingFace repo ID or return path as-is."""
    clean_name = name_or_alias.strip()
    return MODEL_ALIASES.get(clean_name.lower(), clean_name)


def get_alias_for_repo(repo_id: str) -> Optional[str]:
    """Find any friendly alias pointing to this repository ID."""
    clean_repo = repo_id.strip()
    for alias, target in MODEL_ALIASES.items():
        if target.lower() == clean_repo.lower():
            return alias
    return None


def format_size(bytes_size: int) -> str:
    """Format bytes into human-readable string (e.g. 14.2 GB)."""
    if bytes_size <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(bytes_size)
    while size >= 1024.0 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.1f} {units[i]}"


def get_dir_size(path: Path) -> int:
    """Calculate recursive directory size in bytes."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_symlink():
                continue
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(Path(entry.path))
    except (OSError, PermissionError):
        pass
    return total


def get_hf_hub_cache_dir() -> Path:
    """Return the base path for HuggingFace hub cache."""
    return Path(hf_constants.HF_HUB_CACHE)


def list_local_models() -> List[Dict[str, Any]]:
    """Scan local Hugging Face and AirLLM cache directories for downloaded/sharded models."""
    hub_dir = get_hf_hub_cache_dir()
    models = []
    if not hub_dir.exists():
        return models

    # Hugging Face hub folders follow: models--<org>--<repo>
    for entry in hub_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue

        parts = entry.name[len("models--"):].split("--")
        repo_id = "/".join(parts) if len(parts) >= 2 else parts[0]
        alias = get_alias_for_repo(repo_id)

        # Check snapshots and split folders
        snapshots_dir = entry / "snapshots"
        has_snapshot = snapshots_dir.exists() and any(snapshots_dir.iterdir())
        
        # Check if splitted_model exists in snapshot or custom folder
        split_dirs = []
        if has_snapshot:
            for snap in snapshots_dir.iterdir():
                if snap.is_dir():
                    for sub in snap.iterdir():
                        if sub.is_dir() and sub.name.startswith("splitted_model"):
                            split_dirs.append(sub)

        total_size = get_dir_size(entry)
        last_modified = entry.stat().st_mtime

        models.append({
            "name": alias or repo_id,
            "repo_id": repo_id,
            "path": str(entry),
            "size": total_size,
            "size_formatted": format_size(total_size),
            "sharded": len(split_dirs) > 0,
            "shards_count": len(split_dirs),
            "modified": last_modified,
        })

    models.sort(key=lambda m: m["modified"], reverse=True)
    return models


def remove_model_cache(name_or_alias: str) -> bool:
    """Delete the model cache (and its shards) from the local system."""
    repo_id = resolve_model_name(name_or_alias)
    hub_dir = get_hf_hub_cache_dir()

    # Form the folder name 'models--org--repo'
    parts = repo_id.replace("/", "--")
    folder_name = f"models--{parts}"
    target = hub_dir / folder_name

    if target.exists() and target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
        return True

    # If user provided direct path
    p = Path(name_or_alias)
    if p.exists() and p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
        return True

    return False


def get_model_info(name_or_alias: str, hf_token: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve metadata and configuration details about a model."""
    repo_id = resolve_model_name(name_or_alias)
    token_kwargs = {"token": hf_token} if hf_token else {}

    try:
        config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True, **token_kwargs)
        cfg_dict = config.to_dict()
    except Exception as e:
        return {
            "name": name_or_alias,
            "repo_id": repo_id,
            "error": str(e)
        }

    arch = cfg_dict.get("architectures", ["Unknown"])[0] if cfg_dict.get("architectures") else "Unknown"
    num_layers = cfg_dict.get("num_hidden_layers", cfg_dict.get("n_layer", "N/A"))
    hidden_size = cfg_dict.get("hidden_size", cfg_dict.get("n_embd", "N/A"))
    num_heads = cfg_dict.get("num_attention_heads", cfg_dict.get("n_head", "N/A"))
    max_pos = cfg_dict.get("max_position_embeddings", cfg_dict.get("max_seq_len", "N/A"))
    torch_dtype = cfg_dict.get("torch_dtype", "N/A")

    return {
        "name": name_or_alias,
        "repo_id": repo_id,
        "architecture": arch,
        "layers": num_layers,
        "hidden_size": hidden_size,
        "attention_heads": num_heads,
        "context_length": max_pos,
        "dtype": torch_dtype,
    }
