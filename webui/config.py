import os
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE, override=False)


def _getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _getenv_int(name: str, default: int) -> int:
    value = _getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


MODEL_ID_DEFAULT = _getenv("AIRLLM_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MODEL_PATH_DEFAULT = _getenv("AIRLLM_MODEL_PATH")
MODEL_BASE_DIR_DEFAULT = _getenv("AIRLLM_MODEL_BASE_DIR")
DEVICE_DEFAULT = _getenv("AIRLLM_DEVICE", _default_device())
if DEVICE_DEFAULT:
    DEVICE_DEFAULT = DEVICE_DEFAULT.split()[0]
    if DEVICE_DEFAULT == "cuda":
        DEVICE_DEFAULT = "cuda:0"
HF_TOKEN = _getenv("HF_TOKEN")
MAX_INPUT_TOKENS = _getenv_int("AIRLLM_MAX_INPUT_TOKENS", 1024)
PORT = _getenv_int("PORT", 8000)


def _find_favicon() -> Optional[Path]:
    for name in ("favicon.ico", "favicon.png", "favicon.svg", "favicon.jpg", "favicon.jpeg"):
        candidate = PROJECT_ROOT / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def current_favicon_path() -> Optional[Path]:
    return _find_favicon()

