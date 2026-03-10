from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(4, ge=1, le=1024)
    do_sample: bool = False
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1, le=500)


class LoadRequest(BaseModel):
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    model_base_dir: Optional[str] = None
    device: Optional[str] = None
    force_reload: bool = False
    persist_env: bool = False


class SettingsRequest(BaseModel):
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    model_base_dir: Optional[str] = None
    device: Optional[str] = None
    persist_env: bool = True


class HFDownloadRequest(BaseModel):
    model_id: str = Field(..., min_length=1)
    base_dir: Optional[str] = None
    subdir: Optional[str] = None
    revision: Optional[str] = None
    allow_patterns: Optional[str] = None
    ignore_patterns: Optional[str] = None
    set_as_active_model: bool = True
    persist_env: bool = True


class GenerateResponse(BaseModel):
    model_id: str
    model_source: str
    device: str
    runtime_backend: Optional[str] = None
    runtime_note: Optional[str] = None
    text: str
    generated_text: str
