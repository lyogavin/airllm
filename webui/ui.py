from html import escape

from .config import PROJECT_ROOT

UI_TEMPLATE_PATH = PROJECT_ROOT / "templates" / "index.html"


def _load_ui_html_template() -> str:
    if not UI_TEMPLATE_PATH.exists():
        raise RuntimeError(f"UI template not found at {UI_TEMPLATE_PATH}")
    return UI_TEMPLATE_PATH.read_text(encoding="utf-8")


def render_ui_html(state: dict) -> str:
    html = _load_ui_html_template()
    return (
        html.replace("__MODEL_ID_VALUE__", escape(state.get("model_id") or ""))
        .replace("__MODEL_PATH_VALUE__", escape(state.get("model_path") or ""))
        .replace("__MODEL_BASE_DIR_VALUE__", escape(state.get("model_base_dir") or ""))
        .replace("__DEVICE_VALUE__", escape(state.get("device") or ""))
    )