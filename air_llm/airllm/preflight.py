"""Metadata-only compatibility and capacity checks for very large checkpoints."""

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import huggingface_hub

from .utils import estimated_split_size_bytes


KIMI_TEXT_LAYOUT = {
    "embed": "language_model.model.embed_tokens",
    "layer_prefix": "language_model.model.layers",
    "norm": "language_model.model.norm",
    "lm_head": "language_model.lm_head",
}
STANDARD_LAYOUT = {
    "embed": "model.embed_tokens",
    "layer_prefix": "model.layers",
    "norm": "model.norm",
    "lm_head": "lm_head",
}


@dataclass
class ModelPreflightReport:
    model: str
    architecture: str
    transformers_version: str | None
    compatibility: str
    layer_layout: dict[str, str]
    layer_count: int
    tensor_count: int
    shard_count: int
    checkpoint_size_bytes: int
    estimated_split_size_bytes: int
    average_stream_unit_bytes: int | None
    offload_free_bytes: int | None = None
    required_packages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _metadata_path(model, hf_token=None, cache_dir=None):
    local_path = Path(model)
    if local_path.exists():
        return local_path
    return Path(
        huggingface_hub.snapshot_download(
            model,
            token=hf_token,
            cache_dir=cache_dir,
            ignore_patterns=["*.safetensors", "*.bin"],
        )
    )


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_index(model_path):
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        path = model_path / name
        if path.exists():
            return _read_json(path)
    return None


def _layout_for(architecture, weight_names):
    if architecture == "KimiK25ForConditionalGeneration":
        return KIMI_TEXT_LAYOUT, "text-only"
    if any(name.startswith("model.layers.") for name in weight_names):
        return STANDARD_LAYOUT, "supported"
    return STANDARD_LAYOUT, "unknown"


def inspect_model(model, compression=None, hf_token=None, cache_dir=None, offload_dir=None):
    """Inspect config and weight index without downloading model weight shards."""
    model_path = _metadata_path(model, hf_token=hf_token, cache_dir=cache_dir)
    config = _read_json(model_path / "config.json")
    architecture = (config.get("architectures") or [""])[0]
    text_config = config.get("text_config") or {}
    transformers_version = config.get("transformers_version") or text_config.get("transformers_version")

    index = _find_index(model_path)
    if index is None:
        weight_names = []
        checkpoint_size = sum(
            path.stat().st_size
            for pattern in ("*.safetensors", "*.bin")
            for path in model_path.glob(pattern)
        )
        shard_count = 1 if checkpoint_size else 0
    else:
        weight_map = index.get("weight_map", {})
        weight_names = list(weight_map)
        checkpoint_size = int(index.get("metadata", {}).get("total_size", 0))
        shard_count = len(set(weight_map.values()))

    layout, compatibility = _layout_for(architecture, weight_names)
    layer_pattern = re.compile(rf"^{re.escape(layout['layer_prefix'])}\.(\d+)\.")
    indexed_layer_count = len(
        {
            int(match.group(1))
            for name in weight_names
            if (match := layer_pattern.match(name)) is not None
        }
    )
    configured_layer_count = text_config.get("num_hidden_layers", config.get("num_hidden_layers"))
    layer_count = int(configured_layer_count) if configured_layer_count is not None else indexed_layer_count
    split_size = estimated_split_size_bytes(checkpoint_size, compression)
    average_stream_unit = split_size // layer_count if layer_count else None

    warnings = []
    required_packages = []
    quantization_config = config.get("quantization_config") or text_config.get("quantization_config")
    if compression is not None and quantization_config is not None:
        raise ValueError("AirLLM compression cannot be stacked on a pre-quantized checkpoint.")
    if isinstance(quantization_config, dict) and quantization_config.get("quant_method") == "compressed-tensors":
        required_packages.append("compressed-tensors>=0.15.0")
    if architecture == "KimiK25ForConditionalGeneration":
        required_packages.append("transformers>=4.57.1,<5.0.0")
    if architecture == "GlmMoeDsaForCausalLM":
        required_packages.append("transformers>=5.12,<5.13")
        if isinstance(quantization_config, dict) and quantization_config.get("quant_method") == "fp8":
            warnings.append(
                "The portable FP8 path dequantizes one projection at a time; it is compatible but slower than optimized kernels."
            )
    if compatibility == "text-only":
        warnings.append("Kimi K2.5/K2.6/K2.7 support is text-only; vision inputs are rejected.")
    elif compatibility == "unknown":
        warnings.append("No supported decoder-layer layout was found in the checkpoint index.")
    if not checkpoint_size:
        warnings.append("Checkpoint size is unavailable; disk and stream-unit estimates are incomplete.")

    offload_free = None
    if offload_dir is not None:
        offload_path = Path(offload_dir)
        capacity_path = offload_path
        while not capacity_path.exists() and capacity_path != capacity_path.parent:
            capacity_path = capacity_path.parent
        if not capacity_path.exists():
            raise FileNotFoundError(f"Offload directory does not exist: {offload_path}")
        offload_free = shutil.disk_usage(capacity_path).free
        if split_size > offload_free:
            warnings.append(
                f"Offload volume is too small: {split_size} bytes required, {offload_free} bytes free."
            )

    return ModelPreflightReport(
        model=model,
        architecture=architecture,
        transformers_version=transformers_version,
        compatibility=compatibility,
        layer_layout=layout,
        layer_count=layer_count,
        tensor_count=len(weight_names),
        shard_count=shard_count,
        checkpoint_size_bytes=checkpoint_size,
        estimated_split_size_bytes=split_size,
        average_stream_unit_bytes=average_stream_unit,
        offload_free_bytes=offload_free,
        required_packages=required_packages,
        warnings=warnings,
    )


def _gib(value):
    return "unknown" if value is None else f"{value / (1024 ** 3):.2f} GiB"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face repo ID or local checkpoint path")
    parser.add_argument("--compression", choices=("4bit", "8bit"))
    parser.add_argument("--cache-dir")
    parser.add_argument("--offload-dir")
    parser.add_argument("--hf-token")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = inspect_model(
        args.model,
        compression=args.compression,
        hf_token=args.hf_token,
        cache_dir=args.cache_dir,
        offload_dir=args.offload_dir,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return

    print(f"model: {report.model}")
    print(f"architecture: {report.architecture}")
    print(f"compatibility: {report.compatibility}")
    print(f"layers: {report.layer_count}")
    print(f"checkpoint: {_gib(report.checkpoint_size_bytes)}")
    print(f"estimated split: {_gib(report.estimated_split_size_bytes)}")
    print(f"checkpoint/layer size proxy: {_gib(report.average_stream_unit_bytes)}")
    if report.offload_free_bytes is not None:
        print(f"offload free: {_gib(report.offload_free_bytes)}")
    if report.required_packages:
        print(f"required packages: {', '.join(report.required_packages)}")
    for warning in report.warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
