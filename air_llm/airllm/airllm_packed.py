"""Projection-at-a-time execution for compressed-tensors packed weights."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AirLLMPackedLinear(nn.Module):
    """Keep INT packed checkpoint tensors and dequantize only during one linear call."""

    _STATE_NAMES = (
        "weight_packed",
        "weight_scale",
        "weight_shape",
        "weight_zero_point",
        "weight_g_idx",
        "input_global_scale",
        "bias",
    )

    def __init__(self, original):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.quantization_scheme = original.quantization_scheme

        for name in self._STATE_NAMES:
            if name in original._parameters and original._parameters[name] is not None:
                value = original._parameters[name]
                self.register_parameter(
                    name,
                    nn.Parameter(torch.empty_like(value), requires_grad=value.requires_grad),
                )
            elif name in original._buffers and original._buffers[name] is not None:
                self.register_buffer(name, torch.empty_like(original._buffers[name]))
            else:
                self.register_parameter(name, None)

    def _compressed_state(self):
        return {
            name: getattr(self, name)
            for name in self._STATE_NAMES
            if name != "bias" and getattr(self, name, None) is not None
        }

    def forward(self, inputs):
        try:
            from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
        except ImportError as exc:
            raise ImportError(
                "Kimi packed INT weights require compressed-tensors>=0.15.0."
            ) from exc

        state = PackedQuantizationCompressor.decompress(
            self._compressed_state(),
            self.quantization_scheme,
        )
        weight = state["weight"].to(dtype=inputs.dtype)
        bias = self.bias.to(dtype=inputs.dtype) if self.bias is not None else None
        return F.linear(inputs, weight, bias)


def replace_with_airllm_packed_linears(model):
    """Replace compressed Linear modules and remove whole-model decompression."""
    replacements = []
    for module_name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and hasattr(module, "weight_packed")
            and hasattr(module, "quantization_scheme")
        ):
            replacements.append((module_name, AirLLMPackedLinear(module)))

    for module_name, replacement in replacements:
        parent = model
        parts = module_name.split(".")
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        leaf = parts[-1]
        if leaf.isdigit():
            parent[int(leaf)] = replacement
        else:
            setattr(parent, leaf, replacement)

    if hasattr(model, "ct_decompress_hook"):
        model.ct_decompress_hook.remove()
        delattr(model, "ct_decompress_hook")

    if not replacements:
        raise ValueError("No compressed-tensors packed Linear modules were found.")
    return len(replacements)
