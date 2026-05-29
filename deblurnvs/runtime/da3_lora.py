from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_LORA_STATE_TOKENS = ("lora_A", "lora_B", "lora_down", "lora_up")


class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.linear = linear_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)

        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(self.rank, linear_layer.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(linear_layer.out_features, self.rank))
        self.lora_dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        x_dropout = self.lora_dropout(x)
        lora_A = self.lora_A.to(device=x_dropout.device, dtype=x_dropout.dtype)
        lora_B = self.lora_B.to(device=x_dropout.device, dtype=x_dropout.dtype)
        return out + (F.linear(x_dropout, lora_A) @ lora_B.t()) * self.scaling


class LoRAConv2d(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.conv = conv_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)

        for param in self.conv.parameters():
            param.requires_grad = False

        kernel_h, kernel_w = (
            conv_layer.kernel_size if isinstance(conv_layer.kernel_size, tuple) else (conv_layer.kernel_size, conv_layer.kernel_size)
        )
        weight_dim = conv_layer.in_channels * kernel_h * kernel_w
        self.lora_A = nn.Parameter(torch.randn(self.rank, weight_dim) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(conv_layer.out_channels, self.rank))
        self.lora_dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        lora_A = self.lora_A.to(device=x.device, dtype=x.dtype)
        lora_B = self.lora_B.to(device=x.device, dtype=x.dtype)
        kernel_h, kernel_w = (
            self.conv.kernel_size if isinstance(self.conv.kernel_size, tuple) else (self.conv.kernel_size, self.conv.kernel_size)
        )
        lora_weight = (lora_B @ lora_A).view(
            self.conv.out_channels,
            self.conv.in_channels,
            kernel_h,
            kernel_w,
        )
        lora_out = F.conv2d(
            self.lora_dropout(x),
            lora_weight * self.scaling,
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        return out + lora_out


def apply_lora_to_module(
    module: nn.Module,
    target_modules: list[str] | None = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    verbose: bool = False,
) -> nn.Module:
    target_modules = [] if target_modules is None else list(target_modules)
    for name, child in list(module.named_children()):
        should_apply = len(target_modules) == 0 or any(target in name.lower() for target in target_modules)
        if isinstance(child, nn.Linear) and should_apply:
            if verbose:
                print(f"[DeblurNVS] apply LoRA to Linear: {name}")
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        elif isinstance(child, nn.Conv2d) and should_apply:
            if verbose:
                print(f"[DeblurNVS] apply LoRA to Conv2d: {name}")
            setattr(module, name, LoRAConv2d(child, rank=rank, alpha=alpha, dropout=dropout))
        else:
            apply_lora_to_module(child, target_modules, rank, alpha, dropout, verbose)
    return module


def get_lora_parameters(module: nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for name, param in module.named_parameters():
        if "lora_A" in name or "lora_B" in name or "lora_down" in name or "lora_up" in name:
            params.append(param)
    return params


def _is_lora_state_key(name: str) -> bool:
    return any(token in name for token in _LORA_STATE_TOKENS)


def configure_rae_da3_lora(rae: torch.nn.Module, lora_cfg: Any | None) -> dict[str, Any]:
    enabled = bool(lora_cfg is not None and bool(lora_cfg.get("enabled", False)))
    pretrained_backbone = rae.encoder.backbone.pretrained
    if not enabled:
        return {
            "enabled": False,
            "module": pretrained_backbone,
            "trainable_params": [],
            "wrapped_modules": [],
            "target_modules": [],
            "rank": 0,
            "alpha": 0.0,
            "dropout": 0.0,
        }

    target_modules = lora_cfg.get("target_modules")
    if target_modules is None or len(target_modules) == 0:
        target_modules = ["qkv", "proj", "fc1", "fc2"]
    target_modules = [str(name) for name in target_modules]
    exclude_name_contains = lora_cfg.get("exclude_name_contains")
    if exclude_name_contains is None:
        exclude_name_contains = ["patch_embed"]
    exclude_name_contains = [str(pattern) for pattern in exclude_name_contains]

    rank = int(lora_cfg.get("rank", 8))
    alpha = float(lora_cfg.get("alpha", 16.0))
    dropout = float(lora_cfg.get("dropout", 0.0))
    verbose = bool(lora_cfg.get("verbose", False))

    apply_lora_to_module(
        pretrained_backbone,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        verbose=verbose,
    )
    if exclude_name_contains:
        for name, param in pretrained_backbone.named_parameters():
            if any(pattern in name for pattern in exclude_name_contains):
                param.requires_grad = False

    wrapped_modules = [
        name
        for name, module in pretrained_backbone.named_modules()
        if isinstance(module, (LoRALinear, LoRAConv2d))
    ]
    trainable_params = [param for param in get_lora_parameters(pretrained_backbone) if param.requires_grad]
    return {
        "enabled": True,
        "module": pretrained_backbone,
        "trainable_params": trainable_params,
        "wrapped_modules": wrapped_modules,
        "target_modules": target_modules,
        "exclude_name_contains": exclude_name_contains,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
    }


def extract_lora_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = module.state_dict()
    return {name: tensor.detach().cpu() for name, tensor in state.items() if _is_lora_state_key(name)}


def load_lora_state_dict(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor] | None,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    if not state_dict:
        return [], []

    current_state = module.state_dict()
    current_lora_keys = {name for name in current_state.keys() if _is_lora_state_key(name)}
    filtered_state = {name: tensor for name, tensor in state_dict.items() if name in current_lora_keys}
    missing = sorted(name for name in current_lora_keys if name not in filtered_state)
    unexpected = sorted(name for name in state_dict.keys() if name not in current_lora_keys)

    if filtered_state:
        module.load_state_dict(filtered_state, strict=False)

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Failed to strictly load LoRA state: missing={len(missing)} unexpected={len(unexpected)}"
        )
    return missing, unexpected


def count_parameters(params: list[torch.nn.Parameter]) -> int:
    return int(sum(param.numel() for param in params))
