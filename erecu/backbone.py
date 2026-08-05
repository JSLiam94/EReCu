from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch import nn


@dataclass
class BackboneOutput:
    tokens: dict[int, torch.Tensor]
    attentions: dict[int, torch.Tensor]
    grid_size: tuple[int, int]


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        output = (attn @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(output), attn


class Block(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value, attn = self.attn(self.norm1(x))
        x = x + value
        x = x + self.mlp(self.norm2(x))
        return x, attn


class DinoViTSmall8(nn.Module):
    """DINO ViT-S/8 compatible with dino_deitsmall8_pretrain.pth."""

    def __init__(self, image_size: int = 224, patch_size: int = 8, embed_dim: int = 384, depth: int = 12, num_heads: int = 6) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Identity()
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _interpolate_position(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        reference = self.pos_embed.shape[1] - 1
        if grid_h * grid_w == reference and grid_h == grid_w:
            return self.pos_embed
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        source = int(math.sqrt(reference))
        patch_pos = patch_pos.reshape(1, source, source, self.embed_dim).permute(0, 3, 1, 2)
        patch_pos = nn.functional.interpolate(patch_pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, self.embed_dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, image: torch.Tensor, requested_layers: tuple[int, ...] = (4, 8, 12)) -> BackboneOutput:
        batch, _, height, width = image.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"Input {(height, width)} must be divisible by patch size {self.patch_size}")
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        x = self.patch_embed.proj(image).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(batch, -1, -1), x], dim=1)
        x = self.pos_drop(x + self._interpolate_position(x, grid_h, grid_w))
        tokens: dict[int, torch.Tensor] = {}
        attentions: dict[int, torch.Tensor] = {}
        target = set(requested_layers)
        for index, block in enumerate(self.blocks, start=1):
            x, attn = block(x)
            if index in target:
                tokens[index] = x[:, 1:].transpose(1, 2).reshape(batch, self.embed_dim, grid_h, grid_w)
                attentions[index] = attn[:, :, 0, 1:].reshape(batch, attn.shape[1], grid_h, grid_w)
        if set(requested_layers) - set(tokens):
            raise ValueError(f"Requested unavailable layers: {set(requested_layers) - set(tokens)}")
        return BackboneOutput(tokens=tokens, attentions=attentions, grid_size=(grid_h, grid_w))


def load_dino_weights(model: DinoViTSmall8, weight_path: str) -> dict[str, list[str]]:
    path = Path(weight_path)
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    for container in ("teacher", "student", "state_dict"):
        if isinstance(state, dict) and container in state and isinstance(state[container], dict):
            state = state[container]
    state = {key.replace("module.", "").replace("backbone.", ""): value for key, value in state.items()}
    # DINO ViT-S/8 is pretrained at 224px.  For higher-resolution training,
    # interpolate only the patch positional grid while preserving the CLS token.
    # All learned transformer/patch weights remain loaded exactly.
    if "pos_embed" in state and state["pos_embed"].shape != model.pos_embed.shape:
        source_position = state["pos_embed"]
        source_patches = source_position.shape[1] - 1
        target_patches = model.pos_embed.shape[1] - 1
        source_size = int(math.sqrt(source_patches))
        target_size = int(math.sqrt(target_patches))
        if source_size * source_size != source_patches or target_size * target_size != target_patches:
            raise RuntimeError(
                f"Cannot interpolate non-square position grids: source={source_patches}, target={target_patches}"
            )
        cls_position = source_position[:, :1]
        patch_position = source_position[:, 1:].reshape(1, source_size, source_size, model.embed_dim).permute(0, 3, 1, 2)
        patch_position = nn.functional.interpolate(
            patch_position,
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False,
        )
        patch_position = patch_position.permute(0, 2, 3, 1).reshape(1, target_patches, model.embed_dim)
        state["pos_embed"] = torch.cat([cls_position, patch_position], dim=1)
    result = model.load_state_dict(state, strict=False)
    allowed_missing = {"head.weight", "head.bias"}
    missing = [key for key in result.missing_keys if key not in allowed_missing]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"DINO weight mismatch. missing={missing}, unexpected={result.unexpected_keys}")
    return {"missing": missing, "unexpected": list(result.unexpected_keys)}
