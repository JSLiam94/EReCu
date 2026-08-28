from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class BackboneOutput:
    """Intermediate DINOv2 features in the layout consumed by EReCu."""

    tokens: dict[int, torch.Tensor]
    attentions: dict[int, torch.Tensor]
    grid_size: tuple[int, int]


class DinoV2Backbone(nn.Module):
    """DINOv2-Base backbone with intermediate attentions.

    ``model_name`` can be a local Hugging Face ``save_pretrained`` directory
    or the official Meta ``dinov2_vitb14_pretrain.pth`` checkpoint.  The
    latter is converted in memory into the equivalent Transformers layout, so
    training and inference remain fully offline.
    """

    def __init__(
        self,
        model_name: str,
        local_files_only: bool = False,
        frozen: bool = False,
        trainable_last_blocks: int | None = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import Dinov2Config, Dinov2Model
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "DINOv2 requires transformers. Run `python bootstrap_dependencies.py` "
                "after installing the CUDA-compatible PyTorch build."
            ) from error

        model_path = Path(model_name).expanduser()
        if model_path.is_file():
            if model_path.suffix.lower() != ".pth":
                raise ValueError(
                    f"Unsupported DINOv2 checkpoint file {model_path}. "
                    "Use the official dinov2_vitb14_pretrain.pth file."
                )
            config = Dinov2Config(
                image_size=518,
                patch_size=14,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                layerscale_value=1.0,
                layer_norm_eps=1e-6,
            )
            # EReCu uses class-to-patch attention maps. Explicit eager
            # attention prevents Transformers from selecting an implementation
            # that omits them.
            if hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager"
            self.model = Dinov2Model(config)
            self._load_official_vitb14_checkpoint(model_path)
        else:
            # EReCu uses class-to-patch attention maps. Explicit eager
            # attention prevents Transformers from choosing an implementation
            # that omits them.
            try:
                self.model = Dinov2Model.from_pretrained(
                    model_name,
                    local_files_only=local_files_only,
                    attn_implementation="eager",
                )
            except TypeError:  # Compatible with older Transformers releases.
                self.model = Dinov2Model.from_pretrained(model_name, local_files_only=local_files_only)
                if hasattr(self.model.config, "_attn_implementation"):
                    self.model.config._attn_implementation = "eager"

        self.embed_dim = int(self.model.config.hidden_size)
        self.num_heads = int(self.model.config.num_attention_heads)
        self.patch_size = int(self.model.config.patch_size)
        self.frozen = frozen
        self.trainable_last_blocks = trainable_last_blocks
        if self.frozen:
            self.eval()
            for parameter in self.parameters():
                parameter.requires_grad_(False)
        elif trainable_last_blocks is not None:
            depth = int(self.model.config.num_hidden_layers)
            if not 1 <= trainable_last_blocks <= depth:
                raise ValueError(
                    f"trainable_last_blocks must be in 1..{depth}, got {trainable_last_blocks}"
                )
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for block in self.model.encoder.layer[depth - trainable_last_blocks :]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            # The final normalization belongs to the adapted representation.
            for parameter in self.model.layernorm.parameters():
                parameter.requires_grad_(True)

    def _load_official_vitb14_checkpoint(self, checkpoint_path: Path) -> None:
        """Load Meta's fused-QKV ViT-B/14 checkpoint into ``Dinov2Model``.

        Meta checkpoints use ``blocks.N.attn.qkv`` whereas Transformers stores
        independent query/key/value projections.  All other names differ only
        by a deterministic prefix.  Strict loading makes a wrong checkpoint or
        an incomplete conversion fail before training starts.
        """
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0 has no weights_only argument.
            state = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"DINOv2 checkpoint must contain a state dict: {checkpoint_path}")
        for key in ("state_dict", "teacher", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
        state = {str(key).removeprefix("module."): value for key, value in state.items()}

        def source(name: str) -> torch.Tensor:
            value = state.get(name)
            if not isinstance(value, torch.Tensor):
                raise KeyError(f"Missing tensor {name!r} in official DINOv2 checkpoint {checkpoint_path}")
            return value

        converted: dict[str, torch.Tensor] = {
            "embeddings.cls_token": source("cls_token"),
            "embeddings.mask_token": source("mask_token"),
            "embeddings.position_embeddings": source("pos_embed"),
            "embeddings.patch_embeddings.projection.weight": source("patch_embed.proj.weight"),
            "embeddings.patch_embeddings.projection.bias": source("patch_embed.proj.bias"),
            "layernorm.weight": source("norm.weight"),
            "layernorm.bias": source("norm.bias"),
        }
        depth = int(self.model.config.num_hidden_layers)
        for block_index in range(depth):
            source_prefix = f"blocks.{block_index}"
            target_prefix = f"encoder.layer.{block_index}"
            q_weight, k_weight, v_weight = source(f"{source_prefix}.attn.qkv.weight").chunk(3, dim=0)
            q_bias, k_bias, v_bias = source(f"{source_prefix}.attn.qkv.bias").chunk(3, dim=0)
            converted.update(
                {
                    f"{target_prefix}.attention.attention.query.weight": q_weight,
                    f"{target_prefix}.attention.attention.key.weight": k_weight,
                    f"{target_prefix}.attention.attention.value.weight": v_weight,
                    f"{target_prefix}.attention.attention.query.bias": q_bias,
                    f"{target_prefix}.attention.attention.key.bias": k_bias,
                    f"{target_prefix}.attention.attention.value.bias": v_bias,
                    f"{target_prefix}.attention.output.dense.weight": source(f"{source_prefix}.attn.proj.weight"),
                    f"{target_prefix}.attention.output.dense.bias": source(f"{source_prefix}.attn.proj.bias"),
                    f"{target_prefix}.layer_scale1.lambda1": source(f"{source_prefix}.ls1.gamma"),
                    f"{target_prefix}.layer_scale2.lambda1": source(f"{source_prefix}.ls2.gamma"),
                    f"{target_prefix}.norm1.weight": source(f"{source_prefix}.norm1.weight"),
                    f"{target_prefix}.norm1.bias": source(f"{source_prefix}.norm1.bias"),
                    f"{target_prefix}.norm2.weight": source(f"{source_prefix}.norm2.weight"),
                    f"{target_prefix}.norm2.bias": source(f"{source_prefix}.norm2.bias"),
                    f"{target_prefix}.mlp.fc1.weight": source(f"{source_prefix}.mlp.fc1.weight"),
                    f"{target_prefix}.mlp.fc1.bias": source(f"{source_prefix}.mlp.fc1.bias"),
                    f"{target_prefix}.mlp.fc2.weight": source(f"{source_prefix}.mlp.fc2.weight"),
                    f"{target_prefix}.mlp.fc2.bias": source(f"{source_prefix}.mlp.fc2.bias"),
                }
            )
        self.model.load_state_dict(converted, strict=True)

    def train(self, mode: bool = True) -> DinoV2Backbone:
        # A fully frozen extractor stays in evaluation mode. For partial
        # fine-tuning, keep embeddings and early blocks deterministic while
        # the selected final blocks follow the parent model's train/eval mode.
        super().train(False if self.frozen else mode)
        if not self.frozen and self.trainable_last_blocks is not None:
            depth = int(self.model.config.num_hidden_layers)
            self.model.embeddings.eval()
            for block in self.model.encoder.layer[: depth - self.trainable_last_blocks]:
                block.eval()
        return self

    def forward(self, image: torch.Tensor, requested_layers: tuple[int, ...]) -> BackboneOutput:
        _, _, height, width = image.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"DINOv2 input {(height, width)} must be divisible by patch size {self.patch_size}."
            )
        depth = int(self.model.config.num_hidden_layers)
        if not requested_layers or min(requested_layers) < 1 or max(requested_layers) > depth:
            raise ValueError(f"Requested layers {requested_layers} are outside DINOv2's 1..{depth}.")

        result = self.model(
            pixel_values=image,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        if result.attentions is None or result.hidden_states is None:
            raise RuntimeError("DINOv2 did not return attention maps; use eager attention in Transformers.")

        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        patch_count = grid_h * grid_w
        tokens: dict[int, torch.Tensor] = {}
        attentions: dict[int, torch.Tensor] = {}
        for layer in requested_layers:
            # hidden_states[0] is the embedding output. Selecting the final
            # patch_count items also supports DINOv2 variants with register tokens.
            hidden = result.hidden_states[layer][:, -patch_count:]
            tokens[layer] = hidden.transpose(1, 2).reshape(image.shape[0], self.embed_dim, grid_h, grid_w)
            attention = result.attentions[layer - 1]
            attentions[layer] = attention[:, :, 0, -patch_count:].reshape(
                image.shape[0], self.num_heads, grid_h, grid_w
            )
        return BackboneOutput(tokens=tokens, attentions=attentions, grid_size=(grid_h, grid_w))
