from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dinov2_model import DinoV2EReCuModel
from .utils import dice_loss, gradient_magnitude, masked_bce, safe_logit


@torch.no_grad()
def update_ema(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    for teacher_value, student_value in zip(teacher.parameters(), student.parameters()):
        teacher_value.mul_(momentum).add_(student_value, alpha=1.0 - momentum)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)


class DinoV2Trainer:
    def __init__(self, model: DinoV2EReCuModel, config: dict, device: torch.device, output_dir: str | Path) -> None:
        self.model = model.to(device)
        self.teacher = copy.deepcopy(model).to(device)
        self.teacher.layers = tuple(config["model"]["teacher_layers"])
        self.teacher.eval()
        for value in self.teacher.parameters():
            value.requires_grad_(False)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainable_heads = [p for name, p in self.model.named_parameters() if not name.startswith("backbone.") and p.requires_grad]
        optimizer_cfg = config["optimizer"]
        parameter_groups, self.backbone_group_indices = self._backbone_parameter_groups(optimizer_cfg)
        self.head_group_index = len(parameter_groups)
        parameter_groups.append({"params": trainable_heads, "lr": optimizer_cfg["lr"]})
        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=optimizer_cfg["weight_decay"],
        )
        scheduler_cfg = config.get("scheduler", {"type": "cosine"})
        self.scheduler_step_unit = scheduler_cfg.get("step_unit", "epoch")
        warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
        # Pretrained DINOv2 and newly initialized prediction heads can use
        # different warmup factors.  This permits a fast head adaptation
        # without applying a destructive first-epoch update to DINO features.
        backbone_warmup_only = bool(scheduler_cfg.get("backbone_warmup_only", False))
        if warmup_epochs < 0 or warmup_epochs >= config["training"]["epochs"]:
            raise ValueError("scheduler.warmup_epochs must be in [0, training.epochs)")
        if warmup_epochs and self.scheduler_step_unit != "epoch":
            raise ValueError("scheduler warmup currently requires step_unit: epoch")
        scheduler_type = scheduler_cfg.get("type", "cosine").lower()
        if scheduler_type == "steplr":
            if warmup_epochs:
                raise ValueError("scheduler warmup is not supported with StepLR")
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(scheduler_cfg["step_size"]),
                gamma=float(scheduler_cfg["gamma"]),
            )
        elif scheduler_type == "linear_warmup":
            start_factor = float(scheduler_cfg.get("warmup_start_factor", 0.1))
            if not 0.0 < start_factor <= 1.0:
                raise ValueError("scheduler.warmup_start_factor must be in (0, 1]")
            head_start_factor = float(
                scheduler_cfg.get("head_warmup_start_factor", 1.0 if backbone_warmup_only else start_factor)
            )
            if not 0.0 < head_start_factor <= 1.0:
                raise ValueError("scheduler.head_warmup_start_factor must be in (0, 1]")

            def warmup_factor(step: int, group_start_factor: float) -> float:
                progress = min(max(step, 0), warmup_epochs) / max(1, warmup_epochs)
                return group_start_factor + (1.0 - group_start_factor) * progress

            # A single LambdaLR avoids SequentialLR's deprecated transition
            # call while retaining the exact 25-epoch linear ramp.
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=[
                    lambda step, group_start_factor=(
                        start_factor if index in self.backbone_group_indices else head_start_factor
                    ): warmup_factor(step, group_start_factor)
                    for index in range(len(self.optimizer.param_groups))
                ],
            )
        elif scheduler_type == "cosine":
            start_factor = float(scheduler_cfg.get("warmup_start_factor", 0.1))
            if not 0.0 < start_factor <= 1.0:
                raise ValueError("scheduler.warmup_start_factor must be in (0, 1]")
            head_start_factor = float(
                scheduler_cfg.get("head_warmup_start_factor", 1.0 if backbone_warmup_only else start_factor)
            )
            if not 0.0 < head_start_factor <= 1.0:
                raise ValueError("scheduler.head_warmup_start_factor must be in (0, 1]")
            cosine_t_max = int(scheduler_cfg.get("cosine_t_max_epochs", config["training"]["epochs"] - warmup_epochs))
            if cosine_t_max < 1:
                raise ValueError("scheduler.cosine_t_max_epochs must be positive")
            eta_min = float(optimizer_cfg["min_lr"])
            base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]

            def cosine_factor(step: int, base_lr: float, group_start_factor: float) -> float:
                if step <= warmup_epochs:
                    if warmup_epochs == 0:
                        return 1.0
                    progress = min(max(step, 0), warmup_epochs) / max(1, warmup_epochs)
                    return group_start_factor + (1.0 - group_start_factor) * progress
                progress = min(1.0, (step - warmup_epochs) / cosine_t_max)
                end_factor = min(1.0, eta_min / base_lr)
                return end_factor + (1.0 - end_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

            # LambdaLR keeps warmup and cosine annealing in a single scheduler,
            # avoiding SequentialLR's deprecated transition warning.
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=[
                    lambda step, base_lr=base_lr, group_start_factor=(
                        start_factor if index in self.backbone_group_indices else head_start_factor
                    ): cosine_factor(
                        step, base_lr, group_start_factor
                    )
                    for index, base_lr in enumerate(base_lrs)
                ],
            )
        else:
            raise ValueError(f"Unsupported scheduler.type: {scheduler_type}")
        training_cfg = config["training"]
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=training_cfg["amp"] and device.type == "cuda",
            # DINOv2/STAF gradients overflow at PyTorch's default 65536
            # during the first updates. Start from the empirically stable
            # scale instead of silently spending early batches calibrating.
            init_scale=float(training_cfg.get("amp_init_scale", 2048.0)),
            # Keep this scale throughout a 25-epoch run. Re-growing to 4096+
            # can reintroduce the same transient fp16 overflow later on.
            growth_interval=int(training_cfg.get("amp_growth_interval", 1_000_000)),
        )
        self.global_step = 0
        self.skipped_optimizer_steps = 0
        self.consecutive_skipped_updates = 0
        self.start_epoch = 0
        self.best_train_loss = float("inf")
        self.best_epoch = 0
        self.log_path = self.output_dir / "train_metrics.csv"
        self.metric_fields = [
            "epoch", "loss", "global", "epl", "pool", "lpr", "mnp",
            "boundary", "agreement", "area", "mnp_score", "lr",
            "backbone_lr", "optimizer_updates", "skipped_updates",
        ]
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.metric_fields).writeheader()

    def resume(self, checkpoint: str | Path) -> None:
        """Restore a full training checkpoint, including optimizer and EMA teacher."""
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        required = {"epoch", "student", "teacher", "optimizer", "scheduler"}
        missing = required - set(state)
        if missing:
            raise ValueError(f"{checkpoint} is not a resumable training checkpoint; missing {sorted(missing)}")
        self.model.load_state_dict(state["student"], strict=True)
        self.teacher.load_state_dict(state["teacher"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state.get("global_step", 0))
        self.skipped_optimizer_steps = int(state.get("skipped_optimizer_steps", 0))
        self.consecutive_skipped_updates = int(state.get("consecutive_skipped_updates", 0))
        self.start_epoch = int(state["epoch"]) + 1
        self.best_train_loss = float(state.get("best_train_loss", float("inf")))
        self.best_epoch = int(state.get("best_epoch", 0))
        if self.start_epoch >= self.config["training"]["epochs"]:
            raise ValueError(
                f"Checkpoint already completed epoch {state['epoch'] + 1}; "
                f"configured epochs={self.config['training']['epochs']} leaves nothing to train"
            )
        print(f"Resumed from {checkpoint}: next epoch={self.start_epoch + 1}, global_step={self.global_step}")

    def _backbone_parameter_groups(self, optimizer_cfg: dict) -> tuple[list[dict], list[int]]:
        """Use layer-wise learning-rate decay for trainable DINOv2 parameters."""
        backbone_lr = float(optimizer_cfg["backbone_lr"])
        layer_decay = float(optimizer_cfg.get("layer_decay", 1.0))
        if not 0.0 < layer_decay <= 1.0:
            raise ValueError("optimizer.layer_decay must be in (0, 1]")
        depth = int(self.model.backbone.model.config.num_hidden_layers)
        buckets: dict[int, list[torch.nn.Parameter]] = {}
        for name, parameter in self.model.backbone.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("model.encoder.layer."):
                layer = int(name.split(".")[3]) + 1
            elif name.startswith("model.embeddings."):
                layer = 0
            else:
                # Final layer norm/pooler are attached after the transformer.
                layer = depth
            buckets.setdefault(layer, []).append(parameter)
        groups: list[dict] = []
        indices: list[int] = []
        for layer in sorted(buckets):
            lr = backbone_lr * (layer_decay ** (depth - layer))
            indices.append(len(groups))
            groups.append({"params": buckets[layer], "lr": lr})
        return groups, indices

    def _schedule_progress(self, schedule: dict, epoch: int) -> float:
        """Return a clamped curriculum progress without consulting evaluation data."""
        total = int(self.config["training"]["epochs"])
        start = int(schedule.get("schedule_start_epoch", 1)) - 1
        end = int(schedule.get("schedule_end_epoch", total)) - 1
        if end <= start:
            return 1.0 if epoch >= end else 0.0
        return min(1.0, max(0.0, (epoch - start) / (end - start)))

    def _ema_momentum(self, epoch: int) -> float:
        teacher_cfg = self.config["teacher"]
        start, end = teacher_cfg["momentum"], teacher_cfg.get("momentum_end", teacher_cfg["momentum"])
        progress = self._schedule_progress(teacher_cfg, epoch)
        return start + (end - start) * progress

    def _seed_mix(self, epoch: int) -> float:
        schedule = self.config["pseudo"]
        start, end = schedule["seed_mix_start"], schedule["seed_mix_end"]
        progress = self._schedule_progress(schedule, epoch)
        return start + (end - start) * progress

    def _sharpen_temperature(self, epoch: int) -> float:
        schedule = self.config["pseudo"]
        start = schedule.get("temperature_start", 1.0)
        end = schedule.get("temperature_end", start)
        progress = self._schedule_progress(schedule, epoch)
        return start + (end - start) * progress

    @staticmethod
    def _probability_mean(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        logits = safe_logit(first) + safe_logit(second)
        return torch.sigmoid(logits * 0.5)

    def _loss(self, student, teacher, epoch: int, teacher_flip=None) -> tuple[torch.Tensor, dict[str, float]]:
        loss_cfg = self.config["loss"]
        seed_mix = self._seed_mix(epoch)
        first_target = seed_mix * teacher.attention_seed + (1.0 - seed_mix) * teacher.probability_grid
        if teacher_flip is None:
            attention_consensus = teacher.attention_seed
            target = first_target
            agreement = torch.ones_like(target)
        else:
            flip_attention = torch.flip(teacher_flip.attention_seed, dims=[-1])
            flip_probability = torch.flip(teacher_flip.probability_grid, dims=[-1])
            second_target = seed_mix * flip_attention + (1.0 - seed_mix) * flip_probability
            attention_consensus = self._probability_mean(teacher.attention_seed, flip_attention)
            target = self._probability_mean(first_target, second_target)
            agreement = (1.0 - (first_target - second_target).abs()).clamp(0.0, 1.0)
        target = target.detach()
        temperature = self._sharpen_temperature(epoch)
        target = torch.sigmoid(safe_logit(target) / max(temperature, 1e-3))
        confidence_floor = self.config["pseudo"].get("confidence_floor", 0.15)
        confidence_gamma = self.config["pseudo"].get("confidence_gamma", 1.5)
        consensus_gamma = self.config["pseudo"].get("consensus_gamma", 1.0)
        confidence = (
            confidence_floor
            + (1.0 - confidence_floor) * ((target - 0.5).abs() * 2.0).pow(confidence_gamma)
        )
        confidence = (confidence * agreement.pow(consensus_gamma)).detach()
        global_loss = dice_loss(student.probability_grid, target, confidence) + masked_bce(student.probability_grid, target, confidence)

        epl = student.probability_grid.new_zeros(())
        pool_loss = student.probability_grid.new_zeros(())
        for index, (dsc, own_pool, teacher_pool) in enumerate(
            zip(student.dsc_probabilities, student.pool_probabilities, teacher.pool_probabilities)
        ):
            if teacher_flip is not None:
                flip_pool = torch.flip(teacher_flip.pool_probabilities[index], dims=[-1])
                teacher_pool = self._probability_mean(teacher_pool, flip_pool)
            layer_target = (seed_mix * attention_consensus + (1.0 - seed_mix) * teacher_pool).detach()
            epl = epl + dice_loss(dsc, layer_target) + 0.5 * dice_loss(dsc, own_pool.detach())

            pool_loss = pool_loss + dice_loss(own_pool, target, confidence) + 0.5 * masked_bce(own_pool, target, confidence)
        epl = epl / len(student.dsc_probabilities)
        pool_loss = pool_loss / len(student.pool_probabilities)
        epl = epl + loss_cfg.get("pool", 0.25) * pool_loss

        local = teacher.local_pseudo
        if local is not None and teacher_flip is not None and teacher_flip.local_pseudo is not None:

            local = torch.maximum(local, torch.flip(teacher_flip.local_pseudo, dims=[-1]))
        if local is None:
            lpr = student.probability_grid.new_zeros(())
        else:
            lpr = dice_loss(student.probability_grid, local) + masked_bce(student.probability_grid, local)

        layer_mnp = torch.stack(
            [1.0 - self.model.mnp.score(dsc, student.mnp_features, soft=True).mean() for dsc in student.dsc_probabilities]
        ).mean()
        mnp = 0.5 * student.mnp_loss + 0.5 * layer_mnp
        student_boundary = gradient_magnitude(student.probability_grid)
        target_boundary = gradient_magnitude(target).detach()
        boundary_weight = (0.25 + 0.75 * target_boundary).detach()
        boundary = dice_loss(student_boundary, target_boundary) + (
            (student_boundary - target_boundary).abs() * boundary_weight
        ).sum() / boundary_weight.sum().clamp_min(1.0)
        total = (
            loss_cfg["global"] * global_loss
            + loss_cfg["epl"] * epl
            + loss_cfg["lpr"] * lpr
            + loss_cfg["mnp"] * mnp
            + loss_cfg.get("boundary", 0.0) * boundary
        )
        values = {
            "loss": float(total.detach().item()),
            "global": float(global_loss.detach().item()),
            "epl": float(epl.detach().item()),
            "pool": float(pool_loss.detach().item()),
            "lpr": float(lpr.detach().item()),
            "mnp": float(mnp.detach().item()),
            "boundary": float(boundary.detach().item()),
            "agreement": float(agreement.detach().mean().item()),
            "area": float(student.probability_grid.detach().mean().item()),
            "mnp_score": float(student.mnp_score.detach().mean().item()),
        }
        return total, values

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        if self.model.backbone.frozen:
            self.model.backbone.eval()
        self.model.mnp.semantic.eval()
        totals: dict[str, float] = {}
        successful_updates = 0
        skipped_updates = 0
        accumulation = self.config["training"].get("grad_accumulation", 1)
        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
        for step, batch in enumerate(progress, start=1):
            group_start = ((step - 1) // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            native = batch["native"].to(self.device, non_blocking=True)
            student_image = batch["student"].to(self.device, non_blocking=True)
            teacher_image = batch["teacher"].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                student = self.model(student_image, native, make_local_pseudo=False, quality_seed=False)
                with torch.no_grad():
                    teacher = self.teacher(
                        teacher_image,
                        native,
                        make_local_pseudo=True,
                        mnp_features=student.mnp_features.detach(),
                        quality_seed=True,
                    )
                    teacher_flip = None
                    if self.config["training"].get("teacher_flip_consistency", True):
                        teacher_flip = self.teacher(
                            torch.flip(teacher_image, dims=[-1]),
                            torch.flip(native, dims=[-1]),
                            make_local_pseudo=True,
                            mnp_features=torch.flip(student.mnp_features.detach(), dims=[-1]),
                            quality_seed=True,
                        )
                loss, values = self._loss(student, teacher, epoch, teacher_flip)
                # The last DataLoader group can be incomplete because
                # drop_last is disabled. Divide by its actual micro-batch
                # count so every image keeps the same contribution as in a
                # complete accumulation group.
                loss = loss / group_size
            self.scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"].get("grad_norm", 1.0))
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                # GradScaler lowers its scale when it skips optimizer.step()
                # after detecting non-finite gradients. EMA/global_step must
                # advance only after a real parameter update; otherwise a
                # full run can silently save the untouched initialization.
                did_update = (not self.scaler.is_enabled()) or self.scaler.get_scale() >= scale_before
                if did_update:
                    update_ema(self.teacher, self.model, self._ema_momentum(epoch))
                    self.global_step += 1
                    successful_updates += 1
                    self.consecutive_skipped_updates = 0
                    if self.scheduler_step_unit == "optimizer_update":
                        self.scheduler.step()
                else:
                    skipped_updates += 1
                    self.skipped_optimizer_steps += 1
                    self.consecutive_skipped_updates += 1
                    print(
                        "AMP SKIPPED OPTIMIZER UPDATE: "
                        f"epoch={epoch + 1}, batch={step}, scale={scale_before:g}->{self.scaler.get_scale():g}, "
                        f"consecutive={self.consecutive_skipped_updates}",
                        flush=True,
                    )
                    max_skips = int(self.config["training"].get("max_consecutive_skipped_updates", 8))
                    if self.consecutive_skipped_updates >= max_skips:
                        raise FloatingPointError(
                            f"Aborting after {self.consecutive_skipped_updates} consecutive AMP-skipped optimizer updates. "
                            "Use amp: false for a diagnostic run, or resolve the non-finite gradient source."
                        )
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
            # Display a stable running mean rather than the noisy loss of the
            # most recent micro-batch. `values["loss"]` is the unscaled loss,
            # so this matches the epoch loss written to the CSV at completion.
            running_loss = totals["loss"] / step
            progress.set_postfix(loss=f"{running_loss:.3f}", area=f"{values['area']:.3f}", mnp=f"{values['mnp_score']:.3f}")
        result = {key: value / len(loader) for key, value in totals.items()}
        result["lr"] = self.optimizer.param_groups[self.head_group_index]["lr"]
        result["backbone_lr"] = max(
            (self.optimizer.param_groups[index]["lr"] for index in self.backbone_group_indices),
            default=0.0,
        )
        result["optimizer_updates"] = successful_updates
        result["skipped_updates"] = skipped_updates
        if self.scheduler_step_unit == "epoch":
            self.scheduler.step()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            row = {"epoch": epoch + 1, **result}
            csv.DictWriter(handle, fieldnames=self.metric_fields).writerow(row)
        return result

    def _checkpoint_state(self, epoch: int) -> dict:
        teacher_state = self.teacher.state_dict()
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "skipped_optimizer_steps": self.skipped_optimizer_steps,
            "consecutive_skipped_updates": self.consecutive_skipped_updates,
            "student": self.model.state_dict(),
            "teacher": teacher_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_train_loss": self.best_train_loss,
            "best_epoch": self.best_epoch,
            "config": self.config,
        }

    def save_last_checkpoint(self, epoch: int) -> Path:
        """Overwrite the single resumable final checkpoint after each epoch."""
        target = self.output_dir / "checkpoint_last.pth"
        torch.save(self._checkpoint_state(epoch), target)
        return target

    def save_best_checkpoint(self, epoch: int, metadata: dict) -> None:
        """Save the training-loss best full checkpoint and its metadata."""
        target = self.output_dir / "checkpoint_best.pth"
        torch.save(self._checkpoint_state(epoch), target)
        (self.output_dir / "checkpoint_best.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def fit(self, loader: DataLoader, max_epochs: int | None = None) -> None:
        """Train a bounded number of epochs without resetting the global schedule."""
        total_epochs = int(self.config["training"]["epochs"])
        if max_epochs is not None and max_epochs < 1:
            raise ValueError("max_epochs must be >= 1 when provided")
        end_epoch = total_epochs if max_epochs is None else min(total_epochs, self.start_epoch + max_epochs)
        stop_start = max(1, int(self.config["training"].get("early_stop_start_epoch", 1)))
        patience = int(self.config["training"].get("early_stop_patience", 0))
        min_delta = float(self.config["training"].get("early_stop_min_delta", 0.0))
        stale_epochs = 0
        for epoch in range(self.start_epoch, end_epoch):
            values = self.train_epoch(loader, epoch)
            print(json.dumps({"epoch": epoch + 1, **values}, ensure_ascii=False))
            current_epoch = epoch + 1
            is_best = values["loss"] < self.best_train_loss
            significant_improvement = values["loss"] < self.best_train_loss - min_delta
            if is_best:
                self.best_train_loss = values["loss"]
                self.best_epoch = current_epoch
            self.save_last_checkpoint(epoch)
            if is_best:
                self.save_best_checkpoint(
                    epoch,
                    {"criterion": "minimum_training_loss", "epoch": self.best_epoch, "train_loss": self.best_train_loss},
                )
                print(f"TRAIN-LOSS BEST CHECKPOINT: epoch={current_epoch}, loss={self.best_train_loss:.6f}")

            if patience > 0 and current_epoch >= stop_start:
                if significant_improvement:
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= patience:
                        print(
                            f"TRAIN-LOSS EARLY STOP: epoch={current_epoch}, "
                            f"best_loss={self.best_train_loss:.6f}, patience={patience}"
                        )
                        break
