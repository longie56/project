import copy
import contextlib
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import *
from data.solomon_generator import SolomonGenerator
from model.pomo import PC_EGA_POMO


class Trainer:
    def __init__(
        self,
        model=None,
        variant_name="full",
        lr=LR,
        batch_size=BATCH_SIZE,
        n_nodes=N_NODES,
        n_pomo=N_POMO,
        steps_per_epoch=STEPS_PER_EPOCH,
        problem_type="R",
        tightness=0.7,
        save_dir=CHECKPOINT_DIR,
        val_set_size=VAL_SET_SIZE,
        val_batch_chunk=VAL_BATCH_CHUNK,
        scheduler_patience=LR_SCHEDULER_PATIENCE,
        scheduler_factor=LR_SCHEDULER_FACTOR,
        min_lr=MIN_LR,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_epochs=EARLY_STOPPING_MIN_EPOCHS,
        entropy_coef_start=ENTROPY_COEF,
        entropy_coef_end=ENTROPY_COEF_END,
        curriculum_start_tightness=CURRICULUM_START_TIGHTNESS,
        curriculum_epochs=CURRICULUM_EPOCHS,
        ema_decay=EMA_DECAY,
        use_ema_eval=USE_EMA_EVAL,
        normalize_advantage=NORMALIZE_ADVANTAGE,
        scenario="full",
        val_decode_type="greedy",
        val_sample_rollouts=0,
        val_multi_start_rollouts=0,
        val_augment_8_fold=False,
        elite_fraction=0.0,
        elite_loss_weight=0.0,
        bc_teacher_path=None,
        bc_batch_size=0,
        bc_weight_start=0.0,
        bc_weight_end=0.0,
        bc_epochs=0,
        pref_mode="random",
        pref_alpha=0.5,
        pref_beta=0.5,
        val_pref_mode=None,
        val_pref_alpha=None,
        val_pref_beta=None,
        val_problem_types=None,
        grad_accum_steps=1,
        pref_random_low=0.1,
        pref_random_high=0.9,
        amp=False,
        amp_dtype="bfloat16",
    ):
        self.model = (model if model is not None else PC_EGA_POMO()).to(DEVICE)
        self.use_ema_eval = bool(use_ema_eval)
        self.normalize_advantage = bool(normalize_advantage)
        self.ema_decay = float(ema_decay)
        self.ema_model = None
        if self.use_ema_eval:
            self.ema_model = copy.deepcopy(self.model).to(DEVICE)
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad = False
        self.variant_name = variant_name
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_lr,
        )
        self.generator = SolomonGenerator()
        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.n_pomo = n_nodes if n_pomo is None else n_pomo
        self.steps_per_epoch = steps_per_epoch
        self.problem_type = problem_type
        self.scenario = str(scenario).lower()
        self.val_decode_type = str(val_decode_type)
        self.val_sample_rollouts = int(val_sample_rollouts)
        self.val_multi_start_rollouts = int(val_multi_start_rollouts)
        self.val_augment_8_fold = bool(val_augment_8_fold)
        self.elite_fraction = float(elite_fraction)
        self.elite_loss_weight = float(elite_loss_weight)
        self.tightness = tightness
        self.curriculum_start_tightness = float(curriculum_start_tightness)
        self.curriculum_epochs = int(curriculum_epochs)
        self.entropy_coef_start = float(entropy_coef_start)
        self.entropy_coef_end = float(entropy_coef_end)
        self.save_dir = save_dir
        self.losses = []
        self.train_best_costs = []
        self.train_mean_costs = []
        self.val_costs = []
        self.val_scores = []
        self.val_set_size = val_set_size
        self.val_batch_chunk = val_batch_chunk
        self.pref_mode = str(pref_mode).lower()
        self.pref_alpha = float(pref_alpha)
        self.pref_beta = float(pref_beta)
        self.val_pref_mode = str(val_pref_mode).lower() if val_pref_mode is not None else self.pref_mode
        self.val_pref_alpha = float(val_pref_alpha) if val_pref_alpha is not None else self.pref_alpha
        self.val_pref_beta = float(val_pref_beta) if val_pref_beta is not None else self.pref_beta
        self.val_problem_types = self._parse_problem_types(val_problem_types)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.pref_random_low = float(pref_random_low)
        self.pref_random_high = float(pref_random_high)
        if not (0.0 <= self.pref_random_low < self.pref_random_high <= 1.0):
            raise ValueError(
                "Preference random range must satisfy 0 <= low < high <= 1; "
                f"got low={self.pref_random_low}, high={self.pref_random_high}."
            )
        self.use_amp = bool(amp and DEVICE.type == "cuda")
        amp_dtype = str(amp_dtype).lower()
        if amp_dtype in {"bf16", "bfloat16"}:
            self.amp_dtype = torch.bfloat16
            self.amp_dtype_name = "bfloat16"
        elif amp_dtype in {"fp16", "float16", "half"}:
            self.amp_dtype = torch.float16
            self.amp_dtype_name = "float16"
        else:
            raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_epochs = int(early_stopping_min_epochs)
        self.val_data = self._build_validation_sets() if val_set_size > 0 else None
        self.val_phys = []
        self.val_pen = []
        self.val_unvisit = []
        self.val_feasible = []
        self.val_by_type = []
        self.best_val_cost = float("inf")
        self.best_val_score = float("inf")
        self.best_epoch = 0
        self.last_entropy_coef = self.entropy_coef_start
        self.last_tightness = self.tightness
        self.bc_teacher = self._load_bc_teacher(bc_teacher_path)
        self.bc_batch_size = int(bc_batch_size)
        self.bc_weight_start = float(bc_weight_start)
        self.bc_weight_end = float(bc_weight_end)
        self.bc_epochs = int(bc_epochs)

    def _amp_context(self):
        if not self.use_amp:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _load_bc_teacher(self, path):
        if not path:
            return None
        if not os.path.exists(path):
            raise FileNotFoundError(f"BC teacher dataset not found: {path}")
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        required = ["nodes", "pref", "depot", "visited_mask", "teacher_actions"]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"BC teacher dataset missing keys: {missing}")
        size = int(data["nodes"].size(0))
        if size <= 0:
            raise ValueError("BC teacher dataset is empty.")
        print(f"[BC] loaded teacher dataset: {path} ({size} samples)")
        return {
            "nodes": data["nodes"].float(),
            "pref": data["pref"].float(),
            "depot": data["depot"].float(),
            "visited_mask": data["visited_mask"].bool(),
            "teacher_actions": data["teacher_actions"].long(),
        }

    def _current_bc_weight(self, epoch):
        if self.bc_teacher is None or self.bc_batch_size <= 0:
            return 0.0
        if self.bc_epochs <= 0 or epoch >= self.bc_epochs:
            return 0.0
        if self.bc_epochs <= 1:
            return self.bc_weight_start
        progress = min(float(epoch) / float(self.bc_epochs - 1), 1.0)
        return self.bc_weight_start + (self.bc_weight_end - self.bc_weight_start) * progress

    def _sample_bc_batch(self):
        if self.bc_teacher is None:
            return None
        size = int(self.bc_teacher["nodes"].size(0))
        batch_size = min(max(1, self.bc_batch_size), size)
        idx = torch.randint(0, size, (batch_size,))
        return {
            key: value[idx].to(DEVICE)
            for key, value in self.bc_teacher.items()
        }

    def _sample_pref(self, batch_size, mode=None, alpha=None, beta=None):
        mode = self.pref_mode if mode is None else str(mode).lower()
        alpha = self.pref_alpha if alpha is None else float(alpha)
        beta = self.pref_beta if beta is None else float(beta)
        if mode == "fixed":
            pref = torch.tensor(
                [[alpha, beta]],
                dtype=torch.float32,
                device=DEVICE,
            ).expand(batch_size, -1)
            return pref / pref.sum(dim=1, keepdim=True).clamp_min(1e-9)
        alpha_pref = (
            torch.rand(batch_size, 1, device=DEVICE)
            * (self.pref_random_high - self.pref_random_low)
            + self.pref_random_low
        )
        return torch.cat([alpha_pref, 1.0 - alpha_pref], dim=1)

    @staticmethod
    def _parse_problem_types(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            tokens = value
        else:
            tokens = str(value).split(",")
        return [str(token).strip().upper() for token in tokens if str(token).strip()]

    def _build_validation_set(self, problem_type):
        nodes, depot, visited = self.generator.generate(
            self.val_set_size, self.n_nodes, problem_type, self.tightness, DEVICE
        )
        pref = self._sample_pref(
            self.val_set_size,
            mode=self.val_pref_mode,
            alpha=self.val_pref_alpha,
            beta=self.val_pref_beta,
        )
        return {
            "nodes": nodes,
            "depot": depot,
            "visited_mask": visited,
            "pref": pref,
        }

    def _build_validation_sets(self):
        problem_types = self.val_problem_types or [self.problem_type]
        return {
            str(problem_type).upper(): self._build_validation_set(problem_type)
            for problem_type in problem_types
        }

    def _current_tightness(self, epoch, total_epochs):
        if total_epochs <= 1 or self.curriculum_epochs <= 1:
            return self.tightness
        progress = min(float(epoch) / float(max(1, self.curriculum_epochs - 1)), 1.0)
        start = self.curriculum_start_tightness
        target = self.tightness
        return start + (target - start) * progress

    def _current_entropy_coef(self, epoch, total_epochs):
        if total_epochs <= 1:
            return self.entropy_coef_end
        progress = min(float(epoch) / float(max(1, total_epochs - 1)), 1.0)
        return self.entropy_coef_start + (self.entropy_coef_end - self.entropy_coef_start) * progress

    def _update_ema(self):
        if not self.use_ema_eval or self.ema_model is None:
            return
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.mul_(self.ema_decay).add_(model_param.detach(), alpha=1.0 - self.ema_decay)
            for ema_buffer, model_buffer in zip(self.ema_model.buffers(), self.model.buffers()):
                ema_buffer.copy_(model_buffer)

    def validate(self):
        if self.val_data is None:
            return None

        eval_model = self.ema_model if self.use_ema_eval and self.ema_model is not None else self.model
        eval_model.eval()
        by_type = {}
        for problem_type, val_data in self.val_data.items():
            by_type[problem_type] = self._validate_one_set(eval_model, val_data)

        val_cost = sum(metrics["cost"] for metrics in by_type.values()) / len(by_type)
        val_phys = sum(metrics["phys"] for metrics in by_type.values()) / len(by_type)
        val_pen = sum(metrics["pen"] for metrics in by_type.values()) / len(by_type)
        val_unvisit = sum(metrics["unvisit"] for metrics in by_type.values()) / len(by_type)
        val_feasible = sum(metrics["feasible"] for metrics in by_type.values()) / len(by_type)
        val_score = sum(metrics["score"] for metrics in by_type.values()) / len(by_type)
        self.val_costs.append(val_cost)
        self.val_scores.append(val_score)
        self.val_phys.append(val_phys)
        self.val_pen.append(val_pen)
        self.val_unvisit.append(val_unvisit)
        self.val_feasible.append(val_feasible)
        self.val_by_type.append(by_type)
        return {
            "cost": val_cost,
            "score": val_score,
            "phys": val_phys,
            "pen": val_pen,
            "unvisit": val_unvisit,
            "feasible": val_feasible,
            "by_type": by_type,
        }

    def _validate_one_set(self, eval_model, val_data):
        total_costs = []
        total_phys = []
        total_pen = []
        total_unvisit = []
        total_feasible = []
        with torch.no_grad():
            for start in range(0, self.val_set_size, self.val_batch_chunk):
                end = min(start + self.val_batch_chunk, self.val_set_size)
                with self._amp_context():
                    result = eval_model.inference_rollout(
                        val_data["nodes"][start:end],
                        val_data["pref"][start:end],
                        val_data["depot"][start:end],
                        visited_mask=val_data["visited_mask"][start:end],
                        N_pomo=self.n_pomo,
                        decode_type=self.val_decode_type,
                        sample_rollouts=self.val_sample_rollouts,
                        multi_start_rollouts=self.val_multi_start_rollouts,
                        augment_8_fold=self.val_augment_8_fold,
                        scenario=self.scenario,
                    )
                total_costs.append(result["best_cost"])
                total_phys.append(result["best_phys"])
                total_pen.append(result["best_pen"])
                total_unvisit.append(result["best_unvisit"])
                total_feasible.append(result["feasible"].float())
        val_cost = torch.cat(total_costs).mean().item()
        val_phys = torch.cat(total_phys).mean().item()
        val_pen = torch.cat(total_pen).mean().item()
        val_unvisit = torch.cat(total_unvisit).mean().item()
        val_feasible = torch.cat(total_feasible).mean().item()
        val_score = (
            val_cost
            + val_unvisit * TRAIN_UNVISIT_WEIGHT
            + (1.0 - val_feasible) * TRAIN_INCOMPLETE_PENALTY
        )
        return {
            "cost": val_cost,
            "score": val_score,
            "phys": val_phys,
            "pen": val_pen,
            "unvisit": val_unvisit,
            "feasible": val_feasible,
        }

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        current_tightness = self._current_tightness(epoch, total_epochs)
        current_entropy_coef = self._current_entropy_coef(epoch, total_epochs)
        self.last_tightness = current_tightness
        self.last_entropy_coef = current_entropy_coef
        total_loss = 0.0
        total_phys = 0.0
        total_pen = 0.0
        total_unvisit = 0.0
        total_best_cost = 0.0
        total_mean_cost = 0.0
        total_policy_entropy = 0.0
        total_bc_loss = 0.0
        total_bc_invalid = 0.0
        bc_weight = self._current_bc_weight(epoch)

        pbar = tqdm(range(self.steps_per_epoch), desc=f"{self.variant_name} Epoch {epoch + 1}")
        self.optimizer.zero_grad(set_to_none=True)
        for step in pbar:
            nodes, depot, visited = self.generator.generate(
                self.batch_size, self.n_nodes, self.problem_type, current_tightness, DEVICE
            )
            pref = self._sample_pref(self.batch_size)
            curr_pos = depot.clone()
            curr_time = torch.zeros(self.batch_size, device=DEVICE)
            curr_load = torch.ones(self.batch_size, device=DEVICE)
            curr_energy = torch.ones(self.batch_size, device=DEVICE)

            with self._amp_context():
                log_probs, phys, pen, unvisit, ent, _, finished, at_depot = self.model(
                    nodes, pref, depot, visited, curr_pos, curr_time, curr_load, curr_energy, self.n_pomo, scenario=self.scenario
                )

                lp = log_probs.sum(-1)
                success = (unvisit <= 1e-8) & finished & at_depot
                incomplete = (~success).float()
                total_cost_tensor = (
                    phys * PHYS_WEIGHT
                    + pen * PEN_WEIGHT
                    + unvisit * TRAIN_UNVISIT_WEIGHT
                    + incomplete * TRAIN_INCOMPLETE_PENALTY
                )
                reward = -total_cost_tensor
                adv = reward - reward.mean(1, keepdim=True)
                if self.normalize_advantage:
                    adv = adv / adv.std(1, keepdim=True).clamp_min(1e-6)
                policy_entropy = ent.mean()
                loss = -(adv.detach() * lp).mean() - current_entropy_coef * policy_entropy
                if self.elite_fraction > 0.0 and self.elite_loss_weight > 0.0:
                    pomo_count = total_cost_tensor.size(1)
                    elite_count = max(1, min(pomo_count, int(round(pomo_count * self.elite_fraction))))
                    elite_idx = total_cost_tensor.topk(elite_count, dim=1, largest=False).indices
                    elite_mask = torch.zeros_like(total_cost_tensor, dtype=torch.bool)
                    elite_mask.scatter_(1, elite_idx, True)
                    elite_loss = -((adv.detach() * lp).masked_select(elite_mask)).mean()
                    loss = (1.0 - self.elite_loss_weight) * loss + self.elite_loss_weight * (
                        elite_loss - current_entropy_coef * policy_entropy
                    )
                bc_loss_value = None
                bc_invalid_value = 0.0
                if bc_weight > 0.0:
                    bc_batch = self._sample_bc_batch()
                    bc_result = self.model.teacher_forcing_loss(
                        bc_batch["nodes"],
                        bc_batch["pref"],
                        bc_batch["depot"],
                        bc_batch["teacher_actions"],
                        visited_mask=bc_batch["visited_mask"],
                        scenario=self.scenario,
                    )
                    bc_loss = bc_result["loss"]
                    bc_loss_value = float(bc_loss.item())
                    bc_invalid_value = float(bc_result["invalid_steps"].item())
                    loss = loss + bc_weight * bc_loss

            avg_phys = phys.mean().item()
            avg_pen = pen.mean().item()
            avg_unvisit = unvisit.mean().item()
            best_cost = total_cost_tensor.min(dim=1).values.mean().item()
            mean_cost = total_cost_tensor.mean().item()

            loss_value = loss.item()
            (loss / self.grad_accum_steps).backward()
            is_update_step = ((step + 1) % self.grad_accum_steps == 0) or ((step + 1) == self.steps_per_epoch)
            if is_update_step:
                nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._update_ema()

            total_loss += loss_value
            total_phys += avg_phys
            total_pen += avg_pen
            total_unvisit += avg_unvisit
            total_best_cost += best_cost
            total_mean_cost += mean_cost
            total_policy_entropy += policy_entropy.item()
            if bc_loss_value is not None:
                total_bc_loss += bc_loss_value
                total_bc_invalid += bc_invalid_value

            postfix = {
                "Loss": f"{total_loss / (step + 1):.4f}",
                "Best": f"{total_best_cost / (step + 1):.2f}",
                "Mean": f"{total_mean_cost / (step + 1):.2f}",
                "Phys": f"{total_phys / (step + 1):.2f}",
                "Pen": f"{total_pen / (step + 1):.2f}",
                "Unv": f"{total_unvisit / (step + 1):.2f}",
                "EntCoef": f"{current_entropy_coef:.4f}",
                "PolEnt": f"{total_policy_entropy / (step + 1):.3f}",
            }
            if bc_weight > 0.0:
                postfix["BCw"] = f"{bc_weight:.4f}"
                postfix["BC"] = f"{total_bc_loss / (step + 1):.3f}"
                postfix["BCinv"] = f"{total_bc_invalid:.0f}"
            pbar.set_postfix(postfix)

        avg_epoch_loss = total_loss / self.steps_per_epoch
        avg_best_cost = total_best_cost / self.steps_per_epoch
        avg_mean_cost = total_mean_cost / self.steps_per_epoch
        self.losses.append(avg_epoch_loss)
        self.train_best_costs.append(avg_best_cost)
        self.train_mean_costs.append(avg_mean_cost)
        return avg_epoch_loss

    def fit(self, epochs=EPOCHS, save_every=10, checkpoint_prefix=None, start_epoch=0):
        checkpoint_paths = []
        no_improve_epochs = 0
        stopped_early = False
        last_epoch = start_epoch
        for epoch in range(start_epoch, epochs):
            last_epoch = epoch + 1
            self.train_epoch(epoch, epochs)
            val_metrics = self.validate()
            if val_metrics is not None:
                self.scheduler.step(val_metrics["score"])
                current_lr = self.optimizer.param_groups[0]["lr"]
                val_type_msg = ""
                if val_metrics.get("by_type"):
                    val_type_msg = ", " + ", ".join(
                        f"val_{ptype}={metrics['cost']:.4f}/pen={metrics['pen']:.3f}/feas={metrics['feasible']:.3f}"
                        for ptype, metrics in val_metrics["by_type"].items()
                    )
                print(
                    f"[{self.variant_name}] epoch {epoch + 1}: "
                    f"train_best={self.train_best_costs[-1]:.4f}, "
                    f"train_mean={self.train_mean_costs[-1]:.4f}, "
                    f"val_best={val_metrics['cost']:.4f}, "
                    f"val_score={val_metrics['score']:.4f}, "
                    f"val_phys={val_metrics['phys']:.4f}, "
                    f"val_pen={val_metrics['pen']:.4f}, "
                    f"val_unv={val_metrics['unvisit']:.4f}, "
                    f"val_feasible={val_metrics['feasible']:.4f}, "
                    f"lr={current_lr:.6f}, "
                    f"tight={self.last_tightness:.3f}, "
                    f"ent={self.last_entropy_coef:.4f}"
                    f"{val_type_msg}"
                )
                if val_metrics["score"] < self.best_val_score - 1e-6:
                    self.best_val_cost = val_metrics["cost"]
                    self.best_val_score = val_metrics["score"]
                    self.best_epoch = epoch + 1
                    best_path = self._best_checkpoint_path(checkpoint_prefix=checkpoint_prefix)
                    self.save_checkpoint(best_path, epoch + 1, export_ema=self.use_ema_eval)
                    no_improve_epochs = 0
                else:
                    if (epoch + 1) >= self.early_stopping_min_epochs:
                        no_improve_epochs += 1
            should_save = save_every and ((epoch + 1) % save_every == 0 or (epoch + 1) == epochs)
            if should_save:
                path = self._default_checkpoint_path(epoch + 1, checkpoint_prefix=checkpoint_prefix)
                self.save_checkpoint(path, epoch + 1)
                checkpoint_paths.append(path)
            can_early_stop = (epoch + 1) >= self.early_stopping_min_epochs
            if can_early_stop and self.early_stopping_patience > 0 and no_improve_epochs >= self.early_stopping_patience:
                print(
                    f"[{self.variant_name}] early stopping at epoch {epoch + 1} "
                    f"(best_epoch={self.best_epoch}, best_val={self.best_val_cost:.4f})"
                )
                stopped_early = True
                break
        if stopped_early:
            last_path = self._default_checkpoint_path(last_epoch, checkpoint_prefix=checkpoint_prefix)
            self.save_checkpoint(last_path, last_epoch)
            checkpoint_paths.append(last_path)
        return checkpoint_paths

    def _default_checkpoint_path(self, epoch, checkpoint_prefix=None):
        os.makedirs(self.save_dir, exist_ok=True)
        prefix = checkpoint_prefix or self.variant_name
        return os.path.join(self.save_dir, f"{prefix}_epoch{epoch}.pt")

    def _best_checkpoint_path(self, checkpoint_prefix=None):
        os.makedirs(self.save_dir, exist_ok=True)
        prefix = checkpoint_prefix or self.variant_name
        return os.path.join(self.save_dir, f"{prefix}_best.pt")

    def save_checkpoint(self, path, epoch, export_ema=False):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        model_to_save = self.ema_model if (export_ema and self.use_ema_eval and self.ema_model is not None) else self.model
        torch.save(
            {
                "epoch": epoch,
                "variant_name": self.variant_name,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "ema_state_dict": self.ema_model.state_dict() if self.use_ema_eval and self.ema_model is not None else None,
                "losses": self.losses,
                "train_best_costs": self.train_best_costs,
                "train_mean_costs": self.train_mean_costs,
                "val_costs": self.val_costs,
                "val_scores": self.val_scores,
                "val_phys": self.val_phys,
                "val_pen": self.val_pen,
                "val_unvisit": self.val_unvisit,
                "val_feasible": self.val_feasible,
                "val_by_type": self.val_by_type,
                "best_val_cost": self.best_val_cost,
                "best_val_score": self.best_val_score,
                "best_epoch": self.best_epoch,
                "train_config": {
                    "batch_size": self.batch_size,
                    "n_nodes": self.n_nodes,
                    "n_pomo": self.n_pomo,
                    "steps_per_epoch": self.steps_per_epoch,
                    "problem_type": self.problem_type,
                    "scenario": self.scenario,
                    "tightness": self.tightness,
                    "entropy_coef_start": self.entropy_coef_start,
                    "entropy_coef_end": self.entropy_coef_end,
                    "curriculum_start_tightness": self.curriculum_start_tightness,
                    "curriculum_epochs": self.curriculum_epochs,
                    "early_stopping_min_epochs": self.early_stopping_min_epochs,
                    "ema_decay": self.ema_decay,
                    "use_ema_eval": self.use_ema_eval,
                    "normalize_advantage": self.normalize_advantage,
                    "train_unvisit_weight": TRAIN_UNVISIT_WEIGHT,
                    "train_incomplete_penalty": TRAIN_INCOMPLETE_PENALTY,
                    "grad_accum_steps": self.grad_accum_steps,
                    "pref_mode": self.pref_mode,
                    "pref_alpha": self.pref_alpha,
                    "pref_beta": self.pref_beta,
                    "val_pref_mode": self.val_pref_mode,
                    "val_pref_alpha": self.val_pref_alpha,
                    "val_pref_beta": self.val_pref_beta,
                    "val_problem_types": self.val_problem_types,
                    "pref_random_low": self.pref_random_low,
                    "pref_random_high": self.pref_random_high,
                    "amp": self.use_amp,
                    "amp_dtype": self.amp_dtype_name,
                },
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.losses = checkpoint.get("losses", [])
            self.train_best_costs = checkpoint.get("train_best_costs", [])
            self.train_mean_costs = checkpoint.get("train_mean_costs", [])
            self.val_costs = checkpoint.get("val_costs", [])
            self.val_scores = checkpoint.get("val_scores", [])
            self.val_phys = checkpoint.get("val_phys", [])
            self.val_pen = checkpoint.get("val_pen", [])
            self.val_unvisit = checkpoint.get("val_unvisit", [])
            self.val_feasible = checkpoint.get("val_feasible", [])
            self.val_by_type = checkpoint.get("val_by_type", [])
            self.best_val_cost = checkpoint.get("best_val_cost", float("inf"))
            self.best_val_score = checkpoint.get("best_val_score", checkpoint.get("best_val_cost", float("inf")))
            self.best_epoch = checkpoint.get("best_epoch", 0)
            if self.use_ema_eval and self.ema_model is not None:
                ema_state_dict = checkpoint.get("ema_state_dict")
                if ema_state_dict is not None:
                    self.ema_model.load_state_dict(ema_state_dict, strict=False)
                else:
                    self.ema_model.load_state_dict(self.model.state_dict(), strict=False)
            return checkpoint.get("epoch", 0)
        self.model.load_state_dict(checkpoint, strict=False)
        if self.use_ema_eval and self.ema_model is not None:
            self.ema_model.load_state_dict(self.model.state_dict(), strict=False)
        return 0

    def save(self, path):
        self.save_checkpoint(path, len(self.losses))
