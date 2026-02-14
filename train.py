#!/usr/bin/env python3
"""
train.py — PPO for CarRacing-v3 with ASHA hyperparameter search.

Includes a StabilizationCallback that automatically switches to fine-tuning
mode (lower LR, entropy, clip) after the exploration phase completes.

Two-phase training in a single run:
  Phase 1 (iters 0 → finetune_at):  Explore — high LR, high entropy
  Phase 2 (iters finetune_at → end): Stabilize — low LR, low entropy, tighter clip

Usage:
    python train.py
    python train.py --finetune-at 300 --max-iters 800
    python train.py --skip-asha  # Single run with default hparams, no search
"""

import os
import json
import csv
import time
import argparse
import numpy as np
import ray
import torch

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.tune import register_env

from wrappers import make_carracing_env

# ============================================================
# StabilizationCallback — auto fine-tune at a given iteration
# ============================================================
class StabilizationCallback(RLlibCallback):
    """
    At `finetune_at` iteration, automatically lower LR, entropy, and clip
    to stabilize and smooth out the learned policy.

    Args:
        finetune_at: Iteration number to switch to fine-tune mode.
        finetune_lr: Learning rate for fine-tuning phase.
        finetune_entropy: Entropy coefficient for fine-tuning phase.
        finetune_clip: PPO clip parameter for fine-tuning phase.
    """
    def __init__(self, finetune_at=500, finetune_lr=5e-5,
                 finetune_entropy=1e-3, finetune_clip=0.15):
        super().__init__()
        self.finetune_at = finetune_at
        self.finetune_lr = finetune_lr
        self.finetune_entropy = finetune_entropy
        self.finetune_clip = finetune_clip
        self._switched = False

    def on_train_result(self, *, algorithm, result, **kwargs):
        it = result.get("training_iteration", 0)
        if it >= self.finetune_at and not self._switched:
            self._switched = True
            print("\n" + "=" * 60)
            print(f"  PHASE 2: Stabilization mode at iter {it}")
            print(f"  lr: {self.finetune_lr}  entropy: {self.finetune_entropy}"
                  f"  clip: {self.finetune_clip}")
            print("=" * 60 + "\n")

            _apply_overrides(
                algorithm,
                lr=self.finetune_lr,
                entropy=self.finetune_entropy,
                clip=self.finetune_clip,
            )


# ============================================================
# Hyperparameter patching (works on new RLlib API stack)
# ============================================================
def _apply_overrides(algo, lr, entropy, clip):
    """Patch LR, entropy, clip on a live Algorithm (new API stack)."""
    # 1. Patch algo.config for logging consistency
    try:
        algo.config = algo.config.copy(copy_frozen=False)
        algo.config.lr = float(lr)
        algo.config.entropy_coeff = float(entropy)
        algo.config.clip_param = float(clip)
        algo.config.freeze()
    except Exception:
        pass

    # 2. Patch learners
    lg = getattr(algo, "learner_group", None)
    if lg is None:
        return

    def _patch(learner):
        # Unfreeze and patch config
        try:
            learner.config = learner.config.copy(copy_frozen=False)
            learner.config.lr = float(lr)
            learner.config.entropy_coeff = float(entropy)
            learner.config.clip_param = float(clip)
            learner.config.lr_schedule = None
            learner.config.freeze()
        except Exception:
            pass

        # Patch optimizer LR directly
        for optim in learner._named_optimizers.values():
            for pg in optim.param_groups:
                pg["lr"] = float(lr)

        # Clear LR schedulers so they don't overwrite our LR
        for attr in ("_lr_schedulers", "lr_schedulers", "_optimizer_lr_schedules"):
            scheds = getattr(learner, attr, None)
            if scheds is not None:
                if isinstance(scheds, (dict, list)):
                    scheds.clear()

        # Fix Adam foreach/tensor bug if present
        for optim in learner._named_optimizers.values():
            for pg in optim.param_groups:
                pg["foreach"] = False
                if "betas" in pg:
                    b1, b2 = pg["betas"]
                    if isinstance(b1, torch.Tensor):
                        b1 = b1.item()
                    if isinstance(b2, torch.Tensor):
                        b2 = b2.item()
                    pg["betas"] = (b1, b2)
            for state in optim.state.values():
                if "step" in state and isinstance(state["step"], torch.Tensor):
                    state["step"] = state["step"].cpu().item()
        return True

    try:
        lg.foreach_learner(_patch)
    except Exception as e:
        print(f"WARNING: Could not patch learners: {e}")


# ============================================================
# Aggregate CSV Logger (for ASHA multi-trial progress)
# ============================================================
class AggregateProgressCSV(tune.Callback):
    """Writes a single progress.csv across all ASHA trials."""
    def __init__(self, out_csv_path, metric_key):
        self.out_csv_path = out_csv_path
        self.metric_key = metric_key
        self._start = None
        self._header_written = False

    def _ensure_header(self):
        if self._header_written:
            return
        os.makedirs(os.path.dirname(self.out_csv_path), exist_ok=True)
        if not os.path.exists(self.out_csv_path):
            with open(self.out_csv_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "time_s", "trial_id", "training_iteration",
                    self.metric_key, "lr", "entropy_coeff",
                    "train_batch_size_per_learner", "minibatch_size", "num_epochs",
                ])
        self._header_written = True

    def on_experiment_start(self, **info):
        self._start = time.time()
        self._ensure_header()

    def on_trial_result(self, iteration, trials, trial, result, **info):
        self._ensure_header()
        t = time.time() - (self._start or time.time())
        metric = result.get(self.metric_key)
        if metric is None:
            for k in ("episode_return_mean", "env_runners/episode_reward_mean"):
                if k in result:
                    metric = result[k]
                    break
        if metric is None:
            return
        cfg = trial.config or {}
        with open(self.out_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                round(t, 2), str(getattr(trial, "trial_id", "")),
                int(result.get("training_iteration", -1)), float(metric),
                cfg.get("lr"), cfg.get("entropy_coeff"),
                cfg.get("train_batch_size_per_learner"),
                cfg.get("minibatch_size"), cfg.get("num_epochs"),
            ])


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=1000,
                        help="Total training iterations (both phases)")
    parser.add_argument("--finetune-at", type=int, default=500,
                        help="Iteration to switch to fine-tuning phase")
    parser.add_argument("--finetune-lr", type=float, default=5e-5)
    parser.add_argument("--finetune-entropy", type=float, default=1e-3)
    parser.add_argument("--finetune-clip", type=float, default=0.15)
    parser.add_argument("--num-samples", type=int, default=6,
                        help="Number of ASHA trials")
    parser.add_argument("--skip-asha", action="store_true",
                        help="Skip ASHA search; run a single trial with defaults")
    parser.add_argument("--results-dir",
                        default="/mnt/cluster_storage/car-racing-ppo/checkpoints")
    args = parser.parse_args()

    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    EXP_NAME = "carracing_ppo"
    METRIC = "env_runners/episode_return_mean"

    ray.init(
        ignore_reinit_error=True,
        runtime_env={"pip": ["pillow", "gymnasium[box2d]"]},
    )

    cluster = ray.cluster_resources()
    gpu_available = cluster.get("GPU", 0) >= 1
    print(f"Ray cluster: {cluster}")
    print(f"GPU available: {gpu_available}")

    register_env("CarRacing-v3-improved", make_carracing_env())

    # --- Build PPO config ---
    # Store fine-tune args so the callback class can access them
    _ft_at = args.finetune_at
    _ft_lr = args.finetune_lr
    _ft_entropy = args.finetune_entropy
    _ft_clip = args.finetune_clip

    class ConfiguredStabilizationCallback(StabilizationCallback):
        def __init__(self):
            super().__init__(
                finetune_at=_ft_at, finetune_lr=_ft_lr,
                finetune_entropy=_ft_entropy, finetune_clip=_ft_clip,
            )

    base_config = (
        PPOConfig()
        .environment("CarRacing-v3-improved")
        .rl_module(
            model_config=DefaultModelConfig(
                conv_filters=[[32, 8, 4], [64, 4, 2], [64, 3, 1]],
                conv_activation="relu",
                head_fcnet_hiddens=[256],
                head_fcnet_activation="relu",
                vf_share_layers=False,
            )
        )
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=2,
            sample_timeout_s=300.0,
        )
        .training(
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=(1 if gpu_available else 0),
        )
        .callbacks(ConfiguredStabilizationCallback)
        .framework("torch")
    )

    config = base_config.to_dict()

    if args.skip_asha:
        # Single run with reasonable defaults
        config["lr"] = 2e-4
        config["entropy_coeff"] = 0.037
        config["train_batch_size_per_learner"] = 4096
        config["minibatch_size"] = 256
        config["num_epochs"] = 10
        num_samples = 1
    else:
        # ASHA hyperparameter search
        config["lr"] = tune.loguniform(1e-4, 5e-4)
        config["entropy_coeff"] = tune.loguniform(5e-3, 5e-2)
        config["train_batch_size_per_learner"] = tune.choice([4096, 8192])
        config["minibatch_size"] = tune.choice([256, 512])
        config["num_epochs"] = tune.choice([6, 10])
        num_samples = args.num_samples

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=args.max_iters,
        grace_period=15,
        reduction_factor=3,
    )

    print("\n" + "=" * 60)
    print(f"Training CarRacing-v3 PPO")
    print(f"  Phase 1 (explore):    iters 0 → {args.finetune_at}")
    print(f"  Phase 2 (fine-tune):  iters {args.finetune_at} → {args.max_iters}")
    print(f"  Fine-tune LR={args.finetune_lr}  entropy={args.finetune_entropy}"
          f"  clip={args.finetune_clip}")
    print(f"  ASHA trials: {num_samples}")
    print("=" * 60 + "\n")

    agg_csv = os.path.join(args.results_dir, EXP_NAME, "progress.csv")

    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=tune.RunConfig(
            name=EXP_NAME,
            storage_path=args.results_dir,
            verbose=2,
            callbacks=[AggregateProgressCSV(agg_csv, METRIC)],
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
                num_to_keep=3,
            ),
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_samples,
            max_concurrent_trials=1,
            metric=METRIC,
            mode="max",
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric=METRIC, mode="max")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best trial config:")
    print(f"  lr={best.config['lr']:.6f}  entropy={best.config['entropy_coeff']:.6f}")
    print(f"  batch={best.config['train_batch_size_per_learner']}"
          f"  minibatch={best.config['minibatch_size']}"
          f"  epochs={best.config['num_epochs']}")
    print(f"Best reward (scaled): {best.metrics[METRIC]:.1f}"
          f"  (raw ≈ {best.metrics[METRIC] / 0.1:.0f})")
    print(f"Best checkpoint: {best.checkpoint}")
    print("=" * 60)

    # Save best config
    os.makedirs(os.path.join(args.results_dir, EXP_NAME), exist_ok=True)
    cfg_path = os.path.join(args.results_dir, EXP_NAME, "best_config.json")
    with open(cfg_path, "w") as f:
        json.dump({
            "lr": best.config["lr"],
            "entropy_coeff": best.config["entropy_coeff"],
            "train_batch_size_per_learner": best.config["train_batch_size_per_learner"],
            "minibatch_size": best.config["minibatch_size"],
            "num_epochs": best.config["num_epochs"],
            "best_reward_scaled": float(best.metrics[METRIC]),
            "best_checkpoint": str(best.checkpoint),
            "finetune_at": args.finetune_at,
            "finetune_lr": args.finetune_lr,
            "finetune_entropy": args.finetune_entropy,
            "finetune_clip": args.finetune_clip,
        }, f, indent=2)

    print(f"\nProgress CSV: {agg_csv}")
    print(f"Best config: {cfg_path}")
    print(f"\nTo evaluate:\n  python evaluate.py --checkpoint {best.checkpoint}")

    ray.shutdown()


if __name__ == "__main__":
    main()