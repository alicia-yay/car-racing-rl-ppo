#!/usr/bin/env python3
"""
evaluate.py — Generate a GIF from a trained CarRacing-v3 checkpoint.

Lightweight: loads only the RLModule weights on CPU.
No Ray, no GPU, no Algorithm — safe to run alongside training.

Usage:
    python evaluate.py \
    --checkpoint "/mnt/cluster_storage/car-racing-ppo/checkpoints/carracing_ppo/PPO_CarRacing-v3-improved_58c29_00000_0_entropy_coeff=0.0287,lr=0.0004,minibatch_size=256,num_epochs=10,train_batch_size_per_learn_2026-02-14_06-06-49/checkpoint_000000" \
    --progress-csv "/mnt/cluster_storage/car-racing-ppo/checkpoints/carracing_ppo/PPO_CarRacing-v3-improved_58c29_00000_0_entropy_coeff=0.0287,lr=0.0004,minibatch_size=256,num_epochs=10,train_batch_size_per_learn_2026-02-14_06-06-49/progress.csv" \
    --output carracing.gif \
    --plot-output training_curve.png \
    --finetune-at 500 \
    --episodes 3

    python evaluate.py \
    --progress-csv "/mnt/cluster_storage/car-racing-ppo/checkpoints/carracing_ppo/PPO_CarRacing-v3-improved_58c29_00000_0_entropy_coeff=0.0287,lr=0.0004,minibatch_size=256,num_epochs=10,train_batch_size_per_learn_2026-02-14_06-06-49/progress.csv" \
    --plot-output training_curve.png \
    --finetune-at 500
"""

import argparse, os, pickle
import numpy as np
import gymnasium as gym
import imageio
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt

from wrappers import (
    FrameSkip, Resize84x84, ToFloat01, RewardScale, StackToChannels,
)
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers.stateful_observation import FrameStackObservation


def make_eval_env():
    """Create evaluation env with render_mode for frame capture."""
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = FrameSkip(env, skip=4)
    env = GrayscaleObservation(env, keep_dim=True)
    env = Resize84x84(env)
    env = ToFloat01(env)
    env = RewardScale(env, 0.1)
    env = FrameStackObservation(env, stack_size=4)
    env = StackToChannels(env)
    return env


def load_module_from_checkpoint(ckpt_path):
    """
    Load RLModule from checkpoint using the saved class and constructor args.
    Works without Ray or GPU.
    """
    ckpt_path = os.path.expanduser(ckpt_path)

    ctor_path = os.path.join(
        ckpt_path, "learner_group", "learner", "rl_module",
        "default_policy", "class_and_ctor_args.pkl"
    )
    weights_path = os.path.join(
        ckpt_path, "learner_group", "learner", "rl_module",
        "default_policy", "module_state.pkl"
    )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Module weights not found: {weights_path}")
    if not os.path.exists(ctor_path):
        raise FileNotFoundError(f"Module config not found: {ctor_path}")

    # Load weights
    print(f"Loading weights: {weights_path}")
    with open(weights_path, "rb") as f:
        state = pickle.load(f)
    print(f"  {len(state)} weight tensors")

    # Convert numpy → torch tensors
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            state[k] = torch.from_numpy(v.copy())

    # Reconstruct module using saved class + args (exact match to training)
    print(f"Loading module config: {ctor_path}")
    with open(ctor_path, "rb") as f:
        ctor = pickle.load(f)

    module_class = ctor["class"]
    args_tuple, kwargs = ctor["ctor_args_and_kwargs"]
    module = module_class(*args_tuple, **kwargs)
    module.load_state_dict(state, strict=False)
    module.eval()
    print(f"  ✓ Module loaded ({module_class.__name__})")
    return module


def get_action(module, obs):
    """Run forward inference, return mean action (no sampling noise)."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = module.forward_inference({"obs": obs_t})

    if isinstance(out, dict) and "action_dist_inputs" in out:
        dist = out["action_dist_inputs"]
        dim = dist.shape[-1] // 2
        return dist[0, :dim].cpu().numpy().astype(np.float32)
    if isinstance(out, dict) and "actions" in out:
        return out["actions"][0].cpu().numpy().astype(np.float32)
    if isinstance(out, dict):
        v = next(iter(out.values()))
        return v[0].cpu().numpy().astype(np.float32)
    return out[0].cpu().numpy().astype(np.float32)


def postprocess_action(action, prev_filtered, alpha=0.2,
                       steer_deadzone=0.05, steer_clip=0.8):
    """Smooth and clip actions for cleaner driving."""
    # EMA smoothing
    if alpha > 0:
        action = (1 - alpha) * prev_filtered + alpha * action

    # Steering: deadzone + clip
    s = float(action[0])
    if abs(s) < steer_deadzone:
        s = 0.0
    action[0] = np.clip(s, -steer_clip, steer_clip)

    # Gas/brake sanity
    action[1] = max(0.0, float(action[1]))
    action[2] = max(0.0, float(action[2]))
    if action[2] > 0.2:
        action[1] *= 0.5  # reduce gas while braking hard

    return action.copy()


def run_episode(module, max_steps, smooth_alpha, steer_deadzone, steer_clip):
    """Run one episode, return (frames, raw_reward, steps)."""
    env = make_eval_env()
    obs, _ = env.reset()
    frames = []
    ep_reward = 0.0
    a_filt = np.zeros(3, np.float32)

    for step in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        raw_action = get_action(module, obs)
        action = postprocess_action(
            raw_action, a_filt, smooth_alpha, steer_deadzone, steer_clip
        )
        a_filt = action.copy()
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, term, trunc, _ = env.step(action.astype(np.float32))
        ep_reward += float(reward)

        if term or trunc:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            break

    env.close()
    raw_reward = ep_reward / 0.1  # undo RewardScale
    return frames, raw_reward, len(frames)


def plot_training_curve(csv_path, output_path="training_curve.png", finetune_at=None):
    """Generate a training curve plot from RLlib's progress.csv."""
    df = pd.read_csv(csv_path)

    iters = df["training_iteration"]
    mean_ret = df["env_runners/episode_return_mean"]
    max_ret = df["env_runners/episode_return_max"]
    min_ret = df["env_runners/episode_return_min"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(iters, max_ret, "--o", lw=1, ms=3, alpha=0.7, label="Max")
    ax.plot(iters, mean_ret, "-o", lw=2, ms=4, label="Mean")
    ax.plot(iters, min_ret, "--o", lw=1, ms=3, alpha=0.7, label="Min")

    # Mark fine-tune transition
    if finetune_at is not None and finetune_at <= iters.max():
        ax.axvline(x=finetune_at, color="red", linestyle="--", alpha=0.6, lw=1.5)
        ax.text(finetune_at + 2, ax.get_ylim()[1] * 0.95,
                "← Fine-tune", color="red", fontsize=9, va="top")

    start_mean = mean_ret.iloc[0]
    end_mean = mean_ret.iloc[-1]
    ax.set_title("Return per Training Iteration", fontsize=18, pad=14)
    fig.suptitle(
        f"Mean return: ~{start_mean:.1f} → ~{end_mean:.1f} over {len(iters)} iterations",
        fontsize=10, y=0.92, style="italic",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Return (scaled)")
    ax.grid(linestyle=":", lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curve: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained CarRacing agent")
    ap.add_argument("--checkpoint", default=None, help="Path to RLlib checkpoint")
    ap.add_argument("--output", default="carracing.gif", help="Output GIF path")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--smooth-alpha", type=float, default=0.2,
                    help="EMA smoothing (0=disabled)")
    ap.add_argument("--steer-deadzone", type=float, default=0.05)
    ap.add_argument("--steer-clip", type=float, default=0.8)
    ap.add_argument("--progress-csv", default=None,
                    help="Path to progress.csv to plot training curve")
    ap.add_argument("--plot-output", default="training_curve.png",
                    help="Output path for training curve plot")
    ap.add_argument("--finetune-at", type=int, default=None,
                    help="Mark fine-tune transition on plot")
    args = ap.parse_args()

    # Plot training curve if requested
    if args.progress_csv:
        plot_training_curve(args.progress_csv, args.plot_output, args.finetune_at)
        if not args.checkpoint:
            return  # only plotting, no GIF

    if not args.checkpoint:
        ap.error("--checkpoint is required for GIF generation")

    module = load_module_from_checkpoint(args.checkpoint)
    print()

    all_frames = []
    rewards = []

    for ep in range(args.episodes):
        frames, raw_reward, steps = run_episode(
            module, args.max_steps,
            args.smooth_alpha, args.steer_deadzone, args.steer_clip,
        )
        print(f"Episode {ep+1}: reward={raw_reward:.0f}  steps={steps}")
        rewards.append(raw_reward)
        all_frames.extend(frames)

    if args.episodes > 1:
        print(f"\nMean reward: {np.mean(rewards):.0f} ± {np.std(rewards):.0f}")

    imageio.mimsave(args.output, all_frames, fps=args.fps, loop=0)
    print(f"\nSaved: {args.output} ({len(all_frames)} frames)")


if __name__ == "__main__":
    main()