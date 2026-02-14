"""
Shared CarRacing-v3 environment wrappers for training and evaluation.

Preprocessing pipeline:
  CarRacing-v3 (96x96 RGB)
  → FrameSkip(4)
  → Grayscale (H,W,1)
  → Resize to 84x84
  → Float [0,1]
  → RewardScale(0.1)
  → FrameStack(4) → StackToChannels → (84,84,4)
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers.stateful_observation import FrameStackObservation
from PIL import Image


class FrameSkip(gym.Wrapper):
    """Repeat each action for `skip` frames, summing rewards."""
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        assert skip >= 1
        self._skip = int(skip)

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += float(reward)
            if term or trunc:
                return obs, total_reward, term, trunc, info
        return obs, total_reward, False, False, info

    def reset(self, **kw):
        return self.env.reset(**kw)


class Resize84x84(gym.ObservationWrapper):
    """PIL resize; handles (H,W,1) grayscale with keep_dim."""
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self._h, self._w = int(shape[0]), int(shape[1])
        old = env.observation_space
        assert len(old.shape) == 3, f"Expected HWC obs, got {old.shape}"
        c = int(old.shape[2])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self._h, self._w, c), dtype=np.uint8,
        )

    def observation(self, obs):
        arr = np.asarray(obs)
        if arr.ndim == 3 and arr.shape[2] == 1:
            img = Image.fromarray(arr[:, :, 0], mode="L")
            img = img.resize((self._w, self._h), resample=Image.BILINEAR)
            return np.asarray(img, dtype=np.uint8)[:, :, None]
        img = Image.fromarray(arr)
        img = img.resize((self._w, self._h), resample=Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


class ToFloat01(gym.ObservationWrapper):
    """Convert uint8 [0,255] → float32 [0,1]."""
    def __init__(self, env):
        super().__init__(env)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=shp, dtype=np.float32,
        )

    def observation(self, obs):
        return np.asarray(obs, dtype=np.float32) / 255.0


class RewardScale(gym.RewardWrapper):
    """Scale rewards to keep gradients reasonable."""
    def __init__(self, env, scale=0.1):
        super().__init__(env)
        self._scale = float(scale)

    def reward(self, reward):
        return float(reward) * self._scale


class StackToChannels(gym.ObservationWrapper):
    """Convert FrameStack (S,H,W,C) → (H,W,S*C) for RLlib's default CNN."""
    def __init__(self, env):
        super().__init__(env)
        shp = env.observation_space.shape
        assert len(shp) == 4, f"Expected (S,H,W,C), got {shp}"
        s, h, w, c = map(int, shp)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(h, w, s * c), dtype=np.float32,
        )

    def observation(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        s, h, w, c = x.shape
        return np.transpose(x, (1, 2, 0, 3)).reshape(h, w, s * c)


def make_carracing_env(frame_skip=4, resize_shape=(84, 84), frame_stack=4):
    """Factory that returns an env creator function for RLlib's register_env."""
    def _init(_cfg=None):
        env = gym.make("CarRacing-v3")
        env = FrameSkip(env, skip=frame_skip)
        env = GrayscaleObservation(env, keep_dim=True)
        env = Resize84x84(env, shape=resize_shape)
        env = ToFloat01(env)
        env = RewardScale(env, scale=0.1)
        env = FrameStackObservation(env, stack_size=frame_stack)
        env = StackToChannels(env)
        return env
    return _init
