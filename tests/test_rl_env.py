"""Tests for the Gymnasium HardwareDesignEnv (RL/GRPO foundation)."""

import pytest
import gymnasium as gym
from learning.environment import HardwareDesignEnv


class TestHardwareDesignEnv:
    def test_env_reset(self, tmp_path):
        env = HardwareDesignEnv(work_dir=str(tmp_path / "rl_work"))
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "task" in obs
        assert obs["step"] == 0
        assert obs["best_reward"] == 0.0
        assert env.episode_return == 0.0
        assert not env.is_done
        assert "target_module" in info

    def test_action_space(self):
        env = HardwareDesignEnv()
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == len(env.ACTION_SPACE)
        assert "GENERATE_RTL" in env.ACTION_SPACE
        assert "SIMULATE" in env.ACTION_SPACE
        assert "TEST" in env.ACTION_SPACE
        assert "SYNTHESIZE" in env.ACTION_SPACE
        assert "COMPLETE" in env.ACTION_SPACE

    def test_step_sequence_and_reward_accumulation(self, tmp_path):
        env = HardwareDesignEnv(work_dir=str(tmp_path / "rl_work"), max_steps=10)
        env.reset()

        # Step 1: Generate RTL (using string action)
        obs, reward, terminated, truncated, info = env.step("GENERATE_RTL")
        assert reward > 0
        assert not terminated
        assert not truncated
        assert env.episode_return == reward

        # Step 2: Simulate (Verilator lint)
        obs, reward, terminated, truncated, info = env.step("SIMULATE")
        assert reward == 1.0
        assert env.episode_return > 1.0

        # Step 3: Test (Cocotb functional)
        obs, reward, terminated, truncated, info = env.step("TEST")
        assert reward == 5.0
        assert env.episode_return >= 6.1

        # Step 4: Synthesize (Yosys)
        obs, reward, terminated, truncated, info = env.step("SYNTHESIZE")
        assert reward == 1.0
        assert env.best_design_reward >= 7.0

        # Step 5: Complete
        obs, reward, terminated, truncated, info = env.step("COMPLETE")
        assert terminated
        assert obs["status"] == "done"

    def test_discrete_action_indexing(self, tmp_path):
        env = HardwareDesignEnv(work_dir=str(tmp_path / "rl_work"))
        env.reset()
        # Action 2 is GENERATE_RTL
        obs, reward, terminated, truncated, info = env.step(2)
        assert info["action"] == "GENERATE_RTL"
        assert reward > 0
