"""Tests for the Gym-compatible HardwareDesignEnv (RL/GRPO foundation)."""

import pytest
from learning.environment import HardwareDesignEnv


class TestHardwareDesignEnv:
    def test_env_reset(self, tmp_path):
        env = HardwareDesignEnv(work_dir=str(tmp_path / "rl_work"))
        obs = env.reset()
        assert "task" in obs
        assert obs["step"] == 0
        assert obs["best_reward"] == 0.0
        assert env.episode_return == 0.0
        assert not env.is_done

    def test_action_space(self):
        env = HardwareDesignEnv()
        assert "GENERATE_RTL" in env.ACTION_SPACE
        assert "SIMULATE" in env.ACTION_SPACE
        assert "TEST" in env.ACTION_SPACE
        assert "SYNTHESIZE" in env.ACTION_SPACE
        assert "COMPLETE" in env.ACTION_SPACE

    def test_step_sequence_and_reward_accumulation(self, tmp_path):
        env = HardwareDesignEnv(work_dir=str(tmp_path / "rl_work"), max_steps=10)
        env.reset()

        # Step 1: Generate RTL
        obs, reward, done, info = env.step("GENERATE_RTL")
        assert reward > 0
        assert not done
        assert env.episode_return == reward

        # Step 2: Simulate (Verilator lint)
        obs, reward, done, info = env.step("SIMULATE")
        assert reward == 1.0
        assert env.episode_return > 1.0

        # Step 3: Test (Cocotb functional)
        obs, reward, done, info = env.step("TEST")
        assert reward == 5.0
        assert env.episode_return >= 6.1

        # Step 4: Synthesize (Yosys)
        obs, reward, done, info = env.step("SYNTHESIZE")
        assert reward == 1.0
        assert env.best_design_reward >= 7.0

        # Step 5: Complete
        obs, reward, done, info = env.step("COMPLETE")
        assert done
        assert obs["status"] == "done"
