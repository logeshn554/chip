"""Tests for the Gymnasium HardwareDesignEnv (RL/GRPO foundation)."""

import pytest
import gymnasium as gym
from learning.environment import HardwareDesignEnv

SAMPLE_MAC_CODE = """`timescale 1ns / 1ps
module mac #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     valid_in,
    input  logic signed [DATA_WIDTH-1:0] a,
    input  logic signed [DATA_WIDTH-1:0] b,
    output logic signed [ACC_WIDTH-1:0]  accum,
    output logic                     valid_out
);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum     <= '0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            accum     <= accum + (a * b);
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end
endmodule
"""


class TestHardwareDesignEnv:
    def test_env_reset_and_space_contract(self, tmp_path):
        env = HardwareDesignEnv(work_dir=str(tmp_path / "rl_work"))
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "task" in obs
        assert obs["step"] == 0
        assert obs["best_reward"] == 0.0
        assert env.episode_return == 0.0
        assert not env.is_done
        assert "target_module" in info

        # Observation space contract check
        assert env.observation_space.contains(obs)

    def test_action_space(self):
        env = HardwareDesignEnv()
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == len(env.ACTION_SPACE)
        assert "GENERATE_RTL" in env.ACTION_SPACE
        assert "SIMULATE" in env.ACTION_SPACE
        assert "TEST" in env.ACTION_SPACE
        assert "SYNTHESIZE" in env.ACTION_SPACE
        assert "COMPLETE" in env.ACTION_SPACE

    def test_step_requires_code_no_secret_fallback(self, tmp_path):
        env = HardwareDesignEnv(work_dir=str(tmp_path / "rl_work"))
        env.reset()
        # Calling GENERATE_RTL without code must NOT create a secret fallback MAC
        obs, reward, terminated, truncated, info = env.step("GENERATE_RTL")
        assert reward < 0.0
        assert info["status"] == "missing_code"
        assert obs["current_rtl"] == ""

    def test_premature_complete_penalty(self, tmp_path):
        env = HardwareDesignEnv(work_dir=str(tmp_path / "rl_work"))
        env.reset()
        # Calling COMPLETE before compiling/testing must incur a penalty
        obs, reward, terminated, truncated, info = env.step("COMPLETE")
        assert terminated is True
        assert reward < 0.0
        assert info["verified_complete"] is False

    def test_step_sequence_and_design_quality_separation(self, tmp_path):
        env = HardwareDesignEnv(work_dir=str(tmp_path / "rl_work"), max_steps=10)
        env.reset()

        # Step 1: Generate RTL with actual code provided by policy
        action_payload = {"action": "GENERATE_RTL", "params": {"code": SAMPLE_MAC_CODE}}
        obs, reward, terminated, truncated, info = env.step(action_payload)
        assert reward > 0
        assert not terminated
        assert not truncated
        assert obs["current_rtl"] != ""
        assert env.observation_space.contains(obs)

        # Step 2: Simulate (Verilator lint & compile)
        obs, reward, terminated, truncated, info = env.step("SIMULATE")
        assert reward > 0
        assert info["compile_passed"] is True

        # Step 3: Test (Cocotb functional)
        obs, reward, terminated, truncated, info = env.step("TEST")
        assert reward > 0
        assert info["functional_passed"] is True

        # Step 4: Synthesize (Yosys)
        obs, reward, terminated, truncated, info = env.step("SYNTHESIZE")
        assert reward > 0
        assert info["synthesis_passed"] is True

        # Design quality is normalized [0.0, 1.0], strictly distinct from accumulated episode_return
        assert 0.5 <= env.current_design_quality <= 1.0
        assert obs["current_reward"] == round(env.current_design_quality, 4)

        # Step 5: Complete
        obs, reward, terminated, truncated, info = env.step("COMPLETE")
        assert terminated is True
        assert reward > 0  # Terminal success bonus
        assert info["verified_complete"] is True
        assert obs["status"] == "done"
        assert env.observation_space.contains(obs)

    def test_vector_observation_wrapper(self, tmp_path):
        from learning.environment import HardwareVectorObservationWrapper
        import numpy as np
        raw_env = HardwareDesignEnv(work_dir=str(tmp_path / "rl_vec"))
        env = HardwareVectorObservationWrapper(raw_env)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)

    def test_qwen_hardware_design_policy(self):
        from learning.environment import QwenHardwareDesignPolicy
        policy = QwenHardwareDesignPolicy()
        obs = {"current_rtl": "", "last_error": "", "best_reward": 0.0}
        act = policy.select_action(obs)
        assert act["action"] == "GENERATE_RTL"

        obs["current_rtl"] = "module test; endmodule"
        act = policy.select_action(obs)
        assert act["action"] == "SIMULATE"

    def test_benchmark_auto_binding(self, tmp_path):
        env = HardwareDesignEnv(
            work_dir=str(tmp_path / "rl_bench"),
            benchmark_task_id="L3_MAC_8BIT_SIGNED",
        )
        assert "L3_MAC_8BIT_SIGNED" in env.task or "8-bit Signed Multiply-Accumulate" in env.task
        assert env.target_module == "mac"
        assert env.formal_properties is not None
        assert "assert property" in env.formal_properties
