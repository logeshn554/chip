"""Tests for genuine hardware-in-the-loop GRPO and RL Transition Dataset Builder."""

import pytest
from learning.grpo import HardwareRewardEvaluator, compute_group_advantages
from learning.dataset import TrajectoryDatasetBuilder, RLTransition
from agent.schemas import Episode, TrajectoryStep


class TestHardwareGRPO:
    def test_extract_rtl(self):
        evaluator = HardwareRewardEvaluator()
        
        # Test 1: Markdown fenced SystemVerilog
        text1 = "Here is the design:\n```systemverilog\nmodule test; endmodule\n```\nDone."
        assert evaluator.extract_rtl(text1) == "module test; endmodule"

        # Test 2: Bare module declaration
        text2 = "module my_alu (input logic [3:0] a); endmodule"
        assert "module my_alu" in evaluator.extract_rtl(text2)

    def test_group_advantage_computation(self):
        # Given rewards for a group of 4 completions
        rewards = [0.0, 0.5, 0.8, 1.0]
        advantages = compute_group_advantages(rewards)
        
        assert len(advantages) == 4
        # Higher reward must yield higher advantage
        assert advantages[3] > advantages[2] > advantages[1] > advantages[0]
        # Sum of advantages should be close to 0 (zero-mean normalization)
        assert abs(sum(advantages)) < 0.05

    def test_hardware_evaluator_valid_completion(self, tmp_path):
        evaluator = HardwareRewardEvaluator(
            top_module="mac",
            work_dir=str(tmp_path / "grpo_eval"),
        )
        valid_completion = """
```systemverilog
`timescale 1ns / 1ps
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
```
"""
        reward = evaluator.evaluate_completion(valid_completion)
        # Grounded hardware reward must be > 0 (not length proxy)
        assert reward > 0.5

    def test_hardware_evaluator_invalid_syntax(self, tmp_path):
        evaluator = HardwareRewardEvaluator(
            top_module="mac",
            work_dir=str(tmp_path / "grpo_eval"),
        )
        # Invalid RTL with unbalanced syntax
        broken_completion = "```systemverilog\nmodule mac (invalid syntax !!!\n```"
        reward = evaluator.evaluate_completion(broken_completion)
        # Compile failure must trigger hard gate: reward = 0.0
        assert reward == 0.0


class TestRLTransitionBuilder:
    def test_build_rl_transitions(self, tmp_path):
        builder = TrajectoryDatasetBuilder(output_dir=str(tmp_path / "data"))

        ep = Episode(
            episode_id="ep_test_01",
            task="Design an 8-bit signed MAC.",
            metadata={"model": "Qwen3-4B"},
            steps=[
                TrajectoryStep(
                    step_index=0,
                    state_summary="task initiated",
                    action="RETRIEVE_MEMORY",
                    action_params={"query": "MAC"},
                    observation="retrieved documentation",
                    reward=0.1,
                ),
                TrajectoryStep(
                    step_index=1,
                    state_summary="memory retrieved",
                    action="GENERATE_RTL",
                    action_params={"spec": "mac.sv"},
                    observation="generated mac.sv",
                    reward=1.0,
                ),
                TrajectoryStep(
                    step_index=2,
                    state_summary="rtl generated",
                    action="COMPLETE",
                    action_params={},
                    observation="task completed successfully",
                    reward=1.0,
                ),
            ],
            final_reward=8.0,
            episode_return=2.1,
            success=True,
        )

        transitions = builder.build_rl_transitions([ep])
        assert len(transitions) == 3
        
        # Check transition structure (s_t, a_t, r_t, s_{t+1}, done)
        t0 = transitions[0]
        assert t0.action == "RETRIEVE_MEMORY"
        assert t0.reward == 0.1
        assert not t0.done
        assert t0.next_state["step_index"] == 1

        t2 = transitions[2]
        assert t2.action == "COMPLETE"
        assert t2.done is True
        assert t2.next_state["state_summary"] == "terminal"

        # Check export
        saved_file = builder.save_rl_transitions(transitions, name="test_transitions")
        assert saved_file.endswith(".jsonl")
