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

    def test_grpo_trainer_step_advantages(self, tmp_path):
        from learning.grpo import GRPOTrainer
        trainer = GRPOTrainer({"top_module": "mac", "output_dir": str(tmp_path / "models")})
        completions = [
            "```systemverilog\nmodule mac (invalid syntax\n```",
            "non-code text here",
        ]
        rewards, advantages = trainer.compute_step_advantages(completions)
        assert len(rewards) == 2
        assert len(advantages) == 2
        assert all(r == 0.0 for r in rewards)

    def test_dynamic_benchmark_resolution(self):
        evaluator = HardwareRewardEvaluator()
        
        # Resolve by benchmark_task_id
        task_not = evaluator.resolve_benchmark_task(task_id="L1_NOT_GATE")
        assert task_not is not None
        assert task_not.top_module == "not_gate"
        assert len(task_not.public_test_cases) > 0

        # Resolve by prompt keyword
        task_alu = evaluator.resolve_benchmark_task(prompt="Implement the 4-bit ALU benchmark L2_ALU_4BIT")
        assert task_alu is not None
        assert task_alu.top_module == "alu"

        # Resolve by module name in completion
        task_mux = evaluator.resolve_benchmark_task(completion="module mux_2to1 (input logic d0, d1, sel, output logic y); assign y = sel ? d1 : d0; endmodule")
        assert task_mux is not None
        assert task_mux.task_id == "L1_MUX2TO1"

    def test_evaluator_evaluate_non_mac_benchmark(self, tmp_path):
        evaluator = HardwareRewardEvaluator(
            work_dir=str(tmp_path / "grpo_not_gate"),
        )
        not_gate_completion = """
```systemverilog
`timescale 1ns / 1ps
module not_gate (
    input  logic a,
    output logic y
);
    assign y = ~a;
endmodule
```
"""
        # Grounded hardware evaluation for Level 1 NOT gate
        reward = evaluator.evaluate_completion(not_gate_completion, benchmark_task_id="L1_NOT_GATE")
        assert reward > 0.5

    def test_parallel_batch_evaluation(self, tmp_path):
        evaluator = HardwareRewardEvaluator(
            work_dir=str(tmp_path / "grpo_parallel"),
            max_concurrency=4,
        )
        valid_not = "module not_gate (input logic a, output logic y); assign y = ~a; endmodule"
        invalid_not = "module not_gate (syntax error !!!"
        completions = [valid_not, invalid_not, valid_not]
        prompts = ["Design L1_NOT_GATE", "Design L1_NOT_GATE", "Design L1_NOT_GATE"]

        # Run parallel batch evaluation
        rewards = evaluator.evaluate_batch(completions, prompts=prompts, max_concurrency=2)
        assert len(rewards) == 3
        assert rewards[0] > 0.5
        assert rewards[1] == 0.0
        assert rewards[2] > 0.5


class TestRLTransitionBuilder:
    def test_build_rl_transitions(self, tmp_path):
        builder = TrajectoryDatasetBuilder(output_dir=str(tmp_path / "data"))

        ep = Episode(
            episode_id="ep_test_01",
            task="Design an 8-bit signed MAC.",
            metadata={"model": "Qwen-14B"},
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

    def test_build_grpo_prompt_dataset(self, tmp_path):
        builder = TrajectoryDatasetBuilder(output_dir=str(tmp_path / "data"))
        episodes = [
            Episode(
                episode_id="ep_not_01",
                task="Design an inverter NOT gate [L1_NOT_GATE].",
                metadata={"benchmark_task_id": "L1_NOT_GATE", "top_module": "not_gate"},
                steps=[
                    TrajectoryStep(
                        step_index=0,
                        state_summary="rtl generation",
                        action="GENERATE_RTL",
                        action_params={"code": "module not_gate (input a, output y); assign y = ~a; endmodule"},
                        reward=1.0,
                    )
                ],
                final_reward=1.0,
                episode_return=1.0,
                success=True,
            ),
            Episode(
                episode_id="ep_alu_01",
                task="Design a 4-bit ALU [L2_ALU_4BIT].",
                metadata={"benchmark_task_id": "L2_ALU_4BIT", "top_module": "alu_4bit"},
                steps=[
                    TrajectoryStep(
                        step_index=0,
                        state_summary="rtl generation",
                        action="GENERATE_RTL",
                        action_params={"code": "module alu_4bit; endmodule"},
                        reward=0.8,
                    )
                ],
                final_reward=0.8,
                episode_return=0.8,
                success=True,
            ),
        ]

        grpo_prompts = builder.build_grpo_prompt_dataset(episodes)
        assert len(grpo_prompts) == 2
        assert "prompt" in grpo_prompts[0]
        assert grpo_prompts[0]["benchmark_task_id"] == "L1_NOT_GATE"
        assert grpo_prompts[0]["top_module"] == "not_gate"
        assert grpo_prompts[1]["benchmark_task_id"] == "L2_ALU_4BIT"

        saved_path = builder.save_grpo_prompts(grpo_prompts, name="test_grpo_prompts")
        assert saved_path.endswith(".jsonl")

        # Test in-memory Hugging Face Dataset conversion
        hf_ds = builder.to_hf_dataset(grpo_prompts)
        assert len(hf_ds) == 2
        assert "prompt" in hf_ds.column_names
        assert "benchmark_task_id" in hf_ds.column_names


class TestTRLModernAPICompatibility:
    def test_normalize_completion_text(self):
        from learning.grpo import normalize_completion_text
        
        # 1. Plain string
        assert normalize_completion_text("module test; endmodule") == "module test; endmodule"

        # 2. Single message dict
        assert normalize_completion_text({"content": "module test; endmodule"}) == "module test; endmodule"

        # 3. Conversational message list (current TRL format)
        trl_msg_list = [
            {"role": "user", "content": "Write an ALU"},
            {"role": "assistant", "content": "```systemverilog\nmodule alu; endmodule\n```"}
        ]
        norm = normalize_completion_text(trl_msg_list)
        assert "module alu" in norm

    def test_trl_conversational_completions_batch_evaluation(self, tmp_path):
        """Test that evaluate_batch processes modern TRL conversational format and kwargs task routing."""
        evaluator = HardwareRewardEvaluator(work_dir=str(tmp_path / "trl_eval"))

        # Conversational completion objects returned by modern TRL generate loop
        completions = [
            [{"role": "assistant", "content": "```systemverilog\nmodule not_gate (input logic a, output logic y); assign y = ~a; endmodule\n```"}],
            [{"role": "assistant", "content": "syntax error !!!"}],
        ]
        prompts = [
            [{"role": "user", "content": "Design an inverter NOT gate [L1_NOT_GATE]"}],
            [{"role": "user", "content": "Design an inverter NOT gate [L1_NOT_GATE]"}],
        ]

        # Extra dataset columns passed by TRL as kwargs:
        rewards = evaluator.evaluate_batch(
            completions,
            prompts=prompts,
            benchmark_task_id=["L1_NOT_GATE", "L1_NOT_GATE"],
        )
        assert len(rewards) == 2
        assert rewards[0] > 0.5  # Valid SystemVerilog correctly extracted and verified
        assert rewards[1] == 0.0  # Syntax error penalized

    def test_curriculum_environment_factory(self, tmp_path):
        from learning.grpo import make_curriculum_environment_factory
        factory = make_curriculum_environment_factory(work_dir=str(tmp_path / "env_factory"))

        env_not = factory(benchmark_task_id="L1_NOT_GATE")
        assert env_not.target_module == "not_gate"
        assert "L1_NOT_GATE" in env_not.task

        env_alu = factory(benchmark_task_id="L2_ALU_4BIT")
        assert env_alu.target_module == "alu"
        assert "L2_ALU_4BIT" in env_alu.task
