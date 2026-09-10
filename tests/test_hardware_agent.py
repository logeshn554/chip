"""
Unit tests for the autonomous HardwareAgent and its 12-action execution loop.
"""

import os
import pytest
from agent.agent import HardwareAgent
from agent.state import AgentState
from llm.qwen import OllamaQwenClient
from evaluator.reward import RewardEngine


@pytest.mark.asyncio
async def test_hardware_agent_actions(tmp_path):
    client = OllamaQwenClient(mock_mode=True)
    agent = HardwareAgent(llm=client, work_dir=str(tmp_path / "rtl"))

    state = AgentState(task="Design an 8-bit signed MAC")

    # 1. CREATE_RTL
    sv_code = """
    module mac (
        input logic clk, rst_n, en, clr,
        input logic signed [7:0] a, b,
        output logic signed [31:0] out,
        output logic valid
    );
        logic signed [15:0] product;
        logic signed [31:0] acc_reg;
        always_comb product = a * b;
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) acc_reg <= '0;
            else if (en) acc_reg <= acc_reg + {{16{product[15]}}, product};
        end
        assign out = acc_reg;
        assign valid = 1'b1;
    endmodule
    """
    res_create = await agent._execute_action(
        "CREATE_RTL",
        {"filename": "mac.sv", "code": sv_code},
        state,
    )
    assert res_create["status"] == "success"
    assert os.path.exists(os.path.join(str(tmp_path / "rtl"), "mac.sv"))

    # 2. RUN_VERILATOR
    res_ver = await agent._execute_action("RUN_VERILATOR", {"sources": ["mac.sv"]}, state)
    assert res_ver["stage"] == "verilator"
    assert res_ver["status"] == "passed"

    # 3. RUN_COCOTB
    res_sim = await agent._execute_action("RUN_COCOTB", {"rtl_file": "mac.sv"}, state)
    assert res_sim["stage"] == "cocotb"
    assert res_sim["status"] == "passed"

    # 4. RUN_YOSYS
    res_synth = await agent._execute_action("RUN_YOSYS", {"file_path": "mac.sv"}, state)
    assert res_synth["stage"] == "yosys"
    cells = res_synth.get("cells", res_synth.get("heuristic_cell_guess", 0))
    assert cells > 0

    # 5. RETRIEVE_MEMORY
    res_mem = await agent._execute_action("RETRIEVE_MEMORY", {"query": "signed MAC", "memory_type": "knowledge"}, state)
    assert res_mem["status"] == "success"

    # 6. SAVE_DESIGN
    res_save = await agent._execute_action("SAVE_DESIGN", {"module_name": "mac", "version": "v1.0"}, state)
    assert res_save["status"] == "success"

    # 7. FINISH
    res_finish = await agent._execute_action("FINISH", {}, state)
    assert res_finish["status"] == "success"
    assert state.is_finished is True


@pytest.mark.asyncio
async def test_structured_failure_handling(tmp_path):
    """Test that buggy code produces a structured failure and triggers INSPECT_ERROR."""
    client = OllamaQwenClient(mock_mode=True)
    agent = HardwareAgent(llm=client, work_dir=str(tmp_path / "rtl"))

    state = AgentState(task="Design an 8-bit signed MAC")

    # Buggy code: missing endmodule
    buggy_sv = """
    module mac (
        input logic clk
    );
    // Missing endmodule
    """
    await agent._execute_action("CREATE_RTL", {"filename": "mac.sv", "code": buggy_sv}, state)

    # RUN_VERILATOR should produce structured failure
    res_ver = await agent._execute_action("RUN_VERILATOR", {"sources": ["mac.sv"]}, state)
    assert res_ver["status"] == "failed"
    assert res_ver["stage"] == "verilator"
    assert "line" in res_ver
    assert "file" in res_ver
    assert state.current_error is not None

    # INSPECT_ERROR should analyze it
    res_inspect = await agent._execute_action("INSPECT_ERROR", {}, state)
    assert res_inspect["status"] == "success"


def test_grounded_reward_engine():
    engine = RewardEngine()

    # Full success
    res_full = engine.compute_v1_reward(
        compile_success=True,
        all_functional_tests_pass=True,
        synthesis_success=True,
        lint_clean=True,
    )
    assert res_full.total_reward == 8.0  # 1 + 5 + 1 + 1
    assert res_full.normalized_reward == 1.0
    assert res_full.is_valid_hardware is True

    # Failed functional tests: must NOT receive high reward
    res_fail = engine.compute_v1_reward(
        compile_success=True,
        all_functional_tests_pass=False,
        synthesis_success=True,
        lint_clean=True,
    )
    assert res_fail.total_reward <= 2.5
    assert res_fail.is_valid_hardware is False
