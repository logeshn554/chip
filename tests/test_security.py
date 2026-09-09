"""Tests for security sandboxing and anti-reward hacking restrictions."""

import pytest
from agent.action_router import is_path_safe_for_write, ActionRouter
from agent.schemas import AgentAction, ActionType, ActionStatus


class TestSecurityRestrictions:
    def test_forbidden_evaluator_writes(self):
        assert not is_path_safe_for_write("evaluator/reward.py")
        assert not is_path_safe_for_write("evaluator/metrics.py")
        assert not is_path_safe_for_write("./evaluator/reward.py")

    def test_forbidden_test_writes(self):
        assert not is_path_safe_for_write("tests/test_evaluator.py")
        assert not is_path_safe_for_write("benchmarks/tests/hidden_test.py")

    def test_forbidden_config_writes(self):
        assert not is_path_safe_for_write("configs/agent.yaml")
        assert not is_path_safe_for_write("configs/prompts.yaml")

    def test_directory_traversal_rejection(self):
        assert not is_path_safe_for_write("../../etc/passwd")
        assert not is_path_safe_for_write("designs/../../evaluator/reward.py")

    def test_allowed_design_workspace_writes(self):
        assert is_path_safe_for_write("designs/mac/v1.0/mac.sv")
        assert is_path_safe_for_write("rtl/generated/mac.sv")
        assert is_path_safe_for_write("sim_build/test.sv")
        assert is_path_safe_for_write("mac.sv")

    @pytest.mark.asyncio
    async def test_router_blocks_protected_path_write(self):
        router = ActionRouter()
        action = AgentAction(
            action_type=ActionType.GENERATE_RTL,
            params={"filename": "evaluator/fake_reward.py", "code": "def hack(): pass"},
        )
        result = await router.execute(action)
        assert result.status == ActionStatus.ERROR
        assert "Security violation" in result.output
