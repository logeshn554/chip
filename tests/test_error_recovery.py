"""Tests for error recovery, malformed JSON handling, and resilience."""

import pytest
from agent.qwen import QwenClient
from agent.action_router import ActionRouter
from agent.schemas import AgentAction, ActionType, ActionStatus


class TestErrorRecovery:
    def test_json_repair_trailing_comma(self):
        malformed = '{"action": "GENERATE_RTL", "params": {"file": "mac.sv",},}'
        parsed = QwenClient.extract_json(malformed)
        assert parsed.get("action") == "GENERATE_RTL"
        assert parsed.get("params", {}).get("file") == "mac.sv"

    def test_json_repair_with_comments(self):
        with_comments = """
        // This is reasoning comment
        {
            "action": "RUN_SIMULATION",
            "params": {"sources": ["mac.sv"]} /* inline comment */
        }
        """
        parsed = QwenClient.extract_json(with_comments)
        assert parsed.get("action") == "RUN_SIMULATION"

    def test_unparseable_json_returns_error_dict(self):
        unparseable = "Just plain words without any json structure."
        result = QwenClient.extract_json(unparseable)
        assert "error" in result
        assert result["error"] == "JSON_PARSE_ERROR"
        # Must not be converted to COMPLETE

    def test_unknown_action_does_not_complete_task(self):
        router = ActionRouter()
        action = router.parse_action({"action": "FLY_TO_MARS", "params": {}})
        assert action.action_type == ActionType.UNKNOWN
        # Must NOT be ActionType.COMPLETE!

    @pytest.mark.asyncio
    async def test_unknown_action_returns_error_result(self):
        router = ActionRouter()
        action = router.parse_action({"action": "DO_SOMETHING_INVALID"})
        res = await router.execute(action)
        assert res.status == ActionStatus.ERROR
        assert "Unknown action" in res.output
