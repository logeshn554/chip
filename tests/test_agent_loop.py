"""End-to-end tests for the Agent Loop (with mocked tools)."""

import pytest

from agent.schemas import ActionResult, ActionStatus, ActionType, AgentAction
from agent.action_router import ActionRouter


class TestActionRouter:
    def test_parse_action(self):
        router = ActionRouter()
        action = router.parse_action({
            "thinking": "Need to generate RTL",
            "action": "GENERATE_RTL",
            "params": {"spec": "4-bit adder"},
        })
        assert action.action_type == ActionType.GENERATE_RTL
        assert action.params["spec"] == "4-bit adder"
        assert "generate" in action.thinking.lower()

    def test_parse_unknown_action(self):
        router = ActionRouter()
        action = router.parse_action({"action": "UNKNOWN_ACTION"})
        assert action.action_type == ActionType.UNKNOWN

    def test_parse_missing_action(self):
        router = ActionRouter()
        action = router.parse_action({})
        assert action.action_type == ActionType.UNKNOWN

    @pytest.mark.asyncio
    async def test_execute_complete(self):
        router = ActionRouter()
        action = AgentAction(action_type=ActionType.COMPLETE)
        result = await router.execute(action)
        assert result.status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_unregistered(self):
        router = ActionRouter()
        action = AgentAction(action_type=ActionType.SEARCH)
        result = await router.execute(action)
        assert result.status == ActionStatus.ERROR

    @pytest.mark.asyncio
    async def test_execute_registered_handler(self):
        router = ActionRouter()

        async def mock_search(params):
            return ActionResult(
                action=ActionType.SEARCH,
                status=ActionStatus.SUCCESS,
                output="Found 3 results",
                metrics={"results_count": 3},
            )

        router.register(ActionType.SEARCH, mock_search)

        action = AgentAction(
            action_type=ActionType.SEARCH,
            params={"query": "systolic array"},
        )
        result = await router.execute(action)
        assert result.status == ActionStatus.SUCCESS
        assert "3 results" in result.output

    @pytest.mark.asyncio
    async def test_execute_handler_error(self):
        router = ActionRouter()

        async def failing_handler(params):
            raise ValueError("Tool crashed")

        router.register(ActionType.SIMULATE, failing_handler)

        action = AgentAction(action_type=ActionType.SIMULATE)
        result = await router.execute(action)
        assert result.status == ActionStatus.ERROR
        assert "crashed" in result.output.lower()


class TestPlanSchemas:
    def test_plan_progress(self):
        from agent.schemas import Plan, PlanStep

        plan = Plan(
            task="Test",
            steps=[
                PlanStep(id=1, description="Step 1", step_type="search", status="completed"),
                PlanStep(id=2, description="Step 2", step_type="generate", status="pending"),
                PlanStep(id=3, description="Step 3", step_type="simulate", status="pending"),
            ],
        )
        assert plan.progress == pytest.approx(1 / 3)
        assert not plan.is_complete

    def test_plan_next_step(self):
        from agent.schemas import Plan, PlanStep

        plan = Plan(
            task="Test",
            steps=[
                PlanStep(id=1, description="Step 1", step_type="search", status="completed"),
                PlanStep(id=2, description="Step 2", step_type="generate", status="pending", depends_on=[1]),
                PlanStep(id=3, description="Step 3", step_type="simulate", status="pending", depends_on=[2]),
            ],
        )
        next_step = plan.next_step
        assert next_step.id == 2  # Step 2 is next because its dep (1) is completed

    def test_plan_all_complete(self):
        from agent.schemas import Plan, PlanStep

        plan = Plan(
            task="Test",
            steps=[
                PlanStep(id=1, description="Done", step_type="search", status="completed"),
                PlanStep(id=2, description="Done", step_type="generate", status="completed"),
            ],
        )
        assert plan.is_complete
        assert plan.progress == 1.0
        assert plan.next_step is None
