"""
Data models and schemas for the Self-Evolving Chip Agent.

All structured data types used across the agent, memory, evaluator,
and tool layers are defined here for consistency.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Actions ──────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """Actions the agent can take."""
    # Core 15 actions
    SEARCH_WEB = "SEARCH_WEB"
    RETRIEVE_MEMORY = "RETRIEVE_MEMORY"
    READ_SOURCE = "READ_SOURCE"
    GENERATE_RTL = "GENERATE_RTL"
    GENERATE_TESTBENCH = "GENERATE_TESTBENCH"
    EDIT_RTL = "EDIT_RTL"
    DEBUG = "DEBUG"
    RUN_SIMULATION = "RUN_SIMULATION"
    RUN_TESTS = "RUN_TESTS"
    SYNTHESIZE = "SYNTHESIZE"
    FORMAL_VERIFY = "FORMAL_VERIFY"
    COMPARE_DESIGNS = "COMPARE_DESIGNS"
    SAVE_DESIGN = "SAVE_DESIGN"
    SAVE_EXPERIENCE = "SAVE_EXPERIENCE"
    COMPLETE = "COMPLETE"

    # Backward-compatible aliases
    SEARCH = "SEARCH"
    CREATE_RTL = "CREATE_RTL"
    SIMULATE = "SIMULATE"
    RUN_VERILATOR = "RUN_VERILATOR"
    RUN_COCOTB = "RUN_COCOTB"
    RUN_YOSYS = "RUN_YOSYS"
    INSPECT_MEMORY = "INSPECT_MEMORY"
    OPTIMIZE = "OPTIMIZE"
    FINISH = "FINISH"
    UNKNOWN = "UNKNOWN"



class ActionStatus(str, Enum):
    """Status of an executed action."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"


# ── LLM Response ────────────────────────────────────────────────────

@dataclass
class QwenResponse:
    """Structured response from Qwen3-4B."""
    thinking: str = ""
    content: str = ""
    raw_output: str = ""
    tokens_used: int = 0
    generation_time_s: float = 0.0


@dataclass
class AgentAction:
    """A parsed action from the LLM's JSON response."""
    action_type: ActionType
    params: dict[str, Any] = field(default_factory=dict)
    thinking: str = ""
    raw_response: str = ""


# ── Action Results ───────────────────────────────────────────────────

@dataclass
class ActionResult:
    """Result of executing an action through the tool layer."""
    action: ActionType
    status: ActionStatus
    output: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)  # name -> filepath
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0


# ── Planning ─────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    """A single step in a design plan."""
    id: int
    description: str
    step_type: str  # "search", "generate", "simulate", "synthesize", "optimize"
    depends_on: list[int] = field(default_factory=list)
    status: str = "pending"  # "pending", "in_progress", "completed", "failed", "skipped"
    result: ActionResult | None = None


@dataclass
class Plan:
    """A design plan composed of ordered steps."""
    task: str
    steps: list[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    version: int = 1

    @property
    def next_step(self) -> PlanStep | None:
        """Return the next pending step whose dependencies are all completed."""
        completed_ids = {s.id for s in self.steps if s.status == "completed"}
        for step in self.steps:
            if step.status == "pending":
                if all(dep in completed_ids for dep in step.depends_on):
                    return step
        return None

    @property
    def is_complete(self) -> bool:
        return all(s.status in ("completed", "skipped") for s in self.steps)

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.status in ("completed", "skipped"))
        return done / len(self.steps)


# ── Verification Stages ───────────────────────────────────────────────

class VerificationStage(str, Enum):
    """Explicit hardware verification and physical evaluation stages."""
    RTL_SYNTAX_LINT = "rtl_syntax_lint"
    FUNCTIONAL_VERIFICATION = "functional_verification"
    FORMAL_VERIFICATION = "formal_verification"
    LOGIC_SYNTHESIS = "logic_synthesis"
    AREA_TIMING_POWER = "area_timing_power"


# ── Agent State ──────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Complete state of the agent at a given point in time."""
    task: str
    plan: Plan | None = None
    current_step: int = 0
    iteration: int = 0
    design_files: dict[str, str] = field(default_factory=dict)  # name -> content
    test_files: dict[str, str] = field(default_factory=dict)
    last_action: AgentAction | None = None
    last_result: ActionResult | None = None
    cumulative_reward: float = 0.0      # Sum of step rewards across episode
    best_reward: float = 0.0            # Highest single evaluation reward achieved
    current_design_reward: float = 0.0  # Most recent design evaluation score
    episode_return: float = 0.0         # Undiscounted cumulative return for RL
    verification_stages: dict[str, Any] = field(default_factory=dict)  # Explicit stage-based tracking
    history: list[dict[str, Any]] = field(default_factory=list)


# ── EDA Tool Results ─────────────────────────────────────────────────

@dataclass
class CompileResult:
    """Result of compiling SystemVerilog with Verilator."""
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output: str = ""
    binary_path: str | None = None


@dataclass
class LintResult:
    """Result of linting SystemVerilog with Verilator."""
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Result of running a simulation."""
    success: bool
    tests_total: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    errors: list[str] = field(default_factory=list)
    output: str = ""
    vcd_path: str | None = None
    duration_s: float = 0.0

    @property
    def pass_rate(self) -> float:
        if self.tests_total == 0:
            return 0.0
        return self.tests_passed / self.tests_total


@dataclass
class SynthesisResult:
    """Result of synthesis with Yosys."""
    success: bool
    cell_count: int = 0
    wire_count: int = 0
    area_estimate: float = 0.0
    critical_path_ns: float = 0.0
    lut_count: int = 0
    ff_count: int = 0
    bram_count: int = 0
    output: str = ""
    netlist_path: str | None = None
    errors: list[str] = field(default_factory=list)


# ── Evaluation & Reward ──────────────────────────────────────────────

@dataclass
class FunctionalScore:
    """Functional correctness scores."""
    compile_pass: bool = False
    lint_pass: bool = False
    lint_warnings: int = 0
    test_pass_rate: float = 0.0
    tests_total: int = 0
    tests_passed: int = 0

    @property
    def score(self) -> float:
        """Normalized functional score [0, 1]."""
        s = 0.0
        if self.compile_pass:
            s += 0.3
        if self.lint_pass:
            s += 0.1
        s += 0.6 * self.test_pass_rate
        return s


@dataclass
class SynthesisScore:
    """Synthesis quality scores."""
    area_score: float = 0.0       # 0-1, lower area = higher score
    timing_score: float = 0.0     # 0-1, faster = higher score
    power_score: float = 0.0      # 0-1, lower power = higher score
    synthesizable: bool = False

    @property
    def score(self) -> float:
        if not self.synthesizable:
            return 0.0
        return (self.area_score + self.timing_score + self.power_score) / 3.0


@dataclass
class EvaluationResult:
    """Combined evaluation result."""
    functional: FunctionalScore = field(default_factory=FunctionalScore)
    synthesis: SynthesisScore = field(default_factory=SynthesisScore)
    reward: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ── Trajectory ───────────────────────────────────────────────────────

@dataclass
class TrajectoryStep:
    """A single step in a trajectory episode."""
    step_index: int = 0
    action: str = ""
    state_summary: str = ""
    action_params: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    reward: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Episode:
    """A complete trajectory episode from task start to completion."""
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task: str = ""
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_reward: float = 0.0
    best_reward: float = 0.0
    episode_return: float = 0.0
    total_iterations: int = 0
    success: bool = False
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        if self.completed_at is None:
            return time.time() - self.started_at
        return self.completed_at - self.started_at


# ── Memory Documents ─────────────────────────────────────────────────

@dataclass
class Document:
    """A document chunk from the memory system."""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # similarity score from retrieval
    source: str = ""
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


# ── RTL Module ───────────────────────────────────────────────────────

@dataclass
class RTLModule:
    """Represents a SystemVerilog module."""
    name: str
    code: str
    ports: list[dict[str, str]] = field(default_factory=list)
    parameters: list[dict[str, str]] = field(default_factory=list)
    filepath: str | None = None
    testbench: str | None = None
    description: str = ""


# ── Search Results ───────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A web search result."""
    title: str
    url: str
    snippet: str
    content: str = ""
    relevance: float = 0.0
