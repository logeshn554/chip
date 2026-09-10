"""
Data models and schemas for the Self-Evolving Chip Agent.

All structured data types used across the agent, memory, evaluator,
and tool layers are defined here for consistency.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any, Optional
import uuid
from dataclasses import dataclass, field
from enum import Enum


# ── Actions ──────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """Actions the agent can take."""
    # Core 17 actions
    SEARCH_WEB = "SEARCH_WEB"
    RETRIEVE_MEMORY = "RETRIEVE_MEMORY"
    READ_SOURCE = "READ_SOURCE"
    PROPOSE_ARCHITECTURE = "PROPOSE_ARCHITECTURE"
    COMPARE_ARCHITECTURES = "COMPARE_ARCHITECTURES"
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
    request: dict[str, Any] = field(default_factory=dict)
    reward_contribution: float = 0.0


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


# ── Architecture Search ──────────────────────────────────────────────

@dataclass
class HardwareArchitectureCandidate:
    """A hardware architecture candidate in the architectural search space."""
    architecture_id: str
    task_id: str
    parent_architecture_id: Optional[str] = None
    generation: int = 0
    mutation_type: str = "initial"
    specification_version: str = "1.0"
    architecture_description: str = ""
    components: list[Any] = field(default_factory=list)  # list[ComponentEvidence]
    interfaces: list[str] = field(default_factory=list)
    memory_hierarchy: dict[str, Any] = field(default_factory=dict)
    compute_units: dict[str, Any] = field(default_factory=dict)
    accelerator_structure: dict[str, Any] = field(default_factory=dict)
    datapath_structure: str = "direct"
    pipeline_depth: int = 1
    parallelism: int = 1
    memory_organization: str = "registers"
    buffering_strategy: str = "single_buffer"
    arithmetic_strategy: str = "standard_signed"
    interface_strategy: str = "valid_ready"
    rtl_implementation: str = ""
    estimated_resource_requirements: dict[str, Any] = field(default_factory=dict)
    actual_synthesis_metrics: dict[str, Any] = field(default_factory=dict)
    estimated_constraints: dict[str, Any] = field(default_factory=dict)
    measured_constraints: dict[str, Any] = field(default_factory=dict)
    verification_status: str = "unverified"  # "unverified", "passed", "failed"
    reward: float = 0.0
    trajectory_id: Optional[str] = None
    is_hypothesis: bool = True
    validation_status: str = "UNVERIFIED"  # "UNVERIFIED", "VALIDATED_ARCHITECTURE", "FAILED_HYPOTHESIS"
    custom_chip: bool = False
    chip_generation: int = 0
    rejection_reasons: list[str] = field(default_factory=list)
    constraint_states: dict[str, str] = field(default_factory=dict)
    pareto_metrics: dict[str, float] = field(default_factory=dict)
    verification_level: str = "SIMULATION_VALID"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def datapath(self) -> str:
        return self.datapath_structure

    @property
    def rejection_reason(self) -> str:
        return "; ".join(self.rejection_reasons) if self.rejection_reasons else ""

    @rejection_reason.setter
    def rejection_reason(self, val: str) -> None:
        if val and val not in self.rejection_reasons:
            self.rejection_reasons.append(val)


# ── Trajectory ───────────────────────────────────────────────────────

@dataclass
class TrajectoryStep:
    """A single step in a trajectory episode with full provenance."""
    step_index: int = 0
    action: str = ""
    state_summary: str = ""
    action_params: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    tool_output: str = ""
    reward: float = 0.0
    next_state: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    duration_s: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Episode:
    """A complete trajectory episode from task start to completion with full provenance."""
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
    # Rigorous experiment and model provenance tracking
    experiment_id: str = ""
    model_id: str = "qwen3:4b"
    model_version: str = "1.0"
    adapter_version: str = ""
    task_id: str = ""
    benchmark_version: str = "1.0"
    seed: int = 42
    generation: int = 0
    architecture_id: str = ""
    parent_architecture_id: str = ""
    final_rtl: str = ""
    final_verification: dict[str, Any] = field(default_factory=dict)
    final_metrics: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    success_reason: str = ""
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


# ── Physical Envelope & Hardware Target Constraints ─────────────────

@dataclass
class DevicePhysicalEnvelope:
    """Configurable physical, power, thermal, and performance envelope for the target device."""
    target_name: str = "Portable Independent AI Device"
    enclosure_length_mm: float = 100.0
    enclosure_width_mm: float = 30.0
    enclosure_height_mm: float = 12.0
    max_power_w: float = 5.0
    max_junction_temp_c: float = 85.0
    ambient_temp_c: float = 30.0
    thermal_resistance_c_per_w: float = 10.0  # Passive aluminum heat dissipation theta_ja
    min_ram_gb: float = 8.0
    min_storage_gb: float = 256.0
    interface_type: str = "USB-C"
    target_model_name: str = "Qwen3-4B-INT4"
    target_model_params_b: float = 4.0
    weight_bits: int = 4
    min_tokens_per_sec: float = 15.0
    max_latency_ms: float = 200.0

    @property
    def max_pcb_area_mm2(self) -> float:
        """Usable double-sided PCB area with 2mm wall clearance (approx 90% usable per side)."""
        usable_l = max(0.0, self.enclosure_length_mm - 4.0)
        usable_w = max(0.0, self.enclosure_width_mm - 4.0)
        return usable_l * usable_w * 1.8  # Double-sided layout with pass-through margins


@dataclass
class PhysicalProjectionMetrics:
    """Physical hardware projections derived from digital RTL synthesis and system component modeling."""
    digital_die_area_mm2: float = 0.0
    asic_package_area_mm2: float = 0.0
    dram_package_area_mm2: float = 0.0
    storage_package_area_mm2: float = 0.0
    pmic_passives_area_mm2: float = 0.0
    usbc_connector_area_mm2: float = 0.0
    total_component_area_mm2: float = 0.0
    total_pcb_area_mm2: float = 0.0
    asic_power_w: float = 0.0
    dram_power_w: float = 0.0
    system_overhead_power_w: float = 0.0
    total_device_power_w: float = 0.0
    estimated_junction_temp_c: float = 0.0
    effective_memory_bandwidth_gbps: float = 0.0
    compute_tops: float = 0.0
    achievable_tokens_per_sec: float = 0.0
    bottleneck: str = "balanced"  # "memory_bandwidth", "compute", "thermal_throttled"


@dataclass
class ConstraintCheckResult:
    """Detailed result of checking an architecture candidate against the physical envelope."""
    passed: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    projections: PhysicalProjectionMetrics = field(default_factory=PhysicalProjectionMetrics)
    envelope_name: str = ""


# ── Explicit Constraint States & Feasibility Labels ───────────────────

class ConstraintState(str, Enum):
    """Explicit evaluation status for every hardware and physical constraint."""
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class FeasibilityLabel(str, Enum):
    """Engineering label for component and physical fit feasibility."""
    FEASIBLE_ESTIMATE = "FEASIBLE_ESTIMATE"
    INFEASIBLE_ESTIMATE = "INFEASIBLE_ESTIMATE"
    UNKNOWN = "UNKNOWN"


class VerificationLevel(str, Enum):
    """Evidence tier for hardware architecture claims."""
    SIMULATION_VALID = "SIMULATION_VALID"
    SYNTHESIS_VALID = "SYNTHESIS_VALID"
    PHYSICALLY_ESTIMATED = "PHYSICALLY_ESTIMATED"
    PHYSICALLY_VERIFIED = "PHYSICALLY_VERIFIED"
    REAL_WORLD_COMPONENT_VERIFIED = "REAL_WORLD_COMPONENT_VERIFIED"
    SILICON_VERIFIED = "SILICON_VERIFIED"


# ── Real-World Component Evidence ────────────────────────────────────

@dataclass
class ComponentEvidence:
    """A real-world commercial hardware component backed by external web/datasheet evidence."""
    component_id: str
    manufacturer: str
    part_number: str
    category: str  # "dram", "storage", "pmic", "accelerator", "usb_controller", "cpu", "discrete"
    datasheet_url: Optional[str] = None
    source_urls: list[str] = field(default_factory=list)
    package: str = "unknown"
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    mass_g: Optional[float] = None
    power_w: Optional[float] = None
    voltage_v: Optional[float] = None
    interface: str = "unknown"
    memory_capacity_gb: Optional[float] = None
    compute_capability_tops: Optional[float] = None
    temperature_range: Optional[str] = None
    availability_status: str = "active"
    source_date: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    confidence: float = 0.85
    extracted_specification: dict[str, Any] = field(default_factory=dict)
    raw_evidence: str = ""

    @property
    def footprint_area_mm2(self) -> float:
        if self.length_mm is not None and self.width_mm is not None:
            return self.length_mm * self.width_mm
        return 0.0


# ── Target Specification ─────────────────────────────────────────────

@dataclass
class TargetSpecification:
    """Configurable system-level target specification for an independent portable AI computer."""
    target_name: str = "Portable Independent AI Computer"
    max_length_mm: float = 100.0
    max_width_mm: float = 30.0
    max_height_mm: float = 12.0
    max_volume_mm3: float = 36000.0  # 100 * 30 * 12
    max_power_w: float = 5.0
    max_temperature_c: float = 85.0
    min_ram_gb: float = 8.0
    min_storage_gb: float = 256.0
    target_model: str = "Qwen3-4B"
    target_model_quantization: str = "INT4"
    min_tokens_per_second: float = 15.0
    max_latency_ms: float = 200.0
    host_interfaces: list[str] = field(default_factory=lambda: ["USB-C"])
    host_platforms: list[str] = field(default_factory=lambda: ["Android", "Linux", "Windows"])
    max_cost: Optional[float] = None
    process_node: str = "28nm"
    package_constraints: dict[str, Any] = field(default_factory=dict)
    ambient_temp_c: float = 30.0
    thermal_resistance_c_per_w: float = 10.0

    # Backward-compatible property bridges with DevicePhysicalEnvelope
    @property
    def enclosure_length_mm(self) -> float:
        return self.max_length_mm

    @property
    def enclosure_width_mm(self) -> float:
        return self.max_width_mm

    @property
    def enclosure_height_mm(self) -> float:
        return self.max_height_mm

    @property
    def max_junction_temp_c(self) -> float:
        return self.max_temperature_c

    @property
    def min_tokens_per_sec(self) -> float:
        return self.min_tokens_per_second

    @property
    def target_model_name(self) -> str:
        return f"{self.target_model}-{self.target_model_quantization}"

    @property
    def target_model_params_b(self) -> float:
        return 4.0 if "4b" in self.target_model.lower() else 1.5

    @property
    def weight_bits(self) -> int:
        return 4 if "int4" in self.target_model_quantization.lower() else 8

    @property
    def interface_type(self) -> str:
        return self.host_interfaces[0] if self.host_interfaces else "USB-C"

    @property
    def max_pcb_area_mm2(self) -> float:
        usable_l = max(0.0, self.max_length_mm - 4.0)
        usable_w = max(0.0, self.max_width_mm - 4.0)
        return usable_l * usable_w * 1.8


