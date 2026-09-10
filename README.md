# Self-Evolving Hardware Design Agent 🧠⚡

A research system for autonomous hardware architecture search, synthesizable SystemVerilog design, and physical constraint satisfaction driven by small local open-weight LLMs (**Qwen3-4B** via **Ollama**, **Transformers**, or **vLLM**).

> ### 🎯 Top-Level Research Objective
> **“Autonomously discover a hardware architecture that satisfies a configurable physical envelope and AI-performance specification, using LLM-guided architecture evolution and externally grounded EDA rewards.”**
>
> The agent does **not** stop after merely producing compilable RTL. It enters an open-ended architecture evolution loop and terminates only when the candidate design satisfies all mandatory functional, physical, performance, power, thermal, and interface constraints—or when a declared compute/search budget is exhausted.

---

## 1. Physical Constraint Satisfaction & Hardware Evolution Loop

```
TARGET: "Design hardware for an independent portable AI device"
Constraints (Configurable Envelope: configs/target_device_envelope.yaml):
├── Maximum Physical Size (e.g. 100 mm × 30 mm × 12 mm enclosure, double-sided PCB)
├── Maximum Power (e.g. 5.0 W USB-C bus powered limit)
├── Maximum Thermal Limit (e.g. 85.0 °C max junction temp with passive cooling)
├── Minimum Memory & Storage (e.g. 8 GB LPDDR4x/5 RAM, 256 GB UFS Flash)
├── Connectivity Interface (e.g. USB-C 3.2 Gen 2)
├── Target AI Workload (e.g. Qwen3-4B-INT4 quantized parameters)
└── Required Compute Throughput (e.g. >= 12-15 tokens/sec interactive decode)
       ↓
[LLM Proposes Architecture Hypothesis] (Datapath, Pipeline Depth, Parallelism, Memory)
       ↓
[Generate Synthesizable SystemVerilog]
       ↓
[EDA Verification] (Verilator Lint/Compile, Cocotb Functional Vectors, SymbiYosys Formal)
       ↓
[Logic Synthesis & PPA] (Yosys Standard Cell Synthesis, Gate Counts, Fmax)
       ↓
[Physical Constraint & System Integration Model] (Die Size, PCB Geometry, Thermal, BW)
       ↓
    DOES DESIGN PASS ALL CONSTRAINTS?
       ├── NO  ──> Analyze failure (size? power? thermal? throughput?)
       │           Record structured failure context in Experience Store
       │           Generate constraint-guided architecture mutations
       │           Repeat loop ────────┐
       │                               │
       └── YES ──> [WINNING ARCHITECTURE PROMOTED & PERSISTED]
```

### Physical Reality & Implementation Modeling (`evaluator/physical_envelope.py`)
RTL simulation alone cannot prove that a real pendrive- or pocket-sized device will fit or dissipate acceptable power. The system integrates a grounded **Physical Constraint & System Integration Model**:
- **Digital Die Area**: Scaled from synthesized cell counts and process node gate density (28nm / 14nm / 7nm).
- **Package Footprints & PCB Area**: Aggregates flip-chip BGA, 8GB LPDDR4x/5 package (144 mm²), 256GB UFS BGA (149.5 mm²), PMIC/passives, and USB-C receptacle against usable double-sided PCB area.
- **System Power Modeling**: Sums dynamic ASIC switching, static leakage, active DRAM I/O, storage bursts, and PMIC loss.
- **Passive Thermal Model**: Grounded silicon junction temperature $T_j = T_{\text{ambient}} + P_{\text{total}} \times \theta_{ja}$ (e.g. 10 °C/W aluminum body).
- **AI Token Throughput Bottleneck**: Evaluates both compute-bound FLOPs and memory-bandwidth bound decode: $\min(BW_{\text{eff}} / \text{Size}_{\text{token}}, TOPS / \text{FLOPs}_{\text{token}})$.
- **Empirical Evolution**: Progresses across generations (e.g. Gen 1 fails throughput $\rightarrow$ Gen 2 mutates pipeline $\rightarrow$ Gen 3 scales parallel lanes $\rightarrow$ Gen 3 satisfies all constraints!).

---

## 2. Target Architecture & Data Flow

```
┌────────────────────────────────────────────────────────┐
│                   External Policy                      │
│     (Qwen3-4B / PPO / GRPO Policy Generator)           │
└──────────────────────────┬─────────────────────────────┘
                           │ selects typed action
                           ▼
┌────────────────────────────────────────────────────────┐
│             Gymnasium HardwareDesignEnv                │
│    - Sandbox execution & action routing                │
│    - Calls EDA tools (Verilator, Cocotb, Yosys, SBY)   │
│    - Evaluates grounded reward & verification state    │
└──────────────────────────┬─────────────────────────────┘
                           │ (obs, step_reward, done, info)
                           ▼
┌────────────────────────────────────────────────────────┐
│           Trajectory & Advantage Engine                │
│    - Trajectory Store (JSONL + SFT transitions)        │
│    - Group Advantage Normalization (GRPO)              │
│    - TRL GRPOTrainer / Policy Gradient update          │
└────────────────────────────────────────────────────────┘
```

> **Architectural Boundary**:
> `HardwareDesignEnv` is deliberately designed as the underlying Gymnasium execution environment. It does not replace or embed the agent policy. Qwen3-4B (or any external RL policy, PPO, or GRPO actor) acts as the decision-making policy driving the environment.

---

## 2. Supported Action System (15 Predefined Typed Actions)

All agent operations are mediated strictly by `ActionRouter` (`agent/action_router.py`). **Arbitrary shell execution generated by the LLM is strictly prohibited.** File writes are sandbox-restricted to the active design workspace (`./rtl/generated/`, `./designs/`), completely protecting evaluator code, testbenches, and configuration files from tampering.

| Action Type | Description | Sandbox Restrictions |
|---|---|---|
| `PROPOSE_ARCHITECTURE` | Explore microarchitectural trade-offs (pipeline, datapath, memory) | Level A structured search |
| `COMPARE_ARCHITECTURES` | Compare area, latency, and throughput across candidate architectures | Pareto frontier ranking |
| `SEARCH_WEB` | Targeted web research via ScrapeGraphAI adapter | Domain allowlist, token bounded |
| `RETRIEVE_MEMORY` | Query Knowledge, Experience, or Design memories | Bounded ChromaDB retrieval |
| `READ_SOURCE` | Inspect existing RTL or reference files | Read-only access |
| `GENERATE_RTL` | Create new synthesizable SystemVerilog module | Allowed only in `./rtl/generated/` |
| `GENERATE_TESTBENCH`| Create non-heldout verification harness | Allowed only in `./sim/` |
| `EDIT_RTL` | Patch or correct previous SystemVerilog code | Allowed only in `./rtl/generated/` |
| `DEBUG` | Analyze simulator or compiler error diagnostics | Safe inspection |
| `RUN_SIMULATION` | Run Verilator or simulator compilation | Dedicated tool wrapper |
| `RUN_TESTS` | Execute Cocotb functional verification tests | Dedicated tool wrapper |
| `SYNTHESIZE` | Run Yosys logic synthesis for gate counts | Dedicated tool wrapper |
| `FORMAL_VERIFY` | Run SymbiYosys bounded model checking (`sby`) | Dedicated tool wrapper |
| `COMPARE_DESIGNS` | Compare area, latency, and timing of versions | Truthful comparative metrics |
| `SAVE_DESIGN` | Version design to Git repository and Design Store | Tagged commit & metadata |
| `SAVE_EXPERIENCE` | Store error-to-fix experience in ChromaDB | JSON metadata indexed |
| `COMPLETE` | Explicitly signal milestone task completion | Terminates episode |

---

## 3. Two-Level Reasoning & Hardware Architecture Search

Hardware design differs fundamentally from general software programming: choices made at the microarchitectural level (datapath width, pipelining, memory organization, buffering, arithmetic sharing) constrain all downstream RTL implementations.

The agent reasons at two distinct levels:
1. **Level A: Architecture Selection**:
   - The agent proposes and compares structured `HardwareArchitectureCandidate` designs before generating RTL.
   - Explores guided mutations: wider/narrower datapath, deeper/shallower pipelines, systolic vs. SIMD compute, shared vs. duplicated multipliers, and local SRAM buffering.
   - Tracks lineages, mutation operators, and Pareto frontiers using `ArchitectureGenealogy`.
2. **Level B: RTL Implementation**:
   - Synthesizable SystemVerilog generation adhering to the chosen architecture candidate.
   - EDA compilation, simulation, synthesis, and bounded formal verification.

---

## 4. Targeted Web Research via ScrapeGraphAI

The web research pipeline replaces naive search with a domain-bounded, injection-defended extraction pipeline:
- **Primary ScrapeGraphAI Pipeline** (`scraping/scrapegraph_adapter.py`): Compact technical extraction using `SmartScraperGraph` directly over target URLs without redundant preliminary fetches. Model is aligned with **Qwen3-4B** (`qwen3:4b`).
- **Robust Native Fallback**: If ScrapeGraphAI is uninstalled or offline, gracefully falls back to native urllib + BeautifulSoup extraction.
- **Strict Domain Allowlist**: Restricted to authoritative hardware domains (`arxiv.org`, `github.com`, `verilator.org`, `yosyshq.readthedocs.io`, `docs.cocotb.org`, `riscv.org`, `ieee.org`).
- **Sanitization & Anti-Prompt-Injection** (`scraping/content_filter.py`): Web text is treated strictly as untrusted reference context, never executable instructions. Navigation trees, tracking scripts, and cookie notices are stripped.
- **Deduplication & Local Cache** (`scraping/deduplicator.py`): Persistent disk caching in `./data/web_cache/` prevents redundant requests.

---

## 5. 4-Tier Modular Memory System

ChromaDB and structured JSON stores are partitioned into isolated collections:
1. **Knowledge Memory** (`memory/knowledge.py`): SystemVerilog syntax rules, RISC-V specifications, and hardware architecture references.
2. **Experience Memory** (`memory/experience.py`, `memory/experience_store.py`): Semantic catalog of past compiler/simulation errors, root causes, fixes, and outcomes with a complete 9-category hardware failure taxonomy.
3. **Design Memory** (`memory/design_memory.py`): Git-versioned designs with associated cell counts, wire counts, and verification status.
4. **Trajectory Store** (`memory/trajectory.py`): Complete JSONL recording of every step (state, action, params, observation, reward, tool_output, next_state, done).

---

## 6. Grounded Reward Engine & Mathematical Formulation

The evaluator strictly isolates testbenches and evaluator code. It supports explicit operational modes (`FAST_DEVELOPMENT`, `STRICT_EVALUATION`, `TRAINING`) and uses hard gates with dynamic weight re-normalization:

$$R_{\text{modular}} = \sum_{k \in \text{Available}} w_k^{\text{effective}} \cdot R_k, \quad \text{where } \sum_{k \in \text{Available}} w_k^{\text{effective}} = 1.0$$

- **Hard Gate 1 (Compilation)**: If Verilator lint or compilation fails, $R_{\text{quality}} = 0.0$.
- **Hard Gate 2 (Functionality)**: If mandatory functional tests fail, design reward cannot exceed zero.
- **Strict Mode Gating**: In `STRICT_EVALUATION` and `TRAINING` modes, heuristic fallback area estimates are strictly excluded from reward calculation; only real gate synthesis and actual formal proofs contribute to quality score.
- **Strict Separation of RL Rewards vs Design Quality**:
  - **Step Reward ($r_t \in [-0.5, +1.0]$)**: Transition reward for individual actions (exploration bonus, compile progress, or error penalty).
  - **Episode Return ($\sum r_t$)**: Cumulative return across episode steps.
  - **Objective Design Quality ($Q \in [0.0, 1.0]$)**: Grounded hardware quality calculated strictly by `RewardEngine.compute_modular_reward()` from actual EDA outputs (compile, pass rate, area, formal status).
- **Objective Completion Gate**:
  - The `COMPLETE` action verifies compile pass, functional test pass, formal verification pass (if properties specified), and $Q \ge 0.50$. Premature completion incurs a $-0.5$ penalty.
- **No Secret Fallback Generators**: The environment requires policies to supply actual SystemVerilog code. Calling `GENERATE_RTL` without code returns an error penalty (`step_reward = -0.5`), preventing experiment contamination.

---

## 7. 10-Level Hardware Target Hierarchy

The long-term research objective is to develop optimized hardware architectures for a **portable independent AI computer** (CPU + RAM + AI accelerator + unified storage + memory controller + USB-C + power management):

| Level | Family | Description | Status |
|---|---|---|---|
| **Level 1** | Combinational & Basic Gates | NOT, AND/OR, 4:1 Mux, 3:8 Decoder, 4-bit Counter | **Implemented & Verified** |
| **Level 2** | Datapath & Storage | 8-bit Register, Ripple-Carry Adder, ALU, Sync FIFO, Shift Reg | **Implemented & Verified** |
| **Level 3** | Arithmetic Accelerators | 8-bit Signed MAC ($result = a \times b + acc$), 16-bit Pipelined MAC | **Implemented & Verified** |
| **Level 4** | Vector & Array Units | 4-Lane 8-bit Vector ALU, 2x2 Systolic Matrix Multiplier | **Implemented & Verified** |
| **Level 5** | Memory & DMA | Single-Port SRAM Controller, DMA Descriptor Engine | **Implemented & Verified** |
| **Level 6** | Microarchitecture & RISC-V | RV32I Instruction Decoder, 32-bit ALU with Branch Comparator | **Implemented & Verified** |
| **Level 7** | NPU Primitives | Weight Buffer Unit, Activation Engine (ReLU), Quantized Dot Product | **Implemented & Verified** |
| **Level 8** | Transformer Compute Blocks | Multi-Head Self-Attention Datapath & Softmax Scaling Block | *Conceptual Roadmap* |
| **Level 9** | AI Accelerator Subsystem | Systolic Array + Scratchpad + Weight Streaming Controller | *Conceptual Roadmap* |
| **Level 10** | Portable AI Computer Subsystem | Host CPU + NPU Interconnect + Unified Memory + Power Management | *Conceptual Roadmap* |

---

## 8. Generational Self-Evolution Controller

The self-evolution engine (`learning/self_evolution.py`) executes automated learning generations:

```
[USER OBJECTIVE]
       ↓
[TASK ANALYSIS]
       ↓
[ARCHITECTURE GENERATION] (N candidates across Pareto frontier)
       ↓
[CANDIDATE SELECTION & RTL GENERATION]
       ↓
[REAL EDA VERIFICATION] (Verilator + Cocotb + SymbiYosys + Yosys)
       ↓
[GROUNDED REWARD COMPUTATION] (Zero heuristic credit in STRICT/TRAINING)
       ↓
[FAILURE TAXONOMY & EXPERIENCE MEMORY]
       ↓
[TRAJECTORY GENERATION & SFT/GRPO DATASET BUILDER]
       ↓
[DEVICE-AWARE ADAPTER TRAINING] (CPU / CUDA / ROCm LoRA)
       ↓
[HELD-OUT BENCHMARK EVALUATION] (Identical unseen test suite)
       ↓
[PROMOTION DECISION]
   ├── Improved (> Threshold) ──> PROMOTE NEW ADAPTER & ADVANCE CURRICULUM
   └── Stagnant / Regressed ────> ROLLBACK TO PREVIOUS GENERATION
       ↓
[NEXT GENERATION]
```

### Empirical Pilot Experiment (`scripts/run_evolution_pilot.py`)
To prove that model adapters actually evolve and are evaluated independently without data leakage:
- **Task**: `L3_MAC_8BIT_SIGNED` (4 candidates per generation, 2 generations).
- **Generation 1**:
  - Base policy evaluated on held-out test suite.
  - Held-out Pass Rate: **100.0%**, Mean Score: **0.875**.
  - Met promotion threshold ($\ge 80\%$) $\rightarrow$ **PROMOTED** to `adapter_gen_001`. Curriculum advanced to Level 4.
- **Generation 2**:
  - Evolved adapter evaluated on higher-difficulty Level 4 held-out tasks.
  - Held-out Pass Rate: **0.0%** (unsolved Level 4 vector tasks).
  - Failed promotion threshold $\rightarrow$ **ROLLED BACK** safely to `adapter_gen_001`.
  - Machine-readable results and architecture genealogy saved to `sim_build/pilot_mac/pilot_report.json` and `genealogy.json`.

---

## 9. Implementation Status & Technical Transparency Matrix

| Dimension | Classification | Concrete Evidence / Location |
|---|---|---|
| **A. Architecture Search & Genealogy** | **Fully Implemented** | `agent/architecture_search.py`, `agent/schemas.py` |
| **B. 10 Anti-Reward-Hacking Defenses** | **Fully Implemented** | `tests/test_reward_hacking_defenses.py` (10/10 passing) |
| **C. Generational Evolution Controller** | **Fully Implemented** | `learning/self_evolution.py`, `scripts/run_evolution_pilot.py` |
| **D. 4-Tier Memory & Failure Taxonomy** | **Fully Implemented** | `memory/experience_store.py`, `memory/experience.py` |
| **E. Grounded Reward Engine & Modes** | **Fully Implemented** | `evaluator/reward.py` (`FAST_DEV`, `STRICT_EVAL`, `TRAINING`) |
| **F. Gymnasium Environment & Wrappers** | **Fully Implemented** | `learning/environment.py` (`observation_space.contains` strictly enforced) |
| **G. GRPO Evaluator & Advantage Engine** | **Fully Implemented** | `learning/grpo.py`, HuggingFace TRL `GRPOTrainer` compatible |
| **H. L1 - L7 Benchmark Tasks** | **Fully Implemented** | `benchmarks/curriculum.py` (Functional tests, references, formal properties) |
| **I. L8 - L10 Advanced Subsystems** | *Conceptual Roadmap* | Formally defined in `benchmarks/curriculum.py` with `implemented=False` |
| **J. Device-Aware SFT / LoRA Training** | **Framework Ready** | `learning/sft.py` (Auto-detects CPU / CUDA / ROCm, fp16/bf16 fallback) |
| **K. Real EDA Tool Execution** | **Tool Wrapped** | `tools/verilator.py`, `tools/cocotb.py`, `tools/yosys.py`, `tools/formal.py` |
| **L. Dev Fallback vs. Real Inference** | **Strict Boundary** | Silent fallbacks removed; clear `RuntimeError` if Ollama is unreachable |

---

## 10. Installation & Usage

### Prerequisites
- Python 3.11+
- Git
- Local LLM: Ollama with `qwen3:4b` (`ollama pull qwen3:4b`) or Transformers / vLLM.
- EDA tools for native execution: Verilator, Cocotb, Yosys, SymbiYosys (`sby`). (All tests include automated mock fallbacks when EDA binaries are absent).

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Check System Status
```bash
python main.py status
```

### Step 3: Run Full Test Suite (164 Tests)
```bash
python -m pytest tests/ -v
```
*(Runs 164 test cases covering JSON parsing, physical envelope modeling, error recovery, security path restriction, formal verification, Scraping filters, reward math, vector wrappers, active Gymnasium environment, curriculum benchmark resolution, and concurrent GRPO batching).*

### Step 4: Run Reproducible Architecture Evolution Pilot
```bash
python scripts/run_evolution_pilot.py
```
*(Executes multi-generation evolution on `L3_MAC_8BIT_SIGNED`, evaluates held-out benchmark before and after, performs promotion and rollback, and saves `pilot_report.json` and `genealogy.json`).*

### Step 5: Run Physical Envelope Constraint Satisfaction Evolution
```bash
python scripts/run_physical_envelope_evolution.py
```
*(Runs open-ended multi-generation architecture search against the 100x30x12mm, 5W, 85°C, 8GB RAM, USB-C, Qwen3-4B-INT4 >= 12-15 tok/s physical envelope until all constraints pass).*

### Step 6: Run End-to-End Autonomous Agent Milestone
```bash
python scripts/run_mac_milestone.py
```

---

## 11. Security & Anti-Reward-Hacking Defenses

The system implements 10 active defenses against evaluator manipulation:
1. **Evaluator Protection**: System files, testbenches, and evaluators are read-only; path traversal and writes outside `./rtl/generated/` and `./designs/` are blocked.
2. **Mandatory Benchmark IDs**: Fuzzy or heuristic module name inference is rejected; exact benchmark task IDs must be provided.
3. **Public vs. Held-out Isolation**: Golden test vectors and held-out testbenches are isolated in benchmark storage and never exposed to the agent.
4. **External Formal Binding**: Formal properties are injected out-of-band by the evaluation harness; model-generated assertions are not treated as formal proof.
5. **No Self-Authored Proofs**: Self-contained assertions in model RTL do not substitute for formal checks.
6. **Strict Formal Verification**: Skipping formal verification marks formal status as `UNAVAILABLE` or `FAIL`, never `PASS`.
7. **Strict Area Gating**: Heuristic cell estimates are rejected in `STRICT_EVALUATION` and `TRAINING` modes.
8. **Hidden-Test Leak Prevention**: Hardware task generator specs never contain golden test results.
9. **No Silent Fallbacks**: If model inference fails, the system fails clearly and logs the failure; it never generates synthetic training trajectories with pre-baked RTL solutions.
10. **Sandbox Command Isolation**: Model actions are strictly parsed JSON; arbitrary shell executions and command substitutions are prohibited.