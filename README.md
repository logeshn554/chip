# Self-Evolving Hardware Design Agent 🧠⚡

A research prototype for an autonomous hardware design and verification agent built around a small local open-weight LLM (**Qwen3-4B / Qwen2.5-Coder-3B** via **Ollama**, **Transformers**, or **vLLM**). The agent conducts targeted technical research, generates synthesizable SystemVerilog, verifies it automatically via Verilator, Cocotb, SymbiYosys, and Yosys, inspects structured failure feedback, learns from previous episodes, and progressively evolves its designs.

> **Key Architectural Principle**:
> Qwen is the reasoning and planning engine.
> The complete self-evolving environment = **Qwen3-4B + 4-Tier Memory + Targeted ScrapeGraphAI Research + EDA Verification Pipeline + Grounded Reward Engine + Trajectory Store + Gym RL Environment**.

---

## 1. Target Architecture & Data Flow

```
                         ┌──────────────────────────────┐
                         │          USER TASK           │
                         │ "Design an 8-bit signed MAC" │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │       HARDWARE AGENT         │
                         │     (agent/agent_loop.py)    │
                         └──────┬───────────────┬───────┘
                                │               │
                ┌───────────────┘               └───────────────┐
                ▼                                               ▼
   ┌──────────────────────────┐                   ┌──────────────────────────┐
   │         QWEN3-4B         │                   │      4-TIER MEMORY       │
   │   (llm/qwen, agent/qwen) │                   │         (memory/)        │
   │  - Ollama / Transformers │                   ├──────────────────────────┤
   │  - vLLM / Mock Backends  │                   │ 1. Knowledge (ChromaDB)  │
   │  - Safe JSON extraction  │                   │ 2. Experience (Fix pairs)│
   └────────────┬─────────────┘                   │ 3. Design (Git versions) │
                │                                 │ 4. Trajectory (JSONL)    │
                ▼                                 └─────────────┬────────────┘
   ┌──────────────────────────┐                                 │
   │     DECISION & PLAN      │                                 │
   └────────────┬─────────────┘                                 │
                ├───────────────────────────────────────────────┤
                ▼                                               ▼
   ┌──────────────────────────┐                   ┌──────────────────────────┐
   │   TARGETED WEB RESEARCH  │                   │     EXISTING MEMORY      │
   │  - ScrapeGraphAI adapter │                   │  - Relevant SV rules     │
   │  - Allowed domain filter │                   │  - Past error fixes      │
   │  - Injection defense     │                   │  - Prior design metrics  │
   └────────────┬─────────────┘                   └─────────────┬────────────┘
                └───────────────────────┬───────────────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │      15 TYPED ACTIONS        │
                         │    (agent/action_router.py)  │
                         └──────────────┬───────────────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
                ▼                       ▼                       ▼
      ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
      │   GENERATE RTL   │    │ RUN VERIFICATION │    │  SYNTHESIS & EDA │
      │  SystemVerilog   │    │ Verilator,Cocotb │    │  Yosys & Metrics │
      │  (IEEE 1800-2012)│    │ SymbiYosys formal│    │  (Cells, Wires)  │
      └─────────┬────────┘    └────────┬─────────┘    └────────┬─────────┘
                └───────────────────────┼───────────────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │    GROUNDED REWARD ENGINE    │
                         │      (evaluator/reward.py)   │
                         │  - Hard gates (compile/lint) │
                         │  - Normalized weights        │
                         │  - Truthful formal metrics   │
                         └──────────────┬───────────────┘
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
   ┌──────────────────────────┐                   ┌──────────────────────────┐
   │    TRAJECTORY STORE      │                   │      DESIGN MEMORY       │
   │  (JSONL steps & returns) │                   │ (Git-tagged architecture)│
   └────────────┬─────────────┘                   └──────────────────────────┘
                ▼
   ┌──────────────────────────┐
   │   DATASET & SFT / RL     │
   │  - Filter & dedup pairs  │
   │  - Gym RL environment    │
   │  - SFT / GRPO preparation│
   └──────────────────────────┘
```

---

## 2. Supported Action System (15 Predefined Typed Actions)

All agent operations are mediated strictly by `ActionRouter` (`agent/action_router.py`). **Arbitrary shell execution generated by the LLM is strictly prohibited.** File writes are sandbox-restricted to the active design workspace (`./rtl/generated/`, `./designs/`), completely protecting evaluator code, testbenches, and configuration files from tampering.

| Action Type | Description | Sandbox Restrictions |
|---|---|---|
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

## 3. Targeted Web Research via ScrapeGraphAI

The web research pipeline replaces naive search with a domain-bounded, injection-defended extraction pipeline:
- **Primary ScrapeGraphAI Pipeline** (`scraping/scrapegraph_adapter.py`): Compact technical extraction using `SmartScraperGraph` directly over target URLs without redundant preliminary fetches. Model is aligned with **Qwen3-4B** (`qwen3:4b`).
- **Robust Native Fallback**: If ScrapeGraphAI is uninstalled or offline, gracefully falls back to native urllib + BeautifulSoup extraction.
- **Strict Domain Allowlist**: Restricted to authoritative hardware domains (`arxiv.org`, `github.com`, `verilator.org`, `yosyshq.readthedocs.io`, `docs.cocotb.org`, `riscv.org`, `ieee.org`).
- **Sanitization & Anti-Prompt-Injection** (`scraping/content_filter.py`): Web text is treated strictly as untrusted reference context, never executable instructions. Navigation trees, tracking scripts, and cookie notices are stripped.
- **Deduplication & Local Cache** (`scraping/deduplicator.py`): Persistent disk caching in `./data/web_cache/` prevents redundant requests.

---

## 4. 4-Tier Modular Memory System

ChromaDB and structured JSON stores are partitioned into isolated collections:
1. **Knowledge Memory** (`memory/knowledge.py`): SystemVerilog syntax rules, RISC-V specifications, and hardware architecture references.
2. **Experience Memory** (`memory/experience.py`): Semantic catalog of past compiler/simulation errors and their successful fixes.
3. **Design Memory** (`memory/design_memory.py`): Git-versioned designs with associated cell counts, wire counts, and verification status.
4. **Trajectory Store** (`memory/trajectory.py`): Complete JSONL recording of every step (state, action, params, observation, reward).

---

## 5. Grounded Reward Engine & Mathematical Formulation

The evaluator strictly isolates testbenches and evaluator code. It uses hard gates and dynamic weight re-normalization:

$$R_{\text{modular}} = \sum_{k \in \text{Available}} w_k^{\text{effective}} \cdot R_k, \quad \text{where } \sum_{k \in \text{Available}} w_k^{\text{effective}} = 1.0$$

- **Hard Gate 1 (Compilation)**: If Verilator lint or compilation fails, $R_{\text{quality}} = 0.0$.
- **Hard Gate 2 (Functionality)**: If mandatory functional tests fail, design reward cannot exceed zero.
- **Dynamic Weight Re-Normalization**: If area, timing, or power measurements are unavailable, the engine does NOT award artificial full credit (1.0). Instead, weights are dynamically re-normalized strictly over measured objectives.
- **Base vs. Auxiliary Formal Verification**:
  - `compute_v1_reward`: 8-point base reward (`compile(1) + functional(5) + synthesis(1) + lint(1)`). Formal verification is explicitly reported as an auxiliary metric (`formal_score`) to avoid altering the base scale.
  - External formal verification properties are bound outside the generated RTL (`benchmarks/curriculum.py`, `tools/formal.py`) preventing reward hacking through trivial self-generated assertions.
- **Phase Separation**:
  - `episode_return += step_reward`
  - `best_design_reward = max(best_design_reward, current_design_reward)`

---

## 6. 7-Level Benchmark Curriculum (`benchmarks/curriculum.py`)

A standardized progressive hardware curriculum with held-out test suites, external formal specifications, and known-good golden references:

- **Level 1 (Basic Logic)**: NOT Gate, AND/OR Gates, 4:1 Multiplexer, 3-to-8 Decoder, 4-bit Binary Counter.
- **Level 2 (Registers & Arithmetic)**: 8-bit Register with Enable, 8-bit Ripple-Carry Adder, 8-bit Arithmetic Logic Unit (ALU), Synchronous FIFO, 8-bit Shift Register.
- **Level 3 (MAC Units)**: 8-bit Signed MAC Unit ($result = a \times b + acc$), 16-bit Pipelined MAC Unit.
- **Level 4 (Vector & Array Processing)**: 4-Lane 8-bit Vector ALU, 2x2 Systolic Matrix Multiplier.
- **Level 5 (Memory Subsystems)**: Single-Port SRAM Controller, Direct Memory Access (DMA) Descriptor Component.
- **Level 6 (RISC-V)**: RV32I Instruction Decoder, 32-bit ALU with Branch Comparator.
- **Level 7 (NPU Components)**: Weight Buffer Unit, ReLU / Activation Engine, Quantized Dot-Product Unit.

---

## 7. Learning Foundations: SFT, Hardware-in-the-Loop GRPO, and Gymnasium RL Environment

The repository provides concrete, reproducible learning infrastructure:
- **Official Gymnasium RL Environment** (`learning/environment.py`): Inherits `gymnasium.Env`, implements standard spaces (`spaces.Discrete`, `spaces.Dict`), `reset(seed, options) -> (obs, info)`, and `step(action) -> (obs, reward, terminated, truncated, info)` with true reward accumulation.
- **Genuine Hardware-Reward GRPO** (`learning/grpo.py`):
  - `HardwareRewardEvaluator`: Evaluates candidate completions directly with Verilator lint/compile, Cocotb functional simulation, Yosys synthesis, and SymbiYosys formal verification.
  - Eliminates heuristic completion length proxies in favor of grounded hardware rewards.
  - `compute_group_advantages`: Implements group-relative advantage normalization: $A_i = (R_i - \mu)/\sigma$.
- **RL Transition Dataset Builder** (`learning/dataset.py`):
  - Converts raw trajectory logs into standardized $(s_t, a_t, r_t, s_{t+1}, done)$ MDP transitions.
  - Prepares failure/fix pairs for error-recovery SFT.
  - Generates preference pairs (`prompt`, `chosen`, `rejected`) for DPO/GRPO.
- **Honest Status on Self-Evolution**:
  - **In-Episode Adaptation (Phase A)**: Implemented & verified (agent diagnoses compiler/simulation errors and repairs RTL).
  - **Memory-Based Improvement (Phase B)**: Implemented & verified (retrieves past fixes from ChromaDB).
  - **Model Weight Evolution (Phase C/D)**: Foundation, Gymnasium environment, Hardware-Reward GRPO evaluator, and transition datasets are fully implemented and verified. Actual model weight updating requires running offline SFT/GRPO on collected trajectories.

---

## 8. Installation & Usage

### Prerequisites
- Python 3.11+
- Git
- Optional EDA tools for native execution: Verilator, Cocotb, Yosys, SymbiYosys (`sby`). (All tests include automated mock fallbacks when EDA binaries are absent).

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Check System Status
```bash
python main.py status
```

### Step 3: Run Full Test Suite
```bash
python -m pytest tests/ -v
```
*(Runs 91 test cases covering JSON parsing, error recovery, security path restriction, formal verification, Scraping filters, reward math, and the RL environment).*

### Step 4: Run End-to-End Milestone (8-Bit Signed MAC)
```bash
python scripts/run_mac_milestone.py
```
*(Demonstrates complete autonomous loop: research decision, knowledge retrieval, SystemVerilog generation, Verilator compilation, Cocotb testing, Yosys synthesis, grounded reward computation, Git commit, and trajectory recording).*

---

## 9. Security & Sandboxing

- **No Arbitrary Shell Calls**: LLM outputs only typed JSON actions. Commands are dispatched strictly via dedicated wrappers (`tools/verilator.py`, `tools/cocotb.py`, `tools/yosys.py`, `tools/formal.py`).
- **Protected File Boundaries**: The agent cannot modify `evaluator/`, `tests/`, `configs/`, or benchmark suites. Directory traversal (`../`) is intercepted and rejected.
- **Untrusted Web Context**: External web documents retrieved via ScrapeGraphAI are sanitized and embedded only as reference context, neutralized against prompt injection.