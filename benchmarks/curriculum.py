"""
Curriculum Benchmarks for Hardware Agent Learning and Evaluation.

Implements the 7-level curriculum specified in Section 19:
Level 1: Basic Gates & Sequential Blocks (NOT gate, AND/OR, MUX, Decoder, Counter)
Level 2: Datapath & Storage (4-bit ALU, Register File, FIFO, Shift Register)
Level 3: Arithmetic Accelerators (8-bit signed MAC, 16-bit signed MAC)
Level 4: Vector & Matrix Units (Vector ALU, 2x2 Matrix Multiplier)
Level 5: Memory Control & DMA (SRAM Controller, DMA Component)
Level 6: RISC-V Building Blocks (RV32I ALU, Instruction Decoder, Program Counter)
Level 7: NPU Accelerators (NPU Processing Element, Systolic Array Row)

Includes:
- Public task descriptions (visible to agent)
- Private held-out tests (invisible to agent, protecting against reward hacking)
- Known-good reference SystemVerilog RTL
- Strict verification criteria
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BenchmarkTask:
    """A benchmark task in the hardware curriculum."""
    id: str
    level: int
    name: str
    description: str
    top_module: str
    verification_criteria: list[str]
    reference_rtl: str
    public_tests: list[dict[str, Any]] = field(default_factory=list)
    held_out_tests: list[dict[str, Any]] = field(default_factory=list)
    formal_properties: str = ""  # Separate SystemVerilog formal property specification
    target_cells: Optional[float] = None
    target_timing_ns: Optional[float] = None
    implemented: bool = True

    @property
    def task_id(self) -> str:
        """Compatibility property for task id."""
        return self.id

    @property
    def public_test_cases(self) -> list[dict[str, Any]]:
        """Compatibility property for public test cases."""
        return self.public_tests

    def get_public_spec(self) -> str:
        """Text presented to the agent without exposing held-out test vectors."""
        criteria_list = "\n".join(f"- {c}" for c in self.verification_criteria)
        return (
            f"Hardware Design Task [{self.id} — Level {self.level}: {self.name}]\n"
            f"Module Name: {self.top_module}\n\n"
            f"Description:\n{self.description}\n\n"
            f"Verification Criteria:\n{criteria_list}"
        )


class BenchmarkCurriculum:
    """Curriculum containing Level 1 through Level 7 hardware benchmarks."""

    def __init__(self):
        self._tasks: dict[str, BenchmarkTask] = {}
        self._load_all_tasks()

    def get_task(self, task_id: str) -> Optional[BenchmarkTask]:
        """Retrieve a task by ID."""
        return self._tasks.get(task_id)

    def get_level_tasks(self, level: int) -> list[BenchmarkTask]:
        """Retrieve all tasks for a specific curriculum level."""
        return [t for t in self._tasks.values() if t.level == level]

    def list_all_tasks(self) -> list[dict[str, Any]]:
        """List summary of all curriculum tasks."""
        return [
            {"id": t.id, "level": t.level, "name": t.name, "module": t.top_module}
            for t in self._tasks.values()
        ]

    def _load_all_tasks(self):
        # ─────────────────────────────────────────────────────────────
        # LEVEL 1: Gates & Basic Logic
        # ─────────────────────────────────────────────────────────────
        self._register(BenchmarkTask(
            id="L1_NOT_GATE",
            level=1,
            name="Inverter / NOT Gate",
            description="Design a 1-bit combinational NOT gate module with input 'a' and output 'y'.",
            top_module="not_gate",
            verification_criteria=["Output must be bitwise NOT of input", "Zero latency combinational logic"],
            reference_rtl="""module not_gate (input logic a, output logic y);
    assign y = ~a;
endmodule
""",
            public_tests=[{"a": 0, "expected_y": 1}],
            held_out_tests=[{"a": 1, "expected_y": 0}],
            formal_properties="always_comb assert property (y == ~a);",
        ))

        self._register(BenchmarkTask(
            id="L1_AND_OR",
            level=1,
            name="AND-OR Logic Unit",
            description="Design a 2-bit combinational module with inputs a[1:0], b[1:0], and outputs y_and[1:0], y_or[1:0].",
            top_module="and_or_gate",
            verification_criteria=["y_and = a & b", "y_or = a | b"],
            reference_rtl="""module and_or_gate (
    input  logic [1:0] a,
    input  logic [1:0] b,
    output logic [1:0] y_and,
    output logic [1:0] y_or
);
    assign y_and = a & b;
    assign y_or  = a | b;
endmodule
""",
            public_tests=[{"a": 3, "b": 1, "expected_and": 1, "expected_or": 3}],
            held_out_tests=[{"a": 0, "b": 2, "expected_and": 0, "expected_or": 2}],
            formal_properties="always_comb begin assert property (y_and == (a & b)); assert property (y_or == (a | b)); end",
        ))

        self._register(BenchmarkTask(
            id="L1_MUX2TO1",
            level=1,
            name="2-to-1 Multiplexer",
            description="Design a parameterized 2-to-1 multiplexer with inputs d0, d1, select sel, and output y.",
            top_module="mux2to1",
            verification_criteria=["When sel=0 y=d0", "When sel=1 y=d1", "Parameterized width (default 8)"],
            reference_rtl="""module mux2to1 #(parameter WIDTH = 8) (
    input  logic [WIDTH-1:0] d0,
    input  logic [WIDTH-1:0] d1,
    input  logic             sel,
    output logic [WIDTH-1:0] y
);
    assign y = sel ? d1 : d0;
endmodule
""",
            public_tests=[{"sel": 0, "d0": 42, "d1": 99, "expected": 42}],
            held_out_tests=[{"sel": 1, "d0": 42, "d1": 99, "expected": 99}],
            formal_properties="always_comb begin assert property (sel ? (y == d1) : (y == d0)); end",
        ))

        self._register(BenchmarkTask(
            id="L1_DECODER2TO4",
            level=1,
            name="2-to-4 Active-High Decoder",
            description="Design a 2-to-4 decoder with enable: input [1:0] in, input en, output logic [3:0] out.",
            top_module="decoder2to4",
            verification_criteria=["One-hot output when en=1", "All zeros when en=0"],
            reference_rtl="""module decoder2to4 (
    input  logic [1:0] in,
    input  logic       en,
    output logic [3:0] out
);
    always_comb begin
        if (!en) out = 4'b0000;
        else out = (4'b0001 << in);
    end
endmodule
""",
            public_tests=[{"en": 1, "in": 0, "expected": 1}, {"en": 0, "in": 3, "expected": 0}],
            held_out_tests=[{"en": 1, "in": 2, "expected": 4}, {"en": 1, "in": 3, "expected": 8}],
            formal_properties="always_comb begin if (!en) assert property (out == 4'b0000); else assert property (out == (4'b0001 << in)); end",
        ))

        self._register(BenchmarkTask(
            id="L1_COUNTER",
            level=1,
            name="4-bit Synchronous Up-Counter",
            description="Design a 4-bit synchronous up-counter with clk, rst_n, en, and output count[3:0].",
            top_module="counter",
            verification_criteria=["Reset clears count to 0", "When en=1 count increments on posedge clk", "Wraps at 15 to 0"],
            reference_rtl="""module counter (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       en,
    output logic [3:0] count
);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) count <= 4'd0;
        else if (en) count <= count + 4'd1;
    end
endmodule
""",
            public_tests=[{"cycles": 5, "en": 1, "expected": 5}],
            held_out_tests=[{"cycles": 16, "en": 1, "expected": 0}],
            formal_properties="initial assume (!rst_n); always @(posedge clk) begin if (!rst_n) begin end else if ($past(!rst_n)) assert property (count == 4'd0); else if (en && $past(rst_n)) assert property (count == ($past(count) + 4'd1)); end",
        ))

        # ─────────────────────────────────────────────────────────────
        # LEVEL 2: Datapath & Storage
        # ─────────────────────────────────────────────────────────────
        self._register(BenchmarkTask(
            id="L2_ALU_4BIT",
            level=2,
            name="4-bit Arithmetic Logic Unit",
            description="Design a 4-bit ALU with operands a[3:0], b[3:0], opcode[2:0], output result[3:0], zero, and carry_out.",
            top_module="alu",
            verification_criteria=["000: ADD", "001: SUB", "010: AND", "011: OR", "100: XOR", "101: SLL", "Zero flag assert"],
            reference_rtl="""module alu (
    input  logic [3:0] a,
    input  logic [3:0] b,
    input  logic [2:0] op,
    output logic [3:0] result,
    output logic       zero,
    output logic       carry_out
);
    logic [4:0] sum;
    always_comb begin
        carry_out = 1'b0;
        case (op)
            3'b000: begin sum = a + b; result = sum[3:0]; carry_out = sum[4]; end
            3'b001: begin sum = a - b; result = sum[3:0]; carry_out = sum[4]; end
            3'b010: result = a & b;
            3'b011: result = a | b;
            3'b100: result = a ^ b;
            3'b101: result = a << b[1:0];
            default: result = 4'b0000;
        endcase
        zero = (result == 4'b0000);
    end
endmodule
""",
            public_tests=[{"op": 0, "a": 3, "b": 4, "expected_result": 7}],
            held_out_tests=[{"op": 1, "a": 5, "b": 5, "expected_zero": 1}],
            formal_properties="always_comb begin if (op == 3'b000) assert property (result == a + b); if (op == 3'b010) assert property (result == (a & b)); if (result == 4'b0000) assert property (zero == 1'b1); end",
        ))

        self._register(BenchmarkTask(
            id="L2_FIFO_SYNC",
            level=2,
            name="Synchronous FIFO Buffer",
            description="Design a parameterizable synchronous FIFO buffer with write/read pointers, empty, and full flags.",
            top_module="fifo_sync",
            verification_criteria=["Data integrity in FIFO order", "Correct full and empty flag generation"],
            reference_rtl="""module fifo_sync #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH = 8
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  wr_en,
    input  logic [DATA_WIDTH-1:0] wr_data,
    input  logic                  rd_en,
    output logic [DATA_WIDTH-1:0] rd_data,
    output logic                  full,
    output logic                  empty
);
    logic [DATA_WIDTH-1:0] mem [DEPTH-1:0];
    logic [$clog2(DEPTH):0] count;
    logic [$clog2(DEPTH)-1:0] wr_ptr, rd_ptr;

    assign full  = (count == DEPTH);
    assign empty = (count == 0);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= '0; wr_ptr <= '0; rd_ptr <= '0; rd_data <= '0;
        end else begin
            if (wr_en && !full) begin
                mem[wr_ptr] <= wr_data;
                wr_ptr <= (wr_ptr + 1) % DEPTH;
            end
            if (rd_en && !empty) begin
                rd_data <= mem[rd_ptr];
                rd_ptr <= (rd_ptr + 1) % DEPTH;
            end
            case ({wr_en && !full, rd_en && !empty})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: ;
            endcase
        end
    end
endmodule
""",
            public_tests=[{"writes": [10, 20], "reads": 1, "expected_first": 10}],
            held_out_tests=[{"fill": 8, "expected_full": 1}],
            formal_properties="always_comb begin if (count == DEPTH) assert property (full == 1'b1); if (count == 0) assert property (empty == 1'b1); end",
        ))

        # ─────────────────────────────────────────────────────────────
        # LEVEL 3: MAC & Arithmetic
        # ─────────────────────────────────────────────────────────────
        self._register(BenchmarkTask(
            id="L3_MAC_8BIT_SIGNED",
            level=3,
            name="8-bit Signed Multiply-Accumulate (MAC)",
            description="Design a synthesizable 8-bit signed MAC unit: accum <= accum + (a * b) with 32-bit accumulation.",
            top_module="mac",
            verification_criteria=[
                "Signed two's complement arithmetic",
                "Synchronous accumulation across clock cycles",
                "Active-low reset clears accumulator",
                "Synthesizable with no vendor-specific primitives",
            ],
            reference_rtl="""`timescale 1ns / 1ps
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
""",
            public_tests=[
                {"a": 3, "b": 4, "acc_in": 0, "expected": 12},
                {"a": 5, "b": -6, "acc_in": 10, "expected": -20},
            ],
            held_out_tests=[
                {"a": -128, "b": -128, "acc_in": 0, "expected": 16384},
                {"a": -128, "b": 127, "acc_in": 0, "expected": -16256},
            ],
            formal_properties="initial assume (!rst_n); always @(posedge clk) begin if (!rst_n) begin end else if ($past(!rst_n)) assert property (accum == '0); else if ($past(valid_in)) assert property (accum == $past(accum) + ($past(a) * $past(b))); end",
        ))

        # ─────────────────────────────────────────────────────────────
        # LEVEL 4: Vector & Matrix Units
        # ─────────────────────────────────────────────────────────────
        self._register(BenchmarkTask(
            id="L4_VECTOR_ALU",
            level=4,
            name="4-Lane SIMD Vector ALU",
            description="Design a 4-lane SIMD vector ALU processing 4x 8-bit operands in parallel (ADD, SUB, MUL, MAX).",
            top_module="vector_alu",
            verification_criteria=["4 independent parallel lanes", "SIMD operation selected by opcode"],
            reference_rtl="""module vector_alu (
    input  logic [1:0]       op, // 0: ADD, 1: SUB, 2: MUL_LOWER, 3: MAX
    input  logic signed [7:0] a [4],
    input  logic signed [7:0] b [4],
    output logic signed [7:0] y [4]
);
    always_comb begin
        for (int i = 0; i < 4; i++) begin
            case (op)
                2'b00: y[i] = a[i] + b[i];
                2'b01: y[i] = a[i] - b[i];
                2'b10: y[i] = (a[i] * b[i]);
                2'b11: y[i] = (a[i] > b[i]) ? a[i] : b[i];
            endcase
        end
    end
endmodule
""",
            public_tests=[{"op": 0, "a": [1, 2, 3, 4], "b": [10, 20, 30, 40], "expected": [11, 22, 33, 44]}],
            held_out_tests=[{"op": 3, "a": [5, -2, 10, 0], "b": [3, 4, 9, -1], "expected": [5, 4, 10, 0]}],
            formal_properties="always_comb begin if (op == 2'b00) assert property (y[0] == a[0] + b[0]); if (op == 2'b01) assert property (y[0] == a[0] - b[0]); end",
        ))

        # ─────────────────────────────────────────────────────────────
        # LEVEL 5: Memory Subsystems
        # ─────────────────────────────────────────────────────────────
        self._register(BenchmarkTask(
            id="L5_SRAM_CONTROLLER",
            level=5,
            name="Single-Port Synchronous SRAM Controller",
            description="Design an SRAM controller supporting single-cycle read and write bursts with byte enables.",
            top_module="sram_controller",
            verification_criteria=["Read latency: 1 cycle", "Byte-enable masking on write"],
            reference_rtl="""module sram_controller #(
    parameter ADDR_WIDTH = 10,
    parameter DATA_WIDTH = 32
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  ce,
    input  logic                  we,
    input  logic [3:0]            be,
    input  logic [ADDR_WIDTH-1:0] addr,
    input  logic [DATA_WIDTH-1:0] din,
    output logic [DATA_WIDTH-1:0] dout
);
    logic [DATA_WIDTH-1:0] mem [1024];

    always_ff @(posedge clk) begin
        if (ce) begin
            if (we) begin
                if (be[0]) mem[addr][7:0]   <= din[7:0];
                if (be[1]) mem[addr][15:8]  <= din[15:8];
                if (be[2]) mem[addr][23:16] <= din[23:16];
                if (be[3]) mem[addr][31:24] <= din[31:24];
            end
            dout <= mem[addr];
        end
    end
endmodule
""",
            public_tests=[{"write_addr": 4, "data": 305419896, "be": 15}],
            held_out_tests=[{"byte_mask": 1, "expected_modified_only": 1}],
            formal_properties="always @(posedge clk) if (ce && we && be == 4'b1111) assert property (dout == din);",
        ))

        # ─────────────────────────────────────────────────────────────
        # LEVEL 6: RISC-V Components
        # ─────────────────────────────────────────────────────────────
        self._register(BenchmarkTask(
            id="L6_RISCV_ALU",
            level=6,
            name="RISC-V RV32I Execution ALU",
            description="Design a 32-bit RV32I compliant ALU supporting ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND.",
            top_module="riscv_alu",
            verification_criteria=["Signed and unsigned comparisons (SLT/SLTU)", "Arithmetic shift right (SRA) sign propagation"],
            reference_rtl="""module riscv_alu (
    input  logic [31:0] a,
    input  logic [31:0] b,
    input  logic [3:0]  alu_ctrl,
    output logic [31:0] result,
    output logic        zero
);
    always_comb begin
        case (alu_ctrl)
            4'b0000: result = a + b;
            4'b1000: result = a - b;
            4'b0001: result = a << b[4:0];
            4'b0010: result = ($signed(a) < $signed(b)) ? 32'd1 : 32'd0;
            4'b0011: result = (a < b) ? 32'd1 : 32'd0;
            4'b0100: result = a ^ b;
            4'b0101: result = a >> b[4:0];
            4'b1101: result = $signed(a) >>> b[4:0];
            4'b0110: result = a | b;
            4'b0111: result = a & b;
            default: result = 32'd0;
        endcase
        zero = (result == 32'd0);
    end
endmodule
""",
            public_tests=[{"ctrl": 0, "a": 100, "b": 200, "expected": 300}],
            held_out_tests=[{"ctrl": 13, "a": 4294967264, "b": 2, "expected_sra": 4294967288}],
            formal_properties="always_comb begin if (alu_ctrl == 4'b0000) assert property (result == a + b); if (alu_ctrl == 4'b1000) assert property (result == a - b); end",
        ))

        # ─────────────────────────────────────────────────────────────
        # LEVEL 7: NPU Accelerator Components
        # ─────────────────────────────────────────────────────────────
        self._register(BenchmarkTask(
            id="L7_NPU_PE",
            level=7,
            name="Systolic NPU Processing Element (PE)",
            description="Design a weight-stationary NPU Processing Element with input forwarding, activation pass-through, and accumulation.",
            top_module="npu_pe",
            verification_criteria=[
                "Weight register load mode",
                "Horizontal activation forwarding",
                "Vertical partial-sum accumulation",
                "Single-cycle pipelined latency",
            ],
            reference_rtl="""module npu_pe #(
    parameter ACT_WIDTH = 8,
    parameter WT_WIDTH  = 8,
    parameter PSUM_WIDTH = 32
) (
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          load_weight,
    input  logic signed [WT_WIDTH-1:0]    weight_in,
    input  logic signed [ACT_WIDTH-1:0]   act_in,
    input  logic signed [PSUM_WIDTH-1:0]  psum_in,
    output logic signed [ACT_WIDTH-1:0]   act_out,
    output logic signed [PSUM_WIDTH-1:0]  psum_out
);
    logic signed [WT_WIDTH-1:0] weight_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= '0;
            act_out    <= '0;
            psum_out   <= '0;
        end else begin
            if (load_weight) begin
                weight_reg <= weight_in;
            end
            act_out  <= act_in;
            psum_out <= psum_in + (act_in * weight_reg);
        end
    end
endmodule
""",
            public_tests=[{"weight": 2, "act": 3, "psum_in": 10, "expected_psum": 16}],
            held_out_tests=[{"load": 1, "weight": -4, "act": 5, "psum_in": 50, "expected_psum": 30}],
            formal_properties="always @(posedge clk) if (!rst_n) assert property (act_out == '0 && psum_out == '0); else if (load_weight) assert property (weight_reg == weight_in);",
        ))

        # ─────────────────────────────────────────────────────────────
        # FUTURE ROADMAP (Conceptual, Not Implemented)
        # ─────────────────────────────────────────────────────────────
        self._register(BenchmarkTask(
            id="L8_TRANSFORMER_ATTN_BLOCK",
            level=8,
            name="Transformer Attention Compute Engine",
            description="[Conceptual Roadmap] Scaled dot-product attention compute engine accelerating QK^T matrix multiplication and Softmax scaling.",
            top_module="attn_engine",
            verification_criteria=["Matrix transpose & dot-product", "Softmax exponentiation datapath", "Fixed-point Q8.8 scaling"],
            reference_rtl="// Conceptual: pending full reference implementation\nmodule attn_engine; endmodule",
            implemented=False,
            target_cells=1200.0,
            target_timing_ns=4.0,
        ))

        self._register(BenchmarkTask(
            id="L9_AI_ACCELERATOR_SUBSYSTEM",
            level=9,
            name="Integrated AI Accelerator Subsystem",
            description="[Conceptual Roadmap] Heterogeneous accelerator subsystem coupling 2D systolic array, double-buffered scratchpad SRAM, and AXI4-Stream DMA controller.",
            top_module="ai_accelerator_subsystem",
            verification_criteria=["AXI4-Stream slave/master interface", "SRAM double-buffering bank arbitration", "Continuous systolic execution without pipeline stalls"],
            reference_rtl="// Conceptual: pending full reference implementation\nmodule ai_accelerator_subsystem; endmodule",
            implemented=False,
            target_cells=3500.0,
            target_timing_ns=3.5,
        ))

        self._register(BenchmarkTask(
            id="L10_PORTABLE_AI_COMPUTER_SUBSYSTEM",
            level=10,
            name="Portable Independent AI Computer Subsystem",
            description="[Conceptual Roadmap] Complete SoC subsystem: RV32I host CPU, AI matrix accelerator, unified memory controller, USB-C interface, and power management block.",
            top_module="ai_computer_soc",
            verification_criteria=["Host CPU booting and command dispatch", "Direct memory access between RAM and AI engine", "Autonomous battery power throttling"],
            reference_rtl="// Conceptual: pending full reference implementation\nmodule ai_computer_soc; endmodule",
            implemented=False,
            target_cells=10000.0,
            target_timing_ns=3.0,
        ))

    def _register(self, task: BenchmarkTask):
        if task.target_cells is None:
            defaults = {1: 10.0, 2: 50.0, 3: 150.0, 4: 400.0, 5: 300.0, 6: 350.0, 7: 500.0, 8: 1200.0, 9: 3500.0, 10: 10000.0}
            task.target_cells = defaults.get(task.level, 100.0)
        if task.target_timing_ns is None:
            task.target_timing_ns = 5.0
        self._tasks[task.id] = task

    def get_implemented_tasks(self, level: Optional[int] = None) -> list[BenchmarkTask]:
        """Return only fully implemented and verified benchmark tasks."""
        if level is not None:
            return [t for t in self._tasks.values() if t.level == level and t.implemented]
        return [t for t in self._tasks.values() if t.implemented]

    def adjust_difficulty(self, current_level: int, success_rate: float, recent_failures: int = 0, threshold: float = 0.80) -> int:
        """Empirically adjust curriculum difficulty based on measured agent performance.
        
        - If success_rate >= threshold and current_level < 7: advance to next level.
        - If recent_failures >= 3 or success_rate < 0.25 and current_level > 1: return to previous level.
        - Otherwise: remain at current level.
        """
        if success_rate >= threshold and current_level < 7:
            next_lvl = current_level + 1
            return next_lvl
        elif (recent_failures >= 3 or success_rate < 0.25) and current_level > 1:
            prev_lvl = current_level - 1
            return prev_lvl
        return current_level


_CURRICULUM_INSTANCE = BenchmarkCurriculum()
CURRICULUM_BENCHMARKS = _CURRICULUM_INSTANCE._tasks


def get_benchmark(task_id: str) -> Optional[BenchmarkTask]:
    """Get a curriculum benchmark by task ID."""
    return _CURRICULUM_INSTANCE.get_task(task_id)


def get_curriculum_level(level: int) -> list[BenchmarkTask]:
    """Get all benchmark tasks for a given level."""
    return _CURRICULUM_INSTANCE.get_level_tasks(level)


def get_all_benchmarks() -> dict[str, BenchmarkTask]:
    """Get all curriculum benchmark tasks as a mapping."""
    return dict(_CURRICULUM_INSTANCE._tasks)
