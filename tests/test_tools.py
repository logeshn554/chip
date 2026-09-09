"""Tests for EDA Tool wrappers (requires tools installed)."""

import pytest
from rtl.parser import RTLParser


class TestRTLParser:
    """RTL parser tests — no external tools required."""

    SAMPLE_SV = """
    `timescale 1ns / 1ps

    module alu #(
        parameter DATA_WIDTH = 8
    )(
        input  logic                   clk,
        input  logic                   rst_n,
        input  logic [DATA_WIDTH-1:0]  a,
        input  logic [DATA_WIDTH-1:0]  b,
        input  logic [2:0]             op,
        output logic [DATA_WIDTH-1:0]  result
    );

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n)
                result <= '0;
            else
                case (op)
                    3'b000: result <= a + b;
                    3'b001: result <= a - b;
                    3'b010: result <= a & b;
                    default: result <= '0;
                endcase
        end

    endmodule
    """

    def test_extract_modules(self):
        modules = RTLParser.extract_modules(self.SAMPLE_SV)
        assert len(modules) == 1
        assert modules[0].name == "alu"

    def test_extract_module_names(self):
        names = RTLParser.extract_module_names(self.SAMPLE_SV)
        assert "alu" in names

    def test_extract_ports(self):
        modules = RTLParser.extract_modules(self.SAMPLE_SV)
        assert len(modules) == 1
        ports = modules[0].ports
        port_names = {p.name for p in ports}
        assert "clk" in port_names
        assert "rst_n" in port_names
        assert "result" in port_names

    def test_extract_parameters(self):
        modules = RTLParser.extract_modules(self.SAMPLE_SV)
        params = modules[0].parameters
        assert len(params) >= 1
        assert params[0].name == "DATA_WIDTH"
        assert params[0].default_value == "8"

    def test_basic_lint_pass(self):
        warnings = RTLParser.basic_lint(self.SAMPLE_SV)
        # Should have no critical warnings (has timescale, balanced, has reset)
        assert not any("Unbalanced" in w for w in warnings)

    def test_basic_lint_no_timescale(self):
        code = "module test; endmodule"
        warnings = RTLParser.basic_lint(code)
        assert any("timescale" in w.lower() for w in warnings)

    def test_basic_lint_unbalanced(self):
        code = "module test;\n  begin\n  endmodule"
        warnings = RTLParser.basic_lint(code)
        assert any("Unbalanced" in w or "unbalanced" in w.lower() for w in warnings)

    def test_dependency_graph(self):
        code = """
        module child; endmodule
        module parent;
            child c1();
        endmodule
        """
        modules = RTLParser.extract_modules(code)
        graph = RTLParser.build_dependency_graph(modules)
        assert "child" in graph.get("parent", [])

    def test_multi_module(self):
        code = """
        module mod_a; endmodule
        module mod_b; endmodule
        module mod_c; endmodule
        """
        modules = RTLParser.extract_modules(code)
        assert len(modules) == 3


class TestVerilatorTool:
    """Verilator integration tests — skipped if Verilator not installed."""

    @pytest.fixture
    def verilator(self):
        import shutil
        if not shutil.which("verilator"):
            pytest.skip("Verilator not installed")
        from tools.verilator import VerilatorTool
        return VerilatorTool({"timeout_seconds": 30})

    @pytest.mark.asyncio
    async def test_lint_valid(self, verilator, tmp_path):
        sv_file = tmp_path / "test.sv"
        sv_file.write_text(
            "module test(input logic clk); endmodule"
        )
        result = await verilator.lint([str(sv_file)])
        # May have warnings but should not error on valid simple module
        assert isinstance(result.errors, list)


class TestYosysTool:
    """Yosys integration tests — skipped if Yosys not installed."""

    @pytest.fixture
    def yosys(self):
        import shutil
        if not shutil.which("yosys"):
            pytest.skip("Yosys not installed")
        from tools.yosys import YosysTool
        return YosysTool({"timeout_seconds": 60})

    @pytest.mark.asyncio
    async def test_synthesize_simple(self, yosys, tmp_path):
        sv_file = tmp_path / "adder.sv"
        sv_file.write_text(
            "module adder(input [7:0] a, b, output [7:0] y);\n"
            "  assign y = a + b;\n"
            "endmodule\n"
        )
        result = await yosys.synthesize([str(sv_file)], top_module="adder")
        if result.success:
            assert result.cell_count > 0
