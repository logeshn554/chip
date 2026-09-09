"""Tests for the Formal Verification Tool (SymbiYosys wrapper)."""

import os
import pytest
from tools.formal import FormalVerificationTool


class TestFormalVerificationTool:
    def test_init_and_availability(self):
        tool = FormalVerificationTool()
        # Should be a boolean
        assert isinstance(tool.is_available, bool)

    def test_embedded_property_extraction(self):
        tool = FormalVerificationTool()
        sv_code = """
        module test (input logic clk, input logic a, output logic b);
            assert property (@(posedge clk) a |-> b);
            assume property (@(posedge clk) a != 0);
            cover property (@(posedge clk) b == 1);
        endmodule
        """
        props = tool.check_embedded_formal_properties(sv_code)
        assert props["num_assertions"] == 1
        assert props["num_assumptions"] == 1
        assert props["num_covers"] == 1
        assert len(props["assertions"]) == 1

    def test_generate_sby_config(self):
        tool = FormalVerificationTool()
        cfg = tool.generate_sby_config(top_module="mac", rtl_file="rtl/reference/mac.sv", depth=25)
        assert "mode bmc" in cfg
        assert "depth 25" in cfg
        assert "prep -top mac" in cfg

    @pytest.mark.asyncio
    async def test_verify_skipped_when_unavailable(self, tmp_path):
        # Force a non-existent binary to test graceful degradation
        tool = FormalVerificationTool(sby_binary="non_existent_sby_binary_12345", work_dir=str(tmp_path))
        test_file = tmp_path / "test.sv"
        test_file.write_text("module test; endmodule")

        res = await tool.verify(str(test_file), top_module="test")
        assert res["status"] == "SKIPPED"
        assert res["available"] is False
        assert "SKIPPED" in res["output"]

    @pytest.mark.asyncio
    async def test_missing_file_reports_error(self):
        tool = FormalVerificationTool()
        res = await tool.verify("non_existent_file_xyz.sv")
        assert res["status"] == "ERROR"
        assert "not found" in res["output"].lower()

    @pytest.mark.asyncio
    async def test_external_properties_binding(self, tmp_path):
        tool = FormalVerificationTool(sby_binary="non_existent_sby_binary_12345", work_dir=str(tmp_path))
        test_file = tmp_path / "mac.sv"
        test_file.write_text("module mac (input logic clk, output logic out); endmodule")

        external_spec = "module formal_spec; assert property (@(posedge clk) out == 0); endmodule"
        res = await tool.verify(str(test_file), top_module="mac", external_properties=external_spec)
        assert res["status"] == "SKIPPED"
        # Properties checked must account for the external specification
        assert res["properties_checked"] >= 1
        assert os.path.exists(tmp_path / "job_mac" / "mac_formal_spec.sv")

    @pytest.mark.asyncio
    async def test_external_direct_assertion_instrumentation(self, tmp_path):
        tool = FormalVerificationTool(sby_binary="non_existent_sby_binary_12345", work_dir=str(tmp_path))
        test_file = tmp_path / "mac.sv"
        test_file.write_text("module mac (input logic clk, output logic [31:0] accum); endmodule\n")

        external_spec = "always @(posedge clk) assert property (accum == '0);"
        res = await tool.verify(str(test_file), top_module="mac", external_properties=external_spec)
        assert res["properties_checked"] >= 1
        instrumented_file = tmp_path / "job_mac" / "mac_with_formal.sv"
        assert os.path.exists(instrumented_file)
        content = instrumented_file.read_text()
        assert "accum == '0" in content
        assert "`ifdef FORMAL" in content
