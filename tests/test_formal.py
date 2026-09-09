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
