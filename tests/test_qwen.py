"""Tests for the Qwen client (canonical: Qwen-14B / qwen2.5:14b)."""

import pytest
from agent.qwen import QwenClient


class TestQwenClient:
    """Tests for QwenClient — JSON/code extraction and parsing."""

    def test_extract_json_direct(self):
        text = '{"action": "GENERATE_RTL", "params": {}}'
        result = QwenClient.extract_json(text)
        assert result["action"] == "GENERATE_RTL"

    def test_extract_json_code_block(self):
        text = 'Here is the output:\n```json\n{"action": "SEARCH"}\n```'
        result = QwenClient.extract_json(text)
        assert result["action"] == "SEARCH"

    def test_extract_json_generic_block(self):
        text = 'Response:\n```\n{"action": "SIMULATE"}\n```'
        result = QwenClient.extract_json(text)
        assert result["action"] == "SIMULATE"

    def test_extract_json_embedded(self):
        text = 'I think we should do this: {"action": "DEBUG", "params": {"code": "x"}} and then continue.'
        result = QwenClient.extract_json(text)
        assert result["action"] == "DEBUG"

    def test_extract_json_failure(self):
        text = "This is not JSON at all"
        result = QwenClient.extract_json(text)
        assert "error" in result

    def test_extract_code_systemverilog(self):
        text = "Here is the module:\n```systemverilog\nmodule test;\nendmodule\n```"
        code = QwenClient.extract_code(text, "systemverilog")
        assert "module test" in code
        assert "endmodule" in code

    def test_extract_code_python(self):
        text = "```python\nimport cocotb\n@cocotb.test()\nasync def test():\n    pass\n```"
        code = QwenClient.extract_code(text, "python")
        assert "import cocotb" in code

    def test_extract_code_no_block(self):
        text = "module simple; endmodule"
        code = QwenClient.extract_code(text, "systemverilog")
        assert code == text

    def test_parse_thinking_output(self):
        raw = "<think>Let me analyze this...</think>The answer is 42."
        thinking, content = QwenClient._parse_thinking_output(raw)
        assert "analyze" in thinking
        assert "42" in content

    def test_parse_thinking_no_think_block(self):
        raw = "Just a plain response."
        thinking, content = QwenClient._parse_thinking_output(raw)
        assert thinking == ""
        assert content == raw
