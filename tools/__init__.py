"""EDA and research tools for the self-evolving hardware agent."""

from tools.web_research import WebResearchTool, WebResearchArgs
from tools.memory_search import MemorySearchTool, MemorySearchArgs
from tools.verilator import VerilatorTool
from tools.cocotb import CocotbTool
from tools.yosys import YosysTool
from tools.formal import FormalVerificationTool
from tools.git import GitTool

__all__ = [
    "WebResearchTool",
    "WebResearchArgs",
    "MemorySearchTool",
    "MemorySearchArgs",
    "VerilatorTool",
    "CocotbTool",
    "YosysTool",
    "FormalVerificationTool",
    "GitTool",
]

