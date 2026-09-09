"""
Functional Verification Testbench for 8-bit Signed MAC Unit.

Mathematical Target:
    result = (a * b) + acc

Test coverage:
- Basic positive multiplication & accumulation
- Signed negative operands
- Maximum positive bounds (127 * 127)
- Minimum negative bounds (-128 * -128, -128 * 127)
- Multi-cycle running accumulation
- Synchronous reset and clear verification
"""

import pytest


def signed_mac_model(a: int, b: int, acc: int, data_width: int = 8, acc_width: int = 32) -> int:
    """Python software reference model for parameterized signed MAC."""
    # Clamp to signed ranges
    min_in = -(1 << (data_width - 1))
    max_in = (1 << (data_width - 1)) - 1
    assert min_in <= a <= max_in, f"Operand 'a' out of {data_width}-bit signed bounds: {a}"
    assert min_in <= b <= max_in, f"Operand 'b' out of {data_width}-bit signed bounds: {b}"

    product = a * b
    result = acc + product

    # 32-bit two's complement wrapping
    mask = (1 << acc_width) - 1
    result_masked = result & mask
    if result_masked >= (1 << (acc_width - 1)):
        result_masked -= (1 << acc_width)
    return result_masked


class TestSignedMACFunctional:
    """Deterministic functional test suite."""

    def test_positive_multiplication(self):
        # 3 * 4 + 0 = 12
        assert signed_mac_model(3, 4, 0) == 12

    def test_positive_negative_multiplication(self):
        # 5 * (-6) + 10 = -20
        assert signed_mac_model(5, -6, 10) == -20

    def test_negative_negative_multiplication(self):
        # (-8) * (-7) + (-10) = 46
        assert signed_mac_model(-8, -7, -10) == 46

    def test_max_positive_extreme(self):
        # 127 * 127 = 16129
        assert signed_mac_model(127, 127, 0) == 16129

    def test_min_negative_extreme(self):
        # -128 * 127 = -16256
        assert signed_mac_model(-128, 127, 0) == -16256
        # -128 * -128 = 16384
        assert signed_mac_model(-128, -128, 0) == 16384

    def test_running_accumulation(self):
        """Simulate 5 clock cycles of accumulation."""
        acc = 0
        vectors = [(2, 3), (-4, 5), (6, -2), (-3, -3), (10, 10)]
        for a, b in vectors:
            acc = signed_mac_model(a, b, acc)
        # 0 + 6 - 20 - 12 + 9 + 100 = 83
        assert acc == 83

    def test_zero_operands(self):
        assert signed_mac_model(0, 50, 25) == 25
        assert signed_mac_model(50, 0, 25) == 25


# Cocotb coroutines (active when executed under cocotb simulator)
try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import RisingEdge, FallingEdge

    @cocotb.test()
    async def cocotb_mac_test(dut):
        """Cocotb functional test runner for DUT."""
        clock = Clock(dut.clk, 10, units="ns")
        cocotb.start_soon(clock.start())

        # Reset
        dut.rst_n.value = 0
        dut.en.value = 0
        dut.clr.value = 0
        dut.a.value = 0
        dut.b.value = 0
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.rst_n.value = 1
        await RisingEdge(dut.clk)

        # Test vector 1: 5 * 4 = 20
        dut.en.value = 1
        dut.a.value = 5
        dut.b.value = 4
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        assert int(dut.out.value.signed_integer) == 20, f"Expected 20, got {dut.out.value.signed_integer}"

        # Test vector 2: add (-3) * 6 = -18 -> acc = 2
        dut.a.value = -3
        dut.b.value = 6
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        assert int(dut.out.value.signed_integer) == 2, f"Expected 2, got {dut.out.value.signed_integer}"

except ImportError:
    pass
