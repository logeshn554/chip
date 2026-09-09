// =============================================================================
// Golden Reference: 8-bit Signed Multiply-Accumulate (MAC) Unit
// Mathematical target: result = (a * b) + acc
// Features:
// - Synthesizable SystemVerilog (IEEE 1800-2012)
// - Fully signed arithmetic ($signed)
// - Parameterized bit-widths (DATA_WIDTH=8, ACC_WIDTH=32)
// - Synchronous active-low reset and clock enable
// - Deterministic behavior with no vendor-specific primitives
// =============================================================================

`timescale 1ns / 1ps

module mac #(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
) (
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          en,
    input  logic                          clr,
    input  logic signed [DATA_WIDTH-1:0]  a,
    input  logic signed [DATA_WIDTH-1:0]  b,
    output logic signed [ACC_WIDTH-1:0]   out,
    output logic                          valid
);

    // Product of two N-bit signed integers requires 2N bits
    localparam int PROD_WIDTH = 2 * DATA_WIDTH;
    logic signed [PROD_WIDTH-1:0] product;

    // Accumulator register
    logic signed [ACC_WIDTH-1:0]  acc_reg;
    logic                         valid_reg;

    // Sign-extended product to match accumulator width
    logic signed [ACC_WIDTH-1:0]  product_ext;

    // Combinational signed multiplication
    always_comb begin
        product = a * b;
        // Sign extension to accumulator bitwidth
        product_ext = { {(ACC_WIDTH - PROD_WIDTH){product[PROD_WIDTH-1]}}, product };
    end

    // Synchronous accumulator register with active-low asynchronous reset
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg   <= '0;
            valid_reg <= 1'b0;
        end else if (clr) begin
            acc_reg   <= '0;
            valid_reg <= 1'b0;
        end else if (en) begin
            acc_reg   <= acc_reg + product_ext;
            valid_reg <= 1'b1;
        end else begin
            valid_reg <= 1'b0;
        end
    end

    assign out   = acc_reg;
    assign valid = valid_reg;

endmodule
