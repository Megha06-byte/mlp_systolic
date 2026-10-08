// -----------------------------------------------------------------------------
// pe : one cell of a weight-stationary systolic chain
//
//     psum_out <= psum_in + W * x
//
// W is a hardcoded INT8 constant (a parameter). Sparse-weight optimisation:
// if W == 0 (pruned), the generate block builds NO multiplier and NO adder.
// The cell becomes a single pipeline register, which is still needed to keep
// every chain the same length so outputs stay aligned. A non-zero constant
// multiply is also cheap, because synthesis reduces it to a few shift-adds or
// one DSP.
// -----------------------------------------------------------------------------
module pe #(
    parameter [7:0] W   = 8'd0,        // int8 weight, two's complement
    parameter       ACC = 24           // partial-sum width
) (
    input                       clk,
    input      signed [7:0]     x,         // activation (already skewed)
    input      signed [ACC-1:0] psum_in,
    output reg signed [ACC-1:0] psum_out
);
    localparam signed [7:0] WS = W;

    generate
        if (W == 8'd0) begin : g_pruned
            always @(posedge clk)
                psum_out <= psum_in;                      // pass-through only
        end else begin : g_mac
            wire signed [15:0] prod = x * WS;             // 8x8 -> 16 bit
            always @(posedge clk)
                psum_out <= psum_in + {{(ACC-16){prod[15]}}, prod};
        end
    endgenerate
endmodule
