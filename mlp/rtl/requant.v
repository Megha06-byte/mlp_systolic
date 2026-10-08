// -----------------------------------------------------------------------------
// requant : ReLU + INT8 rescale between layer 1 and layer 2 (combinational)
//
//     y = min(127, (max(acc,0) * M + 2^(SH-1)) >> SH)
//
// Integer-only, with no floating point. M / 2^SH approximates
// s_x * s_w1 / s_h. The ReLU is a sign-bit test; a negative acc forces the
// multiplier operand to 0.
// -----------------------------------------------------------------------------
module requant #(
    parameter ACC = 24,
    parameter M   = 16384,
    parameter SH  = 15
) (
    input  signed [ACC-1:0] acc,
    output        [7:0]     y              // 0..127
);
    wire [ACC-1:0]    pos    = acc[ACC-1] ? {ACC{1'b0}} : acc;      // ReLU
    wire [ACC+15:0]   scaled = ({16'd0, pos} * M + (1 << (SH-1))) >> SH;
    assign y = (scaled > 127) ? 8'd127 : scaled[7:0];               // saturate
endmodule
