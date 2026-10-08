// -----------------------------------------------------------------------------
// skew_buf : triangular delay line in front of a systolic array
//
// Element j of the input vector is delayed by j clock cycles, so x[j] reaches
// PE j of every chain exactly when that chain's partial sum arrives there.
// A new vector can enter every cycle (fully pipelined). On FPGA the longer
// delay lines map to SRL16/SRL32 shift registers, not flip-flops.
// -----------------------------------------------------------------------------
module skew_buf #(
    parameter N = 8                       // vector length
) (
    input              clk,
    input  [N*8-1:0]   din,               // element j at [j*8 +: 8]
    output [N*8-1:0]   dout
);
    genvar j;
    generate
        for (j = 0; j < N; j = j + 1) begin : g_lane
            if (j == 0) begin : g_d0
                assign dout[7:0] = din[7:0];            // no delay
            end else begin : g_dn
                reg [7:0] sr [0:j-1];                   // j-stage shift register
                integer k;
                always @(posedge clk) begin
                    sr[0] <= din[j*8 +: 8];
                    for (k = 1; k < j; k = k + 1)
                        sr[k] <= sr[k-1];
                end
                assign dout[j*8 +: 8] = sr[j-1];
            end
        end
    endgenerate
endmodule
