// -----------------------------------------------------------------------------
// mlp_core : 2-layer INT8 MLP as two weight-stationary systolic arrays
//
//   Layer 1 : N1 chains, each M1 PEs long   (one chain per output neuron y_i)
//   Layer 2 : N2 chains, each N1 PEs long
//
//   Chain i:  bias_i -> PE(i,0) -> PE(i,1) -> ... -> PE(i,M-1) -> y_i
//             with PE(i,j) holding w[i][j] and seeing x[j] (skewed by j).
//
// The bias is injected as the chain's initial partial sum, so no extra adder
// is needed. x[j] is broadcast to PE j of every chain. Every stage is
// registered, so a new image enters every clock cycle (II = 1).
//
// Pipeline timing (relative to the cycle x_reg holds an image = cycle 0):
//   layer-1 chain outputs : cycle M1
//   requant register      : cycle M1+1
//   layer-2 chain outputs : cycle M1+1+N1
//   argmax / out register : cycle M1+N1+2
// Input->output latency is M1+N1+3 cycles, counting the input register.
// The valid bit and tag travel through a matching shift register.
// -----------------------------------------------------------------------------
`include "mlp_cfg.vh"

module mlp_core #(
    parameter TAGW = 16
) (
    input                           clk,
    input                           rst,
    input                           in_valid,
    input      [TAGW-1:0]           in_tag,
    input      [`M1*8-1:0]          in_x,        // x[j] at [j*8 +: 8], 0..127
    output reg                      out_valid,
    output reg [TAGW-1:0]           out_tag,
    output reg [3:0]                out_class,
    output reg [`N2*`ACC2-1:0]      out_logits   // logit[i] at [i*ACC2 +: ACC2]
);
    localparam M1 = `M1, N1 = `N1, N2 = `N2, ACC1 = `ACC1, ACC2 = `ACC2;
    localparam LAT = M1 + N1 + 1;              // valid-pipe stages after x_reg

`include "mlp_weights.vh"                      // W1, W2, B1, B2 localparams

    // ------------------------------------------------------------ input register
    reg [M1*8-1:0] x_reg;
    always @(posedge clk) x_reg <= in_x;

    // ------------------------------------------------------------ layer 1
    wire [M1*8-1:0] x_sk;
    skew_buf #(.N(M1)) u_skew1 (.clk(clk), .din(x_reg), .dout(x_sk));

    wire signed [ACC1-1:0] ps1   [0:N1-1][0:M1];   // psum chain wires
    wire        [N1*8-1:0] h_vec;                  // requantised hidden vector
    reg         [N1*8-1:0] h_reg;

    genvar i, j;
    generate
        for (i = 0; i < N1; i = i + 1) begin : g_l1_chain
            assign ps1[i][0] = B1[i*32 +: ACC1];         // bias = initial psum
            for (j = 0; j < M1; j = j + 1) begin : g_pe
                pe #(.W(W1[(i*M1 + j)*8 +: 8]), .ACC(ACC1)) u_pe (
                    .clk(clk), .x(x_sk[j*8 +: 8]),
                    .psum_in(ps1[i][j]), .psum_out(ps1[i][j+1]));
            end
            requant #(.ACC(ACC1), .M(`RQ_M), .SH(`RQ_SH)) u_rq (
                .acc(ps1[i][M1]), .y(h_vec[i*8 +: 8]));
        end
    endgenerate

    always @(posedge clk) h_reg <= h_vec;          // inter-layer pipeline reg

    // ------------------------------------------------------------ layer 2
    wire [N1*8-1:0] h_sk;
    skew_buf #(.N(N1)) u_skew2 (.clk(clk), .din(h_reg), .dout(h_sk));

    wire signed [ACC2-1:0] ps2 [0:N2-1][0:N1];

    generate
        for (i = 0; i < N2; i = i + 1) begin : g_l2_chain
            assign ps2[i][0] = B2[i*32 +: ACC2];
            for (j = 0; j < N1; j = j + 1) begin : g_pe
                pe #(.W(W2[(i*N1 + j)*8 +: 8]), .ACC(ACC2)) u_pe (
                    .clk(clk), .x(h_sk[j*8 +: 8]),
                    .psum_in(ps2[i][j]), .psum_out(ps2[i][j+1]));
            end
        end
    endgenerate

    // ------------------------------------------------------------ argmax
    // first maximum wins (matches numpy argmax). Combinational over N2=10;
    // for a higher Fmax, split it into a registered 2-level compare tree.
    reg signed [ACC2-1:0] best;
    reg        [3:0]      best_i;
    integer               k;
    always @* begin
        best = ps2[0][N1]; best_i = 4'd0;
        for (k = 1; k < N2; k = k + 1)
            if (ps2[k][N1] > best) begin best = ps2[k][N1]; best_i = k; end
    end

    // ------------------------------------------------------------ valid / tag pipe
    reg [LAT-1:0]  v_pipe;
    reg [TAGW-1:0] t_pipe [0:LAT-1];
    reg            v0;
    reg [TAGW-1:0] t0;
    integer        s;
    always @(posedge clk) begin
        v0 <= rst ? 1'b0 : in_valid;                // aligned with x_reg
        t0 <= in_tag;
        v_pipe[0] <= rst ? 1'b0 : v0;
        t_pipe[0] <= t0;
        for (s = 1; s < LAT; s = s + 1) begin
            v_pipe[s] <= rst ? 1'b0 : v_pipe[s-1];
            t_pipe[s] <= t_pipe[s-1];
        end
    end

    // ------------------------------------------------------------ output register
    integer q;
    always @(posedge clk) begin
        out_valid <= rst ? 1'b0 : v_pipe[LAT-1];
        out_tag   <= t_pipe[LAT-1];
        out_class <= best_i;
        for (q = 0; q < N2; q = q + 1)
            out_logits[q*ACC2 +: ACC2] <= ps2[q][N1];
    end
endmodule
