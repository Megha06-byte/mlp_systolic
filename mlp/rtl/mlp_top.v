// -----------------------------------------------------------------------------
// mlp_top : memory system + 2-layer systolic MLP core
//
//   host --> [ ping-pong image buffer ] --512b/cycle--> [ mlp_core ] --> [ result RAM ]
//                 (2 x DEPTH images)                   L1: 16 chains x 64
//                                                      L2: 10 chains x 16
//
// Throughput: 1 image per clock once a bank is committed.
// Latency   : 1 (buffer read) + M1 + N1 + 3 cycles.
// The result RAM stores the predicted class per image tag. The host reads it
// back on a separate port, so result readout never stalls the array.
// -----------------------------------------------------------------------------
`include "mlp_cfg.vh"

module mlp_top #(
    parameter DEPTH = 128,
    parameter AW    = 7,
    parameter TAGW  = 16,
    parameter RDEPTH = 1024,               // result RAM entries
    parameter RAW    = 10
) (
    input                       clk,
    input                       rst,
    // host: image load
    output                      h_ready,
    input                       h_we,
    input      [AW-1:0]         h_addr,
    input      [`M1*8-1:0]      h_data,
    input                       h_commit,
    input      [AW:0]           h_count,
    // host: result readback
    input      [RAW-1:0]        r_addr,
    output reg [3:0]            r_class,
    // result stream (also visible to the host / testbench)
    output                      out_valid,
    output     [TAGW-1:0]       out_tag,
    output     [3:0]            out_class,
    output     [`N2*`ACC2-1:0]  out_logits
);
    wire               e_valid;
    wire [TAGW-1:0]    e_tag;
    wire [`M1*8-1:0]   e_data;

    pingpong_buf #(.WIDTH(`M1*8), .DEPTH(DEPTH), .AW(AW), .TAGW(TAGW)) u_inbuf (
        .clk(clk), .rst(rst),
        .h_ready(h_ready), .h_we(h_we), .h_addr(h_addr), .h_data(h_data),
        .h_commit(h_commit), .h_count(h_count),
        .e_valid(e_valid), .e_tag(e_tag), .e_data(e_data));

    mlp_core #(.TAGW(TAGW)) u_core (
        .clk(clk), .rst(rst),
        .in_valid(e_valid), .in_tag(e_tag), .in_x(e_data),
        .out_valid(out_valid), .out_tag(out_tag),
        .out_class(out_class), .out_logits(out_logits));

    // result RAM: write port from the core, read port for the host
    reg [3:0] res_mem [0:RDEPTH-1];
    always @(posedge clk) begin
        if (out_valid) res_mem[out_tag[RAW-1:0]] <= out_class;
        r_class <= res_mem[r_addr];
    end
endmodule
