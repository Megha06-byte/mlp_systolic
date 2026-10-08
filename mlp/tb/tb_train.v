// -----------------------------------------------------------------------------
// tb_train : host-side driver used by train/fpga_trainer.py (simulation backend)
//
// Plays the role a CPU/PCIe driver would play on a real board:
//   1. load samples, labels, visiting order and initial weights over the host port
//   2. pulse start, wait for done, report the engine's cycle count
//   3. read every trained weight back and write it to a hex file
//
// Plusargs: +NSAMP=<samples> +NSTEPS=<SGD steps>
// Files (in the working dir): samples.hex labels.hex order.hex
//   w1_in.hex b1_in.hex w2_in.hex b2_in.hex  ->  w1_out.hex b1_out.hex ... cycles.txt
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_train;
    parameter P  = 8;
    parameter Q  = 4;
    parameter R  = 5;
    parameter LR = 7;
    localparam NS = 2048, NORD = 32768;

    reg clk = 0, rst = 1;
    always #2.5 clk = ~clk;

    reg          ld_we = 0, start = 0;
    reg  [2:0]   ld_sel = 0, rd_sel = 0;
    reg  [15:0]  ld_addr = 0, rd_addr = 0, n_steps = 0;
    reg  [511:0] ld_data = 0;
    wire [31:0]  rd_data, cycles;
    wire         busy, done;

    mlp_train_engine #(.P(P), .Q(Q), .R(R), .LR(LR), .NS(NS), .NORD(NORD)) dut (
        .clk(clk), .rst(rst),
        .ld_we(ld_we), .ld_sel(ld_sel), .ld_addr(ld_addr), .ld_data(ld_data),
        .rd_sel(rd_sel), .rd_addr(rd_addr), .rd_data(rd_data),
        .start(start), .n_steps(n_steps), .busy(busy), .done(done), .cycles(cycles));

    reg [511:0] samples [0:NS-1];
    reg [31:0]  labels  [0:NS-1];
    reg [31:0]  order   [0:NORD-1];
    reg [31:0]  w1 [0:1023], b1 [0:15], w2 [0:159], b2 [0:9];
    integer nsamp, nsteps, a, fd;

    task load(input [2:0] sel, input integer addr, input [511:0] data);
        begin
            ld_we <= 1; ld_sel <= sel; ld_addr <= addr; ld_data <= data;
            @(posedge clk);
            ld_we <= 0;
        end
    endtask

    task dump(input [2:0] sel, input integer n, input [8*16-1:0] fname);
        begin
            fd = $fopen(fname, "w");
            for (a = 0; a < n; a = a + 1) begin
                rd_sel <= sel; rd_addr <= a;
                repeat (4) @(posedge clk);
                $fdisplay(fd, "%08x", rd_data);
            end
            $fclose(fd);
        end
    endtask

    initial begin
        if (!$value$plusargs("NSAMP=%d", nsamp))   nsamp = 0;
        if (!$value$plusargs("NSTEPS=%d", nsteps)) nsteps = 0;
        $readmemh("samples.hex", samples, 0, nsamp - 1);
        $readmemh("labels.hex",  labels,  0, nsamp - 1);
        $readmemh("order.hex",   order,   0, nsteps - 1);
        $readmemh("w1_in.hex", w1); $readmemh("b1_in.hex", b1);
        $readmemh("w2_in.hex", w2); $readmemh("b2_in.hex", b2);

        repeat (4) @(posedge clk); rst <= 0; @(posedge clk);

        for (a = 0; a < nsamp;  a = a + 1) load(3'd0, a, samples[a]);
        for (a = 0; a < nsamp;  a = a + 1) load(3'd1, a, labels[a]);
        for (a = 0; a < nsteps; a = a + 1) load(3'd2, a, order[a]);
        for (a = 0; a < 1024;   a = a + 1) load(3'd3, a, w1[a]);
        for (a = 0; a < 16;     a = a + 1) load(3'd4, a, b1[a]);
        for (a = 0; a < 160;    a = a + 1) load(3'd5, a, w2[a]);
        for (a = 0; a < 10;     a = a + 1) load(3'd6, a, b2[a]);

        n_steps <= nsteps; start <= 1; @(posedge clk); start <= 0;
        @(posedge done);
        @(posedge clk);

        dump(3'd3, 1024, "w1_out.hex");
        dump(3'd4, 16,   "b1_out.hex");
        dump(3'd5, 160,  "w2_out.hex");
        dump(3'd6, 10,   "b2_out.hex");
        fd = $fopen("cycles.txt", "w"); $fdisplay(fd, "%0d", cycles); $fclose(fd);
        $display("TRAIN DONE steps=%0d cycles=%0d cycles/step=%0d", nsteps, cycles,
                 (nsteps > 0) ? cycles / nsteps : 0);
        $finish;
    end
endmodule
