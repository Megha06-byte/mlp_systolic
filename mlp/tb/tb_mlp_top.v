// -----------------------------------------------------------------------------
// tb_mlp_top : streams all test images through the ping-pong buffer and checks
// every output (all 10 logits + class) bit-exactly against model/gen.py.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
`include "mlp_cfg.vh"
`include "ntest.vh"

module tb_mlp_top;
    localparam DEPTH = 128, AW = 7, TAGW = 16;
    localparam NT = `NTEST, N2 = `N2, ACC2 = `ACC2;

    reg clk = 0, rst = 1;
    always #2.5 clk = ~clk;                         // 200 MHz

    reg               h_we = 0, h_commit = 0;
    reg  [AW-1:0]     h_addr = 0;
    reg  [`M1*8-1:0]  h_data = 0;
    reg  [AW:0]       h_count = 0;
    wire              h_ready;
    reg  [9:0]        r_addr = 0;
    wire [3:0]        r_class;
    wire              out_valid;
    wire [TAGW-1:0]   out_tag;
    wire [3:0]        out_class;
    wire [N2*ACC2-1:0] out_logits;

    mlp_top #(.DEPTH(DEPTH), .AW(AW), .TAGW(TAGW)) dut (
        .clk(clk), .rst(rst),
        .h_ready(h_ready), .h_we(h_we), .h_addr(h_addr), .h_data(h_data),
        .h_commit(h_commit), .h_count(h_count),
        .r_addr(r_addr), .r_class(r_class),
        .out_valid(out_valid), .out_tag(out_tag),
        .out_class(out_class), .out_logits(out_logits));

    reg [`M1*8-1:0] img   [0:NT-1];
    reg [31:0]      glog  [0:NT*N2-1];
    reg [7:0]       gpred [0:NT-1];
    reg [7:0]       label [0:NT-1];

    integer n_out = 0, errors = 0, correct = 0, cyc = 0;
    integer first_out = -1, last_out = 0, first_in = -1, bubbles = 0;
    integer b, i, base, n, q;
    reg signed [ACC2-1:0] lg;
    reg signed [31:0]     gl;

    always @(posedge clk) cyc <= cyc + 1;

    // ---------------------------------------------------------- host: load batches
    initial begin
        $readmemh("x_img.hex",       img);
        $readmemh("gold_logits.hex", glog);
        $readmemh("gold_pred.hex",   gpred);
        $readmemh("label.hex",       label);
        repeat (5) @(posedge clk);
        rst <= 0;
        @(posedge clk);
        for (base = 0; base < NT; base = base + DEPTH) begin
            n = (NT - base < DEPTH) ? NT - base : DEPTH;
            while (!h_ready) @(posedge clk);         // wait for a free bank
            for (i = 0; i < n; i = i + 1) begin      // one image per cycle
                h_we <= 1; h_addr <= i; h_data <= img[base + i];
                @(posedge clk);
            end
            h_we <= 0;
            h_commit <= 1; h_count <= n;
            if (first_in < 0) first_in = cyc;
            @(posedge clk);
            h_commit <= 0;
        end
    end

    // ---------------------------------------------------------- checker
    always @(posedge clk) if (out_valid) begin
        if (first_out < 0) first_out = cyc;
        else if (cyc != last_out + 1) bubbles = bubbles + (cyc - last_out - 1);
        last_out = cyc;
        if (out_tag !== n_out) begin
            $display("TAG ERROR: expected %0d got %0d", n_out, out_tag); errors = errors + 1;
        end
        for (q = 0; q < N2; q = q + 1) begin
            lg = out_logits[q*ACC2 +: ACC2];
            gl = glog[out_tag*N2 + q];
            if (lg !== gl) begin
                if (errors < 10) $display("MISMATCH img %0d logit %0d: rtl %0d gold %0d",
                                          out_tag, q, lg, gl);
                errors = errors + 1;
            end
        end
        if (out_class !== gpred[out_tag][3:0]) begin
            if (errors < 10) $display("MISMATCH img %0d class: rtl %0d gold %0d",
                                      out_tag, out_class, gpred[out_tag]);
            errors = errors + 1;
        end
        if (out_class == label[out_tag][3:0]) correct = correct + 1;
        n_out = n_out + 1;
    end

    // ---------------------------------------------------------- finish + readback
    initial begin
        wait (n_out == NT);
        repeat (3) @(posedge clk);
        for (i = 0; i < 4; i = i + 1) begin          // result RAM readback
            r_addr <= i * 97; @(posedge clk); @(posedge clk);
            if (r_class !== gpred[i*97][3:0]) begin
                $display("READBACK ERROR at %0d", i*97); errors = errors + 1;
            end
        end
        $display("----------------------------------------------------------");
        $display("images           : %0d", NT);
        $display("bit-exact errors : %0d", errors);
        $display("accuracy         : %0d/%0d", correct, NT);
        $display("first-image latency (commit -> out) : %0d cycles", first_out - first_in);
        $display("stream: %0d outputs in %0d cycles, %0d bubble cycles",
                 NT, last_out - first_out + 1, bubbles);
        if (errors == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end

    initial begin #2_000_000; $display("TIMEOUT n_out=%0d", n_out); $finish; end
endmodule
