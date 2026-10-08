// -----------------------------------------------------------------------------
// mlp_train_engine : on-chip SGD training (forward + backprop + update)
//
// Network 64 -> 16 (ReLU, sat 255) -> 10, MSE loss, per-sample SGD, lr = 2^-LR.
// The arithmetic is bit-exact with train/hwref.py.
//
// Per-sample SGD has a hard dependency: sample n+1 must see the weights
// updated by sample n. So the engine processes one sample at a time and
// parallelises inside the sample:
//   P lanes : layer-1 inputs per cycle (x 16 neurons) -> 16*P MACs/cycle
//   Q lanes : layer-2 inputs per cycle (x 10 neurons) -> 10*Q MACs/cycle
//   R lanes : backprop rows per cycle (x 16 hidden)   -> 16*R MACs/cycle
//
// Weight storage:
//   W1 : 16 x P small RAM banks (neuron j, lane p), holding inputs k = s*P + p.
//        W1 never needs transposed access.
//   W2 : 160 registers. Forward walks columns, backprop walks rows, so both
//        orientations are needed and a RAM would force a transpose buffer.
//
// FUSED UPDATE + FORWARD: the update for sample n and the forward pass for
// sample n+1 visit every weight in the same order, so one pass does both:
//   read w -> w' = w - lr*grad(n) -> write w' -> MAC w' for sample n+1.
// Every weight is read and written once per sample.
//
// PREFETCH: the next sample's order entry and pixels are fetched during
// ERR/BWD, so the fetch never sits on the critical path.
//
// Per-sample schedule (cycles):
//   BIAS 1 | L1 64/P+2 | ACT 1 | L2 16/Q+1 | ERR 1 | BWD 10/R | E1 1
//   (P,Q,R) = (1,1,1) -> 97 cycles/sample, (8,4,5) -> 21, (16,8,10) -> 14,
//   (64,16,10) -> 10   (measured in simulation)
// -----------------------------------------------------------------------------
module mlp_train_engine #(
    parameter P    = 8,                 // layer-1 lanes   (divides 64)
    parameter Q    = 4,                 // layer-2 lanes   (divides 16)
    parameter R    = 5,                 // backprop lanes  (divides 10)
    parameter LR   = 7,                 // learning rate 2^-LR (LR >= 1)
    parameter NS   = 2048,              // sample memory depth
    parameter NORD = 32768              // order memory depth
) (
    input              clk,
    input              rst,
    // host load port (only while idle)
    input              ld_we,
    input      [2:0]   ld_sel,          // 0 sample 1 label 2 order 3 W1 4 b1 5 W2 6 b2
    input      [15:0]  ld_addr,
    input      [511:0] ld_data,
    // host readback port (registered)
    input      [2:0]   rd_sel,          // 3 W1 4 b1 5 W2 6 b2
    input      [15:0]  rd_addr,
    output reg [31:0]  rd_data,
    // control
    input              start,
    input      [15:0]  n_steps,         // number of SGD steps (order entries)
    output             busy,
    output reg         done,
    output reg [31:0]  cycles
);
    localparam N0 = 64, N1 = 16, N2 = 10;
    localparam S1 = N0 / P;             // layer-1 issue cycles
    localparam S2 = N1 / Q;             // layer-2 issue cycles
    localparam SB = N2 / R;             // backprop cycles

    // ---------------------------------------------------------------- helpers
    function signed [15:0] sat16;
        input signed [47:0] v;
        sat16 = (v >  48'sd32767) ? 16'sd32767 :
                (v < -48'sd32768) ? -16'sd32768 : v[15:0];
    endfunction

    // round-half-up arithmetic right shift by a constant s (s<=0 -> left shift)
    function signed [47:0] rsr;
        input signed [47:0] v;
        input integer       s;
        rsr = (s > 0) ? ((v + (48'sd1 <<< (s-1))) >>> s) : (v <<< (-s));
    endfunction

    // ---------------------------------------------------------------- memories
    reg [511:0] smem [0:NS-1];          // one 64-pixel sample per word
    reg [3:0]   lmem [0:NS-1];
    reg [10:0]  omem [0:NORD-1];        // sample visiting order

    reg signed [31:0] b1  [0:N1-1];
    reg signed [31:0] w2m [0:N2*N1-1];  // W2 master, index i*16 + j
    reg signed [31:0] b2  [0:N2-1];

    // ---------------------------------------------------------------- state
    localparam S_IDLE = 4'd0, S_F1 = 4'd1, S_F2 = 4'd2, S_BIAS = 4'd3, S_L1 = 4'd4,
               S_ACT = 4'd5, S_L2 = 4'd6, S_ERR = 4'd7, S_BWD = 4'd8, S_E1 = 4'd9,
               S_DONE = 4'd10;
    reg [3:0]  state;
    reg [15:0] n, nsteps_q;
    reg [10:0] oidx;
    reg        have_prev, have_cur;
    reg [7:0]  step;
    reg [511:0] x_cur, x_prev, x_next;
    reg [3:0]  lab, lab_next;

    reg signed [31:0] acc1 [0:N1-1];
    reg signed [31:0] acc2 [0:N2-1];
    reg signed [47:0] s1   [0:N1-1];
    reg        [7:0]  h_cur  [0:N1-1];
    reg        [7:0]  h_prev [0:N1-1];
    reg               mask1  [0:N1-1];
    reg signed [15:0] e1_prev [0:N1-1];
    reg signed [15:0] e2_cur  [0:N2-1];
    reg signed [15:0] e2_prev [0:N2-1];

    assign busy = (state != S_IDLE);

    // ---------------------------------------------------------------- layer-1 banks
    // pipeline: issue (read word s) -> U (update + write back) -> M (MAC)
    reg              l1_v1, l1_v2;
    reg [7:0]        l1_s1;
    wire [N1*P*16-1:0] wf1_flat;
    wire [N1*P*32-1:0] rb_flat;
    wire [P*8-1:0]   x_m_next;
    reg  [P*8-1:0]   x_m;

    genvar gj, gp;
    generate
        for (gp = 0; gp < P; gp = gp + 1) begin : g_xm
            assign x_m_next[gp*8 +: 8] = x_cur[(l1_s1*P + gp)*8 +: 8];
        end
        for (gj = 0; gj < N1; gj = gj + 1) begin : g_n
            for (gp = 0; gp < P; gp = gp + 1) begin : g_l
                reg signed [31:0] mem [0:S1-1];
                reg signed [31:0] rd_q;           // BRAM read register
                reg signed [15:0] wf_q;           // forward weight for the MAC
                reg signed [31:0] rb_q;
                wire [5:0] k_u = l1_s1 * P + gp;  // input index at stage U
                wire signed [8:0]  xp  = {1'b0, x_prev[k_u*8 +: 8]};
                wire signed [47:0] grd = e1_prev[gj] * xp;          // Q.19
                wire signed [47:0] dlt = have_prev ? rsr(grd, LR - 1) : 48'sd0;
                wire signed [31:0] wnew = rd_q - dlt[31:0];

                always @(posedge clk) begin
                    if (ld_we && ld_sel == 3'd3 && ld_addr[9:6] == gj &&
                        (ld_addr[5:0] % P) == gp)
                        mem[ld_addr[5:0] / P] <= ld_data[31:0];
                    if (state == S_L1 && step < S1) rd_q <= mem[step];
                    if (l1_v1) begin
                        mem[l1_s1] <= wnew;
                        wf_q <= sat16(wnew >>> 8);
                    end
                    rb_q <= mem[rd_addr[5:0] / P];
                end
                assign wf1_flat[(gj*P + gp)*16 +: 16] = wf_q;
                assign rb_flat [(gj*P + gp)*32 +: 32] = rb_q;
            end
        end
    endgenerate

    // ---------------------------------------------------------------- layer-2 pipe
    reg               l2_v;
    reg signed [15:0] wf2_q [0:N2*Q-1];   // updated forward weights, stage M
    reg        [7:0]  h_q   [0:Q-1];

    // ---------------------------------------------------------------- main FSM
    integer i, j, p, q, r, jj, ii;
    reg signed [47:0] t2, d, sum;
    reg signed [31:0] wn;

    always @(posedge clk) begin
        if (ld_we) case (ld_sel)
            3'd0: smem[ld_addr[10:0]] <= ld_data;
            3'd1: lmem[ld_addr[10:0]] <= ld_data[3:0];
            3'd2: omem[ld_addr[14:0]] <= ld_data[10:0];
            3'd4: b1[ld_addr[3:0]]    <= ld_data[31:0];
            3'd5: w2m[ld_addr[7:0]]   <= ld_data[31:0];
            3'd6: b2[ld_addr[3:0]]    <= ld_data[31:0];
            default: ;
        endcase

        if (rst) begin
            state <= S_IDLE; done <= 1'b0;
            l1_v1 <= 1'b0; l1_v2 <= 1'b0; l2_v <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state != S_IDLE) cycles <= cycles + 1;

            // ---- layer-1 pipeline control + stage M
            l1_v1 <= (state == S_L1) && (step < S1);
            l1_s1 <= step;
            l1_v2 <= l1_v1;
            x_m   <= x_m_next;
            if (l1_v2 && have_cur)
                for (j = 0; j < N1; j = j + 1) begin
                    sum = acc1[j];
                    for (p = 0; p < P; p = p + 1)
                        sum = sum + $signed(wf1_flat[(j*P + p)*16 +: 16])
                                  * $signed({1'b0, x_m[p*8 +: 8]});
                    acc1[j] <= sum[31:0];
                end

            // ---- layer-2 stage M
            l2_v <= (state == S_L2) && (step < S2);
            if (l2_v && have_cur)
                for (i = 0; i < N2; i = i + 1) begin
                    sum = acc2[i];
                    for (q = 0; q < Q; q = q + 1)
                        sum = sum + wf2_q[i*Q + q] * $signed({1'b0, h_q[q]});
                    acc2[i] <= sum[31:0];
                end

            case (state)
            S_IDLE: if (start) begin
                        n <= 0; nsteps_q <= n_steps; cycles <= 0;
                        have_prev <= 1'b0;
                        state <= (n_steps == 0) ? S_DONE : S_F1;
                    end
            // ---- first-sample fetch only (later samples are prefetched)
            S_F1:   begin oidx <= omem[0]; state <= S_F2; end
            S_F2:   begin
                        x_cur <= smem[oidx]; lab <= lmem[oidx]; have_cur <= 1'b1;
                        state <= S_BIAS;
                    end
            // ---- bias update for sample n-1, accumulator init for sample n
            S_BIAS: begin
                        for (j = 0; j < N1; j = j + 1) begin
                            d  = have_prev ? rsr(e1_prev[j], LR - 7) : 48'sd0;
                            wn = b1[j] - d[31:0];
                            b1[j] <= wn; acc1[j] <= wn;
                        end
                        for (i = 0; i < N2; i = i + 1) begin
                            d  = have_prev ? rsr(e2_prev[i], LR - 7) : 48'sd0;
                            wn = b2[i] - d[31:0];
                            b2[i] <= wn; acc2[i] <= wn;
                        end
                        step <= 0; state <= S_L1;
                    end
            // ---- fused W1 update (n-1) + layer-1 forward (n)
            //      exits in the cycle the last MAC (stage M) is applied
            S_L1:   begin
                        if (step < S1) step <= step + 1;
                        else if (!l1_v1) begin
                            state <= have_cur ? S_ACT : S_L2;   // flush skips ACT
                            step  <= 0;
                        end
                    end
            S_ACT:  begin
                        for (j = 0; j < N1; j = j + 1) begin
                            t2 = acc1[j] >>> 12;
                            h_cur[j] <= (acc1[j] <= 0) ? 8'd0 :
                                        (t2 >= 255)    ? 8'd255 : t2[7:0];
                            mask1[j] <= (acc1[j] > 0) && (t2 < 255);
                        end
                        step <= 0; state <= S_L2;
                    end
            // ---- fused W2 update (n-1) + layer-2 forward (n): Q columns per cycle
            S_L2:   begin
                        if (step < S2) begin
                            for (q = 0; q < Q; q = q + 1) begin
                                jj = step*Q + q;
                                h_q[q] <= h_cur[jj];
                                for (i = 0; i < N2; i = i + 1) begin
                                    t2 = e2_prev[i] * $signed({1'b0, h_prev[jj]});  // Q.19
                                    d  = have_prev ? rsr(t2, LR - 1) : 48'sd0;
                                    wn = w2m[i*N1 + jj] - d[31:0];
                                    w2m[i*N1 + jj]  <= wn;
                                    wf2_q[i*Q + q]  <= sat16(wn >>> 8);
                                end
                            end
                            step <= step + 1;
                        end else begin                  // last stage-M MAC this cycle
                            state <= have_cur ? S_ERR : S_DONE;
                        end
                    end
            // ---- output error (MSE gradient), Q.19 -> Q.12; prefetch order n+1
            S_ERR:  begin
                        for (i = 0; i < N2; i = i + 1) begin
                            t2 = acc2[i] - ((lab == i) ? 32'sd524288 : 32'sd0);
                            e2_cur[i] <= sat16(t2 >>> 7);
                        end
                        for (j = 0; j < N1; j = j + 1) s1[j] <= 0;
                        oidx <= omem[n[14:0] + 15'd1];
                        step <= 0; state <= S_BWD;
                    end
            // ---- backprop: s1[j] += sum_r W2[i][j] * e2[i], R rows per cycle;
            //      prefetch pixels of sample n+1
            S_BWD:  begin
                        for (j = 0; j < N1; j = j + 1) begin
                            sum = s1[j];
                            for (r = 0; r < R; r = r + 1) begin
                                ii  = step*R + r;
                                sum = sum + sat16(w2m[ii*N1 + j] >>> 8) * e2_cur[ii];
                            end
                            s1[j] <= sum;
                        end
                        if (step == 0) begin x_next <= smem[oidx]; lab_next <= lmem[oidx]; end
                        if (step == SB - 1) state <= S_E1;
                        else                step  <= step + 1;
                    end
            // ---- hidden error, roll sample n -> prev, n+1 -> cur
            S_E1:   begin
                        for (j = 0; j < N1; j = j + 1) begin
                            e1_prev[j] <= mask1[j] ? sat16(s1[j] >>> 12) : 16'sd0;
                            h_prev[j]  <= h_cur[j];
                        end
                        for (i = 0; i < N2; i = i + 1) e2_prev[i] <= e2_cur[i];
                        x_prev    <= x_cur;
                        x_cur     <= x_next;
                        lab       <= lab_next;
                        have_prev <= 1'b1;
                        have_cur  <= (n + 16'd1 < nsteps_q);   // else: flush pass
                        n         <= n + 1;
                        state     <= S_BIAS;
                    end
            S_DONE: begin done <= 1'b1; state <= S_IDLE; end
            default: state <= S_IDLE;
            endcase
        end
    end

    // ---------------------------------------------------------------- readback mux
    reg [31:0] rb_w1;
    always @(posedge clk) begin
        rb_w1 <= rb_flat[(rd_addr[9:6]*P + rd_addr[5:0] % P)*32 +: 32];
        case (rd_sel)
            3'd3:    rd_data <= rb_w1;
            3'd4:    rd_data <= b1[rd_addr[3:0]];
            3'd5:    rd_data <= w2m[rd_addr[7:0]];
            3'd6:    rd_data <= b2[rd_addr[3:0]];
            default: rd_data <= 32'd0;
        endcase
    end
endmodule
