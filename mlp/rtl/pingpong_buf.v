// -----------------------------------------------------------------------------
// pingpong_buf : double-buffered image memory feeding the systolic core
//
// Two banks, each holding up to DEPTH images. One image is a single WIDTH-bit
// word (64 pixels x 8 bit = 512 bits), so the engine reads a whole image per
// cycle. On FPGA, a 512-bit word = 8 x BRAM36 (72 bits wide) side by side.
//
//   host side   : fills the "write" bank one image per cycle, then commits it
//   engine side : streams the "read" bank, one image per cycle, back-to-back
//
// While the engine drains one bank, the host fills the other, so loading
// overlaps computation and the array never waits once the host keeps up.
// -----------------------------------------------------------------------------
module pingpong_buf #(
    parameter WIDTH = 512,
    parameter DEPTH = 128,
    parameter AW    = 7,                    // log2(DEPTH)
    parameter TAGW  = 16
) (
    input                   clk,
    input                   rst,
    // host write port
    output                  h_ready,        // write bank is free
    input                   h_we,
    input      [AW-1:0]     h_addr,
    input      [WIDTH-1:0]  h_data,
    input                   h_commit,       // hand the write bank to the engine
    input      [AW:0]       h_count,        // images written (1..DEPTH)
    // engine read stream
    output reg              e_valid,
    output reg [TAGW-1:0]   e_tag,          // running image number
    output reg [WIDTH-1:0]  e_data
);
    reg [WIDTH-1:0] bank0 [0:DEPTH-1];
    reg [WIDTH-1:0] bank1 [0:DEPTH-1];

    reg         wsel, rsel;                  // bank being written / read
    reg  [1:0]  full;                        // bank holds committed images
    reg  [AW:0] cnt0, cnt1;
    reg  [AW:0] rptr;
    reg  [TAGW-1:0] tag;

    assign h_ready = ~full[wsel];

    wire [AW:0] rcnt   = rsel ? cnt1 : cnt0;
    wire        issue  = full[rsel];
    wire        last   = issue && (rptr == rcnt - 1'b1);
    wire        commit = h_commit && h_ready && (h_count != 0);

    // ---- memories: host write port + engine read port (simple dual-port RAM)
    always @(posedge clk) begin
        if (h_we && h_ready && !wsel) bank0[h_addr] <= h_data;
        if (h_we && h_ready &&  wsel) bank1[h_addr] <= h_data;
        if (issue) e_data <= rsel ? bank1[rptr[AW-1:0]] : bank0[rptr[AW-1:0]];
    end

    // ---- bank hand-over control
    always @(posedge clk) begin
        if (rst) begin
            wsel <= 1'b0; rsel <= 1'b0; full <= 2'b00;
            rptr <= 0; tag <= 0; e_valid <= 1'b0; e_tag <= 0;
        end else begin
            // full flags: set on commit, clear when the engine finishes a bank
            full[0] <= (commit && !wsel) | (full[0] & ~(last && !rsel));
            full[1] <= (commit &&  wsel) | (full[1] & ~(last &&  rsel));
            if (commit) begin
                if (wsel) cnt1 <= h_count; else cnt0 <= h_count;
                wsel <= ~wsel;
            end

            e_valid <= issue;                // data arrives one cycle later
            e_tag   <= tag;
            if (issue) begin
                tag  <= tag + 1'b1;
                rptr <= last ? 0 : rptr + 1'b1;
                if (last) rsel <= ~rsel;
            end
        end
    end
endmodule
