`timescale 1ns / 1ps

module tb_top;

    // --- Parameters matching the DUT ---
    parameter DATA_WIDTH        = 8;
    parameter ADDR_WIDTH        = 11;
    parameter N                 = 5;
    parameter MAX_WIDTH         = 32;
    parameter PE_DATA_WIDTH     = 22;
    parameter PE_PORT_WIDTH     = 8;

    // --- Testbench Internal Signals ---
    // Inputs to DUT
    reg                      clk;
    reg                      rst_n;
    reg                      start;

    // Outputs from DUT
    wire                     busy;
    wire                     done;
    wire [DATA_WIDTH-1:0]    result;
    
    integer i;

    // --- Instantiate the Device Under Test (DUT) ---
    top #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N(N),
        .MAX_WIDTH(MAX_WIDTH),
        .PE_DATA_WIDTH(PE_DATA_WIDTH),
        .PE_PORT_WIDTH(PE_PORT_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .busy(busy),
        .done(done),
        .result(result)
    );

    // --- Clock Generator (100MHz, 20ns period) ---
    initial begin
        clk = 0;
        forever #10 clk = ~clk;
    end

    // --- Test Sequence ---
    initial begin
        // 1. Initialization and Reset
        $display("--- [%0t] Test Start: Initializing and Resetting System ---", $time);
        rst_n = 1'b0;
        start = 1'b0;

        // Wait for a few clock cycles with reset asserted
        #200;
        
        // De-assert reset away from the clock edge for a clean release
        rst_n = 1'b1;
        $display("--- [%0t] Reset Released ---", $time);

        // 2. Send a single-cycle start pulse
        $display("--- [%0t] Sending Start Pulse ---", $time);
        start = 1'b1;
        #20;
        start = 1'b0;

        // 3. Run until stop time (no early finish)
        $display("--- [%0t] Test running, will stop at 3000000 ns ---", $time);
        
        // Let simulation run, wait until global stop
    end

    // --- Signal Monitoring ---
    initial begin
        $monitor("Time=%0t | rst_n=%b, start=%b | busy=%b, done=%b | result=%d",
            $time, rst_n, start, busy, done, result);
    end

    // --- Global Stop after 3,000,000 ns ---
    initial begin
        #60000;
        $display("--- [%0t] Simulation reached stop time (3,000,000 ns). Finishing. ---", $time);
        $finish;
    end

endmodule
