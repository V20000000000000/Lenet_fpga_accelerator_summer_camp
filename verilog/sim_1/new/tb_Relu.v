`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/20 15:13:41
// Design Name: 
// Module Name: tb_Relu
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module tb_Relu;

    // --- Parameters ---
    localparam DATA_WIDTH = 8;
    localparam CLK_PERIOD = 10; // 10ns, 100MHz

    // --- Testbench Signals ---
    reg                      rst_n;
    reg                      clk;
    // ** Important: Declare data_in as signed to test negative values **
    reg signed [DATA_WIDTH-1:0] data_in;

    wire [DATA_WIDTH-1:0]    data_out;

    // --- Test Helper Variables ---
    integer i;
    integer error_count = 0;
    
    // "Golden Model" to calculate the expected output
    reg [DATA_WIDTH-1:0] expected_out;
    // Register to store the previous cycle's input for verification
    reg signed [DATA_WIDTH-1:0] data_in_d1;


    // --- Instantiate the Device Under Test (DUT) ---
    Relu #(
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .rst_n(rst_n),
        .clk(clk),
        .data_in(data_in),
        .data_out(data_out)
    );

    // --- Clock Generator ---
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2) clk = ~clk;
    end

    // --- Main Test Procedure ---
    initial begin
        #200;
        $display("========================================");
        $display("T=%0t | ReLU Testbench Started", $time);
        
        // 1. Initialize and apply reset
        data_in <= 0;
        rst_n   <= 1'b0; // Apply reset
        repeat(5) @(posedge clk);
        rst_n   <= 1'b1; // Release reset
        $display("T=%0t | System Reset Released", $time);
        
        // --- Stream test vectors and verify the output ---
        $display("\n--- Streaming test data from -10 to 10 ---");
        for (i = -10; i <= 10; i = i + 1) begin
            @(posedge clk);
            data_in <= i;
            
            // Store the previous input value
            data_in_d1 <= data_in;

            // Calculate the expected output based on the *previous* input
            // This accounts for the 1-cycle latency of the DUT
            expected_out = (data_in_d1 > 0) ? data_in_d1 : 0;
            
            // Start verification after the first valid data has propagated
            if (i > -10) begin
                if (data_out === expected_out) begin
                    $display("T=%0t | PASS: In = %d, Out = %d, Expected = %d", 
                                $time, data_in_d1, data_out, expected_out);
                end else begin
                    $display("T=%0t | **** FAIL ****: In = %d, Out = %d, Expected = %d", 
                                $time, data_in_d1, data_out, expected_out);
                    error_count = error_count + 1;
                end
            end
        end
        
        // One final clock cycle to check the last input value (i=10)
        @(posedge clk);
        data_in_d1 <= data_in;
        expected_out = (data_in_d1 > 0) ? data_in_d1 : 0;
        if (data_out === expected_out) begin
            $display("T=%0t | PASS: In = %d, Out = %d, Expected = %d", 
                        $time, data_in_d1, data_out, expected_out);
        end else begin
            $display("T=%0t | **** FAIL ****: In = %d, Out = %d, Expected = %d", 
                        $time, data_in_d1, data_out, expected_out);
            error_count = error_count + 1;
        end


        // --- Test Summary ---
        $display("\n========================================");
        if (error_count == 0) begin
            $display("T=%0t | All ReLU tests PASSED!", $time);
        end else begin
            $display("T=%0t | Test FAILED with %d errors!", $time, error_count);
        end
        $display("========================================");
        $finish;
    end

endmodule