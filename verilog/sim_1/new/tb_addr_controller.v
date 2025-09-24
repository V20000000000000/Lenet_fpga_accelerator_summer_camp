`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/31 15:33:20
// Design Name: 
// Module Name: tb_addr_controller
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


module tb_addr_controller;

    // --- 參數定義 (需與 DUT 完整匹配) ---
    parameter ADDR_WIDTH            = 11;
    parameter KERNEL_SIZE           = 5;
    // Map 1
    parameter FEATURE_MAP1_SIZE       = 32;
    parameter FEATURE_MAP1_START_ADDR = 174;
    // Map 2
    parameter FEATURE_MAP2_SIZE       = 28;
    parameter FEATURE_MAP2_START_ADDR = 158;
    // Map 3
    parameter FEATURE_MAP3_SIZE       = 14;
    parameter FEATURE_MAP3_START_ADDR = 102;
    // Map 4
    parameter FEATURE_MAP4_SIZE       = 10;
    parameter FEATURE_MAP4_START_ADDR = 86;
    // Map 5
    parameter FEATURE_MAP5_SIZE       = 5;
    parameter FEATURE_MAP5_START_ADDR = 61;


    // --- Testbench 內部信號 ---
    reg                           clk;
    reg                           rst_n;
    reg  [2:0]                    ctrl_mode;
    reg  [ADDR_WIDTH-1:0]         ctrl_read_addr;

    wire [ADDR_WIDTH-1:0]         ctrl_write_addr;
    wire                          valid_out;
    wire [1:0]                    state_out;
    wire [ADDR_WIDTH-1:0]         row_counter_out;
    wire [ADDR_WIDTH-1:0]         col_counter_out;
    
    // --- 實例化待測模組 (DUT) ---
    addr_controller #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .KERNEL_SIZE(KERNEL_SIZE),
        .FEATURE_MAP1_SIZE(FEATURE_MAP1_SIZE),
        .FEATURE_MAP1_START_ADDR(FEATURE_MAP1_START_ADDR),
        .FEATURE_MAP2_SIZE(FEATURE_MAP2_SIZE),
        .FEATURE_MAP2_START_ADDR(FEATURE_MAP2_START_ADDR),
        .FEATURE_MAP3_SIZE(FEATURE_MAP3_SIZE),
        .FEATURE_MAP3_START_ADDR(FEATURE_MAP3_START_ADDR),
        .FEATURE_MAP4_SIZE(FEATURE_MAP4_SIZE),
        .FEATURE_MAP4_START_ADDR(FEATURE_MAP4_START_ADDR),
        .FEATURE_MAP5_SIZE(FEATURE_MAP5_SIZE),
        .FEATURE_MAP5_START_ADDR(FEATURE_MAP5_START_ADDR)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .ctrl_mode(ctrl_mode),
        .en(1'b1), // 永遠啟用
        .ctrl_read_addr(ctrl_read_addr),
        .ctrl_write_addr(ctrl_write_addr),
        .valid_out(valid_out),
        .state_out(state_out),
        .row_counter_out(row_counter_out),
        .col_counter_out(col_counter_out)
    );

    // --- 時脈產生器 (100MHz) ---
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // --- 定義一個可重複使用的測試任務 ---
    task automatic test_map(
        input [2:0] mode_to_test,
        input [ADDR_WIDTH-1:0] scan_start_addr,
        input [ADDR_WIDTH-1:0] scan_end_addr
    );
        // 修正: 為了相容 Verilog-2001, 迴圈變數需在此宣告
        integer j;
        begin
            $display("\n--- [%0t] 開始測試 Map (mode %d) ---", $time, mode_to_test);
            ctrl_mode <= mode_to_test;
            
            // 修正: 使用預先宣告的變數 j
            for (j = scan_start_addr; j <= scan_end_addr; j = j + 1) begin
                @(posedge clk);
                ctrl_read_addr <= j;
            end
            
            #50; // 在每個測試階段後稍作等待
        end
    endtask

    // --- 主測試流程 ---
    initial begin
        // 1. 初始化與重置
        $display("--- [%0t] 測試開始：初始化與重置 ---", $time);
        rst_n          <= 1'b0;
        ctrl_mode      <= 3'b000;
        ctrl_read_addr <= 0;
        #20;
        rst_n          <= 1'b1;
        #500;

        // --- 依序測試所有 Map ---
        test_map(3'b000, 0, 2047); // 測試 Map 1 (啟動位址 158)
        test_map(3'b001, 0, 2047); // 測試 Map 2 (啟動位址 142)
        test_map(3'b010, 0, 2047); // 測試 Map 3 (啟動位址 94)
        test_map(3'b011, 0, 2047); // 測試 Map 4 (啟動位址 86)
        test_map(3'b100, 0, 2047); // 測試 Map 5 (啟動位址 61)
        test_map(3'b101, 0, 2047); // 測試 Map 6 (啟動位址 43)
        test_map(3'b110, 0, 2047); // 測試 Map 7 (啟動位址 28)

        $display("\n--- [%0t] 所有模式測試結束 ---", $time);
        $finish;
    end

    // --- 監控信號 ---
    initial begin
        #30; // 等待重置結束後再開始監控
        $monitor("Time=%0t ns | Mode: %b, ReadAddr: %3d | State: %b, Valid: %b | Row: %2d, Col: %2d | WriteAddr: %3d",
                $time, ctrl_mode, ctrl_read_addr, state_out, valid_out, row_counter_out, col_counter_out, ctrl_write_addr);
    end

endmodule
