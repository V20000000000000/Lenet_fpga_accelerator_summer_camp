`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/30 01:35:09
// Design Name: 
// Module Name: tb_delay
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


module tb_delay;

    // --- 參數定義 ---
    parameter DATA_WIDTH      = 8;
    parameter WAVEFRONT_DELAY = 4;
    parameter CLK_PERIOD      = 10; // 10ns -> 100MHz

    // --- Testbench 內部信號 ---
    reg                      clk;
    reg                      rst_n;
    reg [DATA_WIDTH-1:0]     tb_line_0;
    reg [DATA_WIDTH-1:0]     tb_line_1;
    reg [DATA_WIDTH-1:0]     tb_line_2;
    reg [DATA_WIDTH-1:0]     tb_line_3;
    reg [DATA_WIDTH-1:0]     tb_line_4;
    
    wire [DATA_WIDTH-1:0]    out_line_0;
    wire [DATA_WIDTH-1:0]    out_line_1;
    wire [DATA_WIDTH-1:0]    out_line_2;
    wire [DATA_WIDTH-1:0]    out_line_3;
    wire [DATA_WIDTH-1:0]    out_line_4;

    // --- 實例化待測模組 (DUT) ---
    delay #(
        .DATA_WIDTH(DATA_WIDTH),
        .WAVEFRONT_DELAY(WAVEFRONT_DELAY)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .line_0_in(tb_line_0),
        .line_1_in(tb_line_1),
        .line_2_in(tb_line_2),
        .line_3_in(tb_line_3),
        .line_4_in(tb_line_4),
        .line_0_out(out_line_0),
        .line_1_out(out_line_1),
        .line_2_out(out_line_2),
        .line_3_out(out_line_3),
        .line_4_out(out_line_4)
    );

    // --- 時脈產生器 ---
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    integer i;
    // --- 測試流程 ---
    initial begin
        // 1. 初始化與重置
        $display("--- 測試開始: 初始化與重置 ---");
        tb_line_0 <= 8'h00;
        tb_line_1 <= 8'h00;
        tb_line_2 <= 8'h00;
        tb_line_3 <= 8'h00;
        tb_line_4 <= 8'h00;
        rst_n     <= 1'b0;
        #(CLK_PERIOD * 2);
        rst_n     <= 1'b1;
        
        #500;

        // 2. 啟動數據流
        $display("\n--- 階段 1: 啟動數據流 (en=1) ---");
        // 使用 negedge clk 驅動輸入，避免競爭狀況
        @(negedge clk); 
        
        // 產生連續的數據流，每條 line 有不同偏移量以便觀察
        for (i = 0; i < 30; i = i + 1) begin
            @(negedge clk);
            tb_line_0 <= i;
            tb_line_1 <= i + 100;
            tb_line_2 <= i + 200;
            tb_line_3 <= i + 300;
            tb_line_4 <= i + 400;
        end

        // 3. 暫停數據流，觀察暫存器是否保持數據
        $display("\n--- 階段 2: 暫停數據流 (en=0) ---");
        @(negedge clk);
        // 讓數據保持不變，觀察輸出是否也保持不變
        tb_line_0 <= 8'hAA;
        tb_line_1 <= 8'hBB;
        tb_line_2 <= 8'hCC;
        tb_line_3 <= 8'hDD;
        tb_line_4 <= 8'hEE;
        #(CLK_PERIOD * 5);

        // 4. 再次啟動數據流
        $display("\n--- 階段 3: 再次啟動數據流 (en=1) ---");
        @(negedge clk);
        // 讓數據繼續流動
        for (i = 30; i < 40; i = i + 1) begin
            @(negedge clk);
            tb_line_0 <= i;
            tb_line_1 <= i + 100;
            tb_line_2 <= i + 200;
            tb_line_3 <= i + 300;
            tb_line_4 <= i + 400;
        end
        
        $display("\n--- 測試結束 ---");
        $finish;
    end

    // --- 監控輸出 ---
    initial begin
        $monitor("Time=%0t | in: %03d %03d %03d %03d %03d | out: %03d %03d %03d %03d %03d",
                $time, 
                tb_line_0, tb_line_1, tb_line_2, tb_line_3, tb_line_4,
                out_line_0, out_line_1, out_line_2, out_line_3, out_line_4);
    end

endmodule
