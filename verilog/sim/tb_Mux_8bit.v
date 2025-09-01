`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/31 04:01:56
// Design Name: 
// Module Name: tb_Mux_8bit
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


module tb_Mux_8bit;

    // --- Testbench 內部信號 ---
    // 使用 reg 來驅動 DUT 的輸入
    reg [7:0] tb_in0;
    reg [7:0] tb_in1;
    reg [7:0] tb_in2;
    reg [7:0] tb_in3;
    reg [1:0] tb_sel;

    // 使用 wire 來接收 DUT 的輸出
    wire [7:0] tb_out;

    // --- 實例化待測模組 (DUT - Design Under Test) ---
    // 模組名稱 Mux_8bit 必須與您的設計檔中的 module 名稱完全相符
    Mux_8bit dut (
        .in0(tb_in0),
        .in1(tb_in1),
        .in2(tb_in2),
        .in3(tb_in3),
        .sel(tb_sel),
        .out(tb_out)
    );

    // --- 測試流程 ---
    initial begin
        // 1. 初始化所有輸入信號
        $display("--- 開始 Mux 測試 ---");
        tb_in0 = 8'd10;   // 輸入 10
        tb_in1 = 8'd25;   // 輸入 25
        tb_in2 = 8'd100;  // 輸入 100
        tb_in3 = 8'd255;  // 輸入 255
        tb_sel = 2'b00;
        
        // 2. 循環遍歷所有可能的 sel 值，並觀察輸出
        #10; // 延遲 10 個時間單位，讓初始值穩定

        tb_sel = 2'b00;
        #10;
        $display("時間=%0t, sel=%b, 期望輸出=%d, 實際輸出=%d", $time, tb_sel, tb_in0, tb_out);

        tb_sel = 2'b01;
        #10;
        $display("時間=%0t, sel=%b, 期望輸出=%d, 實際輸出=%d", $time, tb_sel, tb_in1, tb_out);

        tb_sel = 2'b10;
        #10;
        $display("時間=%0t, sel=%b, 期望輸出=%d, 實際輸出=%d", $time, tb_sel, tb_in2, tb_out);

        tb_sel = 2'b11;
        #10;
        $display("時間=%0t, sel=%b, 期望輸出=%d, 實際輸出=%d", $time, tb_sel, tb_in3, tb_out);
        
        // 3. 結束模擬
        #10;
        $display("--- Mux 測試結束 ---");
        $finish;
    end

endmodule
