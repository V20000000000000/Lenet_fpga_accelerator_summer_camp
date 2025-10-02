`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/31 03:49:13
// Design Name: 
// Module Name: Mux_8bit
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


module Mux_8bit (
    // --- 輸入 ---
    input wire [7:0] in0,   // 第 0 個 8 位元輸入
    input wire [7:0] in1,   // 第 1 個 8 位元輸入
    input wire [7:0] in2,   // 第 2 個 8 位元輸入
    input wire [7:0] in3,   // 第 3 個 8 位元輸入
    input wire [1:0] sel,   // 2 位元的選擇信號
    input wire clk,

    // --- 輸出 ---
    output reg [7:0] out    // 8 位元的輸出
);

    // 使用 always @(*) 來實現組合邏輯
    // 當任何輸入 (in0, in1, in2, in3, sel) 發生變化時，這個區塊會重新計算
    always @(posedge clk) begin
        case (sel)
            2'b00:  out <= in0;  
            2'b01:  out <= in1;  
            2'b10:  out <= in2;  
            2'b11:  out <= in3;  
            default: out <= 8'hXX; 
        endcase
    end

endmodule