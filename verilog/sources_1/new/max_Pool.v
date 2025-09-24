`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/20 14:08:13
// Design Name: 
// Module Name: max_Pool
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 2x2 Max Pooling
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module max_Pool #(
    parameter DATA_WIDTH = 8
)(
    input rst_n,
    input clk,
    input [DATA_WIDTH-1:0] line_1,
    input [DATA_WIDTH-1:0] line_2,
    output [DATA_WIDTH-1:0] max_out
    );

    reg [DATA_WIDTH-1:0] max_val_reg, max_out_reg;

    reg [DATA_WIDTH-1:0] a, b, c, d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a <= 0;
            b <= 0;
            c <= 0;
            d <= 0;
        end else begin
            // 讀取輸入數據
            a <= line_1;
            b <= a;
            c <= line_2;
            d <= c;
        end
    end

    // combinational logic
    always @(*) begin
        max_val_reg = a > b ? a : b;
        max_val_reg = max_val_reg > c ? max_val_reg : c;
        max_val_reg = max_val_reg > d ? max_val_reg : d;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_out_reg <= 0;
        end else begin
            max_out_reg <= max_val_reg;
        end
    end

    assign max_out = max_out_reg;

endmodule
