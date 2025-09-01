`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/20 15:08:00
// Design Name: 
// Module Name: Relu
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// Relu unit
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module Relu#(
    parameter DATA_WIDTH = 8
)(
    input rst_n,
    input clk,
    // *** 在這裡加上 signed 關鍵字 ***
    input signed [DATA_WIDTH-1:0] data_in, 
    output [DATA_WIDTH-1:0] data_out
    );

    reg [DATA_WIDTH-1:0] data_out_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out_reg <= 0;
        end else begin
            // 現在這個 > 比較會被當作 signed 比較
            data_out_reg <= (data_in > 0) ? data_in : 0;
        end
    end

    assign data_out = data_out_reg;

endmodule
