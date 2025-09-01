`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/30 01:15:27
// Design Name: 
// Module Name: delay
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


module delay#(
        DATA_WIDTH = 8,
        WAVEFRONT_DELAY = 4
    )(
        input wire clk,
        input wire rst_n,
        input wire [DATA_WIDTH-1:0] line_0_in,
        input wire [DATA_WIDTH-1:0] line_1_in,
        input wire [DATA_WIDTH-1:0] line_2_in,
        input wire [DATA_WIDTH-1:0] line_3_in,
        input wire [DATA_WIDTH-1:0] line_4_in,
        output wire [DATA_WIDTH-1:0] line_0_out,
        output wire [DATA_WIDTH-1:0] line_1_out,
        output wire [DATA_WIDTH-1:0] line_2_out,
        output wire [DATA_WIDTH-1:0] line_3_out,
        output wire [DATA_WIDTH-1:0] line_4_out
    );

    reg [DATA_WIDTH-1:0] delay_buffer_1 [0:WAVEFRONT_DELAY-1];
    reg [DATA_WIDTH-1:0] delay_buffer_2 [0:2*WAVEFRONT_DELAY-1];
    reg [DATA_WIDTH-1:0] delay_buffer_3 [0:3*WAVEFRONT_DELAY-1];
    reg [DATA_WIDTH-1:0] delay_buffer_4 [0:4*WAVEFRONT_DELAY-1];
    reg [DATA_WIDTH-1:0] out_reg [0:4];

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < WAVEFRONT_DELAY; i = i + 1) delay_buffer_1[i] <= 0;
            for (i = 0; i < 2*WAVEFRONT_DELAY; i = i + 1) delay_buffer_2[i] <= 0;
            for (i = 0; i < 3*WAVEFRONT_DELAY; i = i + 1) delay_buffer_3[i] <= 0;
            for (i = 0; i < 4*WAVEFRONT_DELAY; i = i + 1) delay_buffer_4[i] <= 0;
        end else begin
            delay_buffer_1[0] <= line_1_in;
            delay_buffer_2[0] <= line_2_in;
            delay_buffer_3[0] <= line_3_in;
            delay_buffer_4[0] <= line_4_in;
            
            for (i = 1; i < WAVEFRONT_DELAY; i = i + 1) delay_buffer_1[i] <= delay_buffer_1[i-1];
            for (i = 1; i < 2*WAVEFRONT_DELAY; i = i + 1) delay_buffer_2[i] <= delay_buffer_2[i-1];
            for (i = 1; i < 3*WAVEFRONT_DELAY; i = i + 1) delay_buffer_3[i] <= delay_buffer_3[i-1];
            for (i = 1; i < 4*WAVEFRONT_DELAY; i = i + 1) delay_buffer_4[i] <= delay_buffer_4[i-1];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_reg[0] <= 0;
            out_reg[1] <= 0;
            out_reg[2] <= 0;
            out_reg[3] <= 0;
            out_reg[4] <= 0;
        end else begin
            out_reg[0] <= line_0_in; // line 0 不延遲
            out_reg[1] <= delay_buffer_1[WAVEFRONT_DELAY-1];
            out_reg[2] <= delay_buffer_2[2*WAVEFRONT_DELAY-1];
            out_reg[3] <= delay_buffer_3[3*WAVEFRONT_DELAY-1];
            out_reg[4] <= delay_buffer_4[4*WAVEFRONT_DELAY-1];
        end
    end

    assign line_0_out = out_reg[0];
    assign line_1_out = out_reg[1];
    assign line_2_out = out_reg[2];
    assign line_3_out = out_reg[3];
    assign line_4_out = out_reg[4];

endmodule
