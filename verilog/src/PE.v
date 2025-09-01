`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/10 02:32:01
// Design Name: 
// Module Name: PE
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


module PE#(
    parameter DATA_WIDTH = 22,  // DSP: 25x18 multiplier
    parameter PORT_WIDTH = 8
    )(
    input wire signed [PORT_WIDTH-1:0] a,
    input wire signed [DATA_WIDTH-1:0] b,
    input wire signed [PORT_WIDTH-1:0] weight,
    input wire clk,
    input wire rst_n,
    input wire mode,
    output wire signed [PORT_WIDTH-1:0] a_out,
    output wire signed [DATA_WIDTH-1:0] b_out, 
    output wire signed [PORT_WIDTH-1:0] weight_out
    );

    
    reg signed [PORT_WIDTH-1:0] a_reg, a_pipe_1_reg, a_pipe_2_reg, a_pipe_3_reg;
    
    reg signed [DATA_WIDTH-1:0] b_reg, b_pipe_1_reg;
    // Internal weight register
    reg signed [PORT_WIDTH-1:0] weight_reg;

    wire signed [DATA_WIDTH-1:0] b_from_MAC;

    assign a_out = a_reg;
    assign b_out = b_reg;

    // instantiate xbip_multadd
    xbip_multadd_0 MAC_inst (
        .A(a),
        .B(weight_reg),
        .C(b_pipe_1_reg),
        .CLK(clk),
        .CE(1'b1),
        .SCLR(~rst_n),
        .SUBTRACT(1'b0),
        .P(b_from_MAC),
        .PCOUT()
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_reg <= 0;
            b_reg <= 0;
            a_pipe_1_reg <= 0;
            a_pipe_2_reg <= 0;
            b_pipe_1_reg <= 0;
            weight_reg <= 0;
        end else begin
            if(mode) begin  // weight preload
                weight_reg <= weight;
                a_reg <= 0;
                b_reg <= 0;
                a_pipe_1_reg <= 0;
                a_pipe_2_reg <= 0;
                b_pipe_1_reg <= 0;
            end else begin  // inference
                weight_reg <= weight_reg; // keep the weight register value
                a_pipe_1_reg <= a;
                a_pipe_2_reg <= a_pipe_1_reg;
                a_pipe_3_reg <= a_pipe_2_reg;
                a_reg <= a_pipe_3_reg;
                b_pipe_1_reg <= b;
                b_reg <= b_from_MAC;
            end
        end
    end

    assign weight_out = weight_reg;

endmodule
