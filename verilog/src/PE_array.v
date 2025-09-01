`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/10 14:23:09
// Design Name: 
// Module Name: PE_array
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


module PE_array #(
        parameter DATA_WIDTH = 22,
        parameter PORT_WIDTH = 8,
        parameter N = 5
    )(
        input  wire                          clk,
        input  wire                          rst_n,
        input wire WorI, // 1: weight preload, 0: inference

        // input data
        input wire signed [PORT_WIDTH-1:0] a0_in,
        input wire signed [PORT_WIDTH-1:0] a1_in,
        input wire signed [PORT_WIDTH-1:0] a2_in,
        input wire signed [PORT_WIDTH-1:0] a3_in,
        input wire signed [PORT_WIDTH-1:0] a4_in,

        // input bias
        input wire signed [PORT_WIDTH-1:0] b0_in,
        input wire signed [PORT_WIDTH-1:0] b1_in,
        input wire signed [PORT_WIDTH-1:0] b2_in,
        input wire signed [PORT_WIDTH-1:0] b3_in,
        input wire signed [PORT_WIDTH-1:0] b4_in,

        // input weight (5x5)
        input wire signed [PORT_WIDTH-1:0] weight_in,
        input wire [$clog2(N*N)-1:0] weight_location,

        output wire signed [DATA_WIDTH-1:0] y0_out,
        output wire signed [DATA_WIDTH-1:0] y1_out,
        output wire signed [DATA_WIDTH-1:0] y2_out,
        output wire signed [DATA_WIDTH-1:0] y3_out,
        output wire signed [DATA_WIDTH-1:0] y4_out,

        output wire signed [DATA_WIDTH+3:0] acc_out
        // test s0, s1, s2, s3
        // output wire signed [DATA_WIDTH:0] s0,
        // output wire signed [DATA_WIDTH+1:0] s1,
        // output wire signed [DATA_WIDTH+2:0] s2,
        // output wire signed [DATA_WIDTH+3:0] s3
        // output wire signed [PORT_WIDTH-1:0] a00_out,
        // output wire signed [PORT_WIDTH-1:0] a01_out,
        // output wire signed [PORT_WIDTH-1:0] a02_out,
        // output wire signed [PORT_WIDTH-1:0] a03_out,
        // output wire signed [PORT_WIDTH-1:0] a04_out,
        // output wire signed [PORT_WIDTH-1:0] a10_out,
        // output wire signed [PORT_WIDTH-1:0] a11_out,
        // output wire signed [PORT_WIDTH-1:0] a12_out,
        // output wire signed [PORT_WIDTH-1:0] a13_out,
        // output wire signed [PORT_WIDTH-1:0] a14_out,
        // output wire signed [PORT_WIDTH-1:0] a20_out,
        // output wire signed [PORT_WIDTH-1:0] a21_out,
        // output wire signed [PORT_WIDTH-1:0] a22_out,
        // output wire signed [PORT_WIDTH-1:0] a23_out,
        // output wire signed [PORT_WIDTH-1:0] a24_out,
        // output wire signed [PORT_WIDTH-1:0] a30_out,
        // output wire signed [PORT_WIDTH-1:0] a31_out,
        // output wire signed [PORT_WIDTH-1:0] a32_out,
        // output wire signed [PORT_WIDTH-1:0] a33_out,
        // output wire signed [PORT_WIDTH-1:0] a34_out,
        // output wire signed [PORT_WIDTH-1:0] a40_out,
        // output wire signed [PORT_WIDTH-1:0] a41_out,
        // output wire signed [PORT_WIDTH-1:0] a42_out,
        // output wire signed [PORT_WIDTH-1:0] a43_out,
        // output wire signed [PORT_WIDTH-1:0] a44_out,

        // output wire signed [DATA_WIDTH-1:0] b00_out,
        // output wire signed [DATA_WIDTH-1:0] b01_out,
        // output wire signed [DATA_WIDTH-1:0] b02_out,
        // output wire signed [DATA_WIDTH-1:0] b03_out,
        // output wire signed [DATA_WIDTH-1:0] b04_out,
        // output wire signed [DATA_WIDTH-1:0] b10_out,
        // output wire signed [DATA_WIDTH-1:0] b11_out,
        // output wire signed [DATA_WIDTH-1:0] b12_out,
        // output wire signed [DATA_WIDTH-1:0] b13_out,
        // output wire signed [DATA_WIDTH-1:0] b14_out,
        // output wire signed [DATA_WIDTH-1:0] b20_out,
        // output wire signed [DATA_WIDTH-1:0] b21_out,
        // output wire signed [DATA_WIDTH-1:0] b22_out,
        // output wire signed [DATA_WIDTH-1:0] b23_out,
        // output wire signed [DATA_WIDTH-1:0] b24_out,
        // output wire signed [DATA_WIDTH-1:0] b30_out,
        // output wire signed [DATA_WIDTH-1:0] b31_out,
        // output wire signed [DATA_WIDTH-1:0] b32_out,
        // output wire signed [DATA_WIDTH-1:0] b33_out,
        // output wire signed [DATA_WIDTH-1:0] b34_out,
        // output wire signed [DATA_WIDTH-1:0] b40_out,
        // output wire signed [DATA_WIDTH-1:0] b41_out,
        // output wire signed [DATA_WIDTH-1:0] b42_out,
        // output wire signed [DATA_WIDTH-1:0] b43_out,
        // output wire signed [DATA_WIDTH-1:0] b44_out,
        
        // output wire signed [PORT_WIDTH-1:0] w00_out,
        // output wire signed [PORT_WIDTH-1:0] w01_out,
        // output wire signed [PORT_WIDTH-1:0] w02_out,
        // output wire signed [PORT_WIDTH-1:0] w03_out,
        // output wire signed [PORT_WIDTH-1:0] w04_out,
        // output wire signed [PORT_WIDTH-1:0] w10_out,
        // output wire signed [PORT_WIDTH-1:0] w11_out,
        // output wire signed [PORT_WIDTH-1:0] w12_out,
        // output wire signed [PORT_WIDTH-1:0] w13_out,
        // output wire signed [PORT_WIDTH-1:0] w14_out,
        // output wire signed [PORT_WIDTH-1:0] w20_out,
        // output wire signed [PORT_WIDTH-1:0] w21_out,
        // output wire signed [PORT_WIDTH-1:0] w22_out,
        // output wire signed [PORT_WIDTH-1:0] w23_out,
        // output wire signed [PORT_WIDTH-1:0] w24_out,
        // output wire signed [PORT_WIDTH-1:0] w30_out,
        // output wire signed [PORT_WIDTH-1:0] w31_out,
        // output wire signed [PORT_WIDTH-1:0] w32_out,
        // output wire signed [PORT_WIDTH-1:0] w33_out,
        // output wire signed [PORT_WIDTH-1:0] w34_out,
        // output wire signed [PORT_WIDTH-1:0] w40_out,
        // output wire signed [PORT_WIDTH-1:0] w41_out,
        // output wire signed [PORT_WIDTH-1:0] w42_out,
        // output wire signed [PORT_WIDTH-1:0] w43_out,
        // output wire signed [PORT_WIDTH-1:0] w44_out
    );

    // declare the 2d interconnected wire of PE_array
    wire signed [PORT_WIDTH-1:0] a [0:N-1][0:N-1];
    wire signed [DATA_WIDTH-1:0] b [0:N-1][0:N-1];
    wire signed [PORT_WIDTH-1:0] w [0:N-1][0:N-1];
    wire signed [DATA_WIDTH-1:0] y [0:N-1];
    wire [N*N-1:0] pe_mode_signals;

    assign a[0][0] = a0_in;
    assign a[1][0] = a1_in;
    assign a[2][0] = a2_in;
    assign a[3][0] = a3_in;
    assign a[4][0] = a4_in;

    assign b[0][0] = {16'b0, b0_in};
    assign b[0][1] = {16'b0, b1_in};
    assign b[0][2] = {16'b0, b2_in};
    assign b[0][3] = {16'b0, b3_in};
    assign b[0][4] = {16'b0, b4_in};

    assign y0_out = y[0];
    assign y1_out = y[1];
    assign y2_out = y[2];
    assign y3_out = y[3];
    assign y4_out = y[4];

    // connect PE instances
    generate 
        for (genvar i = 0; i < N; i = i + 1) begin: PE_row
            for (genvar j = 0; j < N; j = j + 1) begin: PE
                localparam PE_ID = N * i + j;

                assign pe_mode_signals[PE_ID] = WorI && (weight_location == PE_ID);
                if(j == N-1 && i < N-1) begin // Right edge PEs (except bottom-right corner)
                    PE #(
                        .DATA_WIDTH(DATA_WIDTH),
                        .PORT_WIDTH(PORT_WIDTH)
                    ) pe_inst (
                        .clk(clk),
                        .rst_n(rst_n),
                        .mode(pe_mode_signals[PE_ID]),
                        .a(a[i][j]),
                        .b(b[i][j]),
                        .weight(weight_in),
                        .a_out(),
                        .b_out(b[i+1][j]),
                        .weight_out(w[i][j])
                    );
                end
                else if(i == N-1 && j < N-1) begin // Bottom edge PEs (except bottom-right corner)
                    PE #(
                        .DATA_WIDTH(DATA_WIDTH),
                        .PORT_WIDTH(PORT_WIDTH)
                    ) pe_inst (
                        .clk(clk),
                        .rst_n(rst_n),
                        .mode(pe_mode_signals[PE_ID]),
                        .a(a[i][j]),
                        .b(b[i][j]),
                        .weight(weight_in),
                        .a_out(a[i][j+1]),
                        .b_out(y[j]),
                        .weight_out(w[i][j])
                    );
                end
                else if(i == N-1 && j == N-1) begin // Bottom-right corner PE
                    PE #(
                        .DATA_WIDTH(DATA_WIDTH),
                        .PORT_WIDTH(PORT_WIDTH)
                    ) pe_inst (
                        .clk(clk),
                        .rst_n(rst_n),
                        .mode(pe_mode_signals[PE_ID]),
                        .a(a[i][j]),
                        .b(b[i][j]),
                        .weight(weight_in),
                        .a_out(),
                        .b_out(y[j]),
                        .weight_out(w[i][j])
                    );
                end
                else begin // Normal PEs
                    PE #(
                        .DATA_WIDTH(DATA_WIDTH),
                        .PORT_WIDTH(PORT_WIDTH)
                    ) pe_inst (
                        .clk(clk),
                        .rst_n(rst_n),
                        .mode(pe_mode_signals[PE_ID]),
                        .a(a[i][j]),
                        .b(b[i][j]),
                        .weight(weight_in),
                        .a_out(a[i][j+1]),
                        .b_out(b[i+1][j]),
                        .weight_out(w[i][j])
                    );
                end
            end
        end
    endgenerate

    reg signed [DATA_WIDTH-1:0] y0_pipe_0_reg, y0_pipe_1_reg, y0_pipe_2_reg, y0_pipe_3_reg, y0_pipe_4_reg;
    reg signed [DATA_WIDTH:0] s0_pipe_1_reg, s0_pipe_2_reg, s0_pipe_3_reg;
    reg signed [DATA_WIDTH+1:0] s1_pipe_1_reg, s1_pipe_2_reg, s1_pipe_3_reg;
    reg signed [DATA_WIDTH+2:0] s2_pipe_1_reg, s2_pipe_2_reg, s2_pipe_3_reg;
    reg signed [DATA_WIDTH+3:0] acc_out_reg;

    // assign a[i][j] to a(N*i+j)_out
    

    wire signed [DATA_WIDTH:0] s0;
    wire signed [DATA_WIDTH+1:0] s1;
    wire signed [DATA_WIDTH+2:0] s2;
    wire signed [DATA_WIDTH+3:0] s3;

    // assign a00_out = a[0][0];
    // assign a01_out = a[0][1];
    // assign a02_out = a[0][2];
    // assign a03_out = a[0][3];
    // assign a04_out = a[0][4];
    // assign a10_out = a[1][0];
    // assign a11_out = a[1][1];
    // assign a12_out = a[1][2];
    // assign a13_out = a[1][3];
    // assign a14_out = a[1][4];
    // assign a20_out = a[2][0];
    // assign a21_out = a[2][1];
    // assign a22_out = a[2][2];
    // assign a23_out = a[2][3];
    // assign a24_out = a[2][4];
    // assign a30_out = a[3][0];
    // assign a31_out = a[3][1];
    // assign a32_out = a[3][2];
    // assign a33_out = a[3][3];
    // assign a34_out = a[3][4];
    // assign a40_out = a[4][0];
    // assign a41_out = a[4][1];
    // assign a42_out = a[4][2];
    // assign a43_out = a[4][3];
    // assign a44_out = a[4][4];

    // assign b00_out = b[0][0];
    // assign b01_out = b[0][1];
    // assign b02_out = b[0][2];
    // assign b03_out = b[0][3];
    // assign b04_out = b[0][4];
    // assign b10_out = b[1][0];
    // assign b11_out = b[1][1];
    // assign b12_out = b[1][2];
    // assign b13_out = b[1][3];
    // assign b14_out = b[1][4];
    // assign b20_out = b[2][0];
    // assign b21_out = b[2][1];
    // assign b22_out = b[2][2];
    // assign b23_out = b[2][3];
    // assign b24_out = b[2][4];
    // assign b30_out = b[3][0];
    // assign b31_out = b[3][1];
    // assign b32_out = b[3][2];
    // assign b33_out = b[3][3];
    // assign b34_out = b[3][4];
    // assign b40_out = b[4][0];
    // assign b41_out = b[4][1];
    // assign b42_out = b[4][2];
    // assign b43_out = b[4][3];
    // assign b44_out = b[4][4];

    // assign w00_out = w[0][0];
    // assign w01_out = w[0][1];
    // assign w02_out = w[0][2];
    // assign w03_out = w[0][3];
    // assign w04_out = w[0][4];
    // assign w10_out = w[1][0];
    // assign w11_out = w[1][1];
    // assign w12_out = w[1][2];
    // assign w13_out = w[1][3];
    // assign w14_out = w[1][4];
    // assign w20_out = w[2][0];
    // assign w21_out = w[2][1];
    // assign w22_out = w[2][2];
    // assign w23_out = w[2][3];
    // assign w24_out = w[2][4];
    // assign w30_out = w[3][0];
    // assign w31_out = w[3][1];
    // assign w32_out = w[3][2];
    // assign w33_out = w[3][3];
    // assign w34_out = w[3][4];
    // assign w40_out = w[4][0];
    // assign w41_out = w[4][1];
    // assign w42_out = w[4][2];
    // assign w43_out = w[4][3];
    // assign w44_out = w[4][4];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out_reg <= 0;
            y0_pipe_0_reg <= 0;
            y0_pipe_1_reg <= 0;
            y0_pipe_2_reg <= 0;
            y0_pipe_3_reg <= 0;
            y0_pipe_4_reg <= 0;
            s0_pipe_1_reg <= 0;
            s0_pipe_2_reg <= 0;
            s0_pipe_3_reg <= 0;
            s1_pipe_1_reg <= 0;
            s1_pipe_2_reg <= 0;
            s1_pipe_3_reg <= 0;
            s2_pipe_1_reg <= 0;
            s2_pipe_2_reg <= 0;
            s2_pipe_3_reg <= 0;
        end else begin
            y0_pipe_0_reg <= y[0];
            y0_pipe_1_reg <= y0_pipe_0_reg;
            y0_pipe_2_reg <= y0_pipe_1_reg;
            y0_pipe_3_reg <= y0_pipe_2_reg;
            y0_pipe_4_reg <= y0_pipe_3_reg;
            s0_pipe_1_reg <= s0;
            s0_pipe_2_reg <= s0_pipe_1_reg;
            s0_pipe_3_reg <= s0_pipe_2_reg;
            s1_pipe_1_reg <= s1;
            s1_pipe_2_reg <= s1_pipe_1_reg;
            s1_pipe_3_reg <= s1_pipe_2_reg;
            s2_pipe_1_reg <= s2;
            s2_pipe_2_reg <= s2_pipe_1_reg;
            s2_pipe_3_reg <= s2_pipe_2_reg;
        end
    end

    // instantiate acc_adder

    acc_adder_22  acc_adder_y0y1_inst (
        .CLK(clk),
        .A(y0_pipe_4_reg),
        .B(y[1]),
        .S(s0),
        .CE(1'b1)
    );

    acc_adder_23  acc_adder_y1y2_inst (
        .CLK(clk),
        .A(s0_pipe_3_reg),
        .B({y[2][DATA_WIDTH-1], y[2]}),
        .S(s1),
        .CE(1'b1)
    );

    acc_adder_24  acc_adder_y2y3_inst (
        .CLK(clk),
        .A(s1_pipe_3_reg),
        .B({{2{y[3][DATA_WIDTH-1]}}, y[3]}),
        .S(s2),
        .CE(1'b1)
    );

    acc_adder_25  acc_adder_y3y4_inst (
        .CLK(clk),
        .A(s2_pipe_3_reg),
        .B({{3{y[4][DATA_WIDTH-1]}}, y[4]}),
        .S(s3),
        .CE(1'b1)
    );

    assign acc_out = s3;

endmodule
