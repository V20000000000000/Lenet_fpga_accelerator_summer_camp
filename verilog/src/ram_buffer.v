`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/22 18:10:45
// Design Name: 
// Module Name: ram_buffer
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

//////////////////////////////////////////////////////////////////////////////////
// Design Name: ram_buffer
// Description: Connects an image RAM to a Line Buffer module.
//              It controls reading from the RAM and feeding data into the buffer.
//////////////////////////////////////////////////////////////////////////////////

module ram_buffer #(
    // --- 頂層參數 ---
    parameter DATA_WIDTH        = 8,
    parameter ADDR_WIDTH        = 11,
    parameter MAX_WIDTH         = 32,
    
    // --- 特徵圖尺寸參數 ---
    parameter FEATURE_MAP1_SIZE   = 32,
    parameter FEATURE_MAP2_SIZE   = 28,
    parameter FEATURE_MAP3_SIZE   = 14,
    parameter FEATURE_MAP4_SIZE   = 10,
    parameter FEATURE_MAP5_SIZE   = 5,
    
    // --- 波前延遲參數 ---
    parameter WAVEFRONT_DELAY   = 4
)(
    // --- 外部接口 ---
    input wire                      clk,
    input wire                      rst_n,
    input wire                      WorI,   // 1: weight preload, 0: inference
    input wire [2:0]                mode,
    input wire [ADDR_WIDTH-1:0]     read_addr,
    input wire                      ram_write_en,
    input wire [ADDR_WIDTH-1:0]     ram_write_addr,
    input wire [DATA_WIDTH-1:0]     ram_write_data,
    input wire                      en, // 來自外部 Controller 的全局致能信號

    // --- 輸出 ---
    output wire [DATA_WIDTH-1:0] l_out_0,
    output wire [DATA_WIDTH-1:0] l_out_1,
    output wire [DATA_WIDTH-1:0]    out0,
    output wire [DATA_WIDTH-1:0]    out1,
    output wire [DATA_WIDTH-1:0]    out2,
    output wire [DATA_WIDTH-1:0]    out3,
    output wire [DATA_WIDTH-1:0]    out4,
    output wire [4:0]               out_valid,
    output wire [ADDR_WIDTH-1:0]    read_addr_out,
    output wire [DATA_WIDTH-1:0]    weight_out,
    output wire [DATA_WIDTH-1:0]    ram_output
);

    // --- 內部連線和訊號 ---
    wire [DATA_WIDTH-1:0]           ram_dout;
    wire [DATA_WIDTH-1:0] lb_out0, lb_out1, lb_out2, lb_out3, lb_out4;

    // --- 波前延遲邏輯的移位暫存器 ---
    reg [DATA_WIDTH-1:0] sr1 [0:WAVEFRONT_DELAY-1];       // 4級，用於 out1
    reg [DATA_WIDTH-1:0] sr2 [0:2*WAVEFRONT_DELAY-1];     // 8級，用於 out2
    reg [DATA_WIDTH-1:0] sr3 [0:3*WAVEFRONT_DELAY-1];     // 12級，用於 out3
    reg [DATA_WIDTH-1:0] sr4 [0:4*WAVEFRONT_DELAY-1];     // 16級，用於 out4
    // reg [DATA_WIDTH-1:0] out_reg [0:4];
    reg [1:0] WorI_pipe;

    // reg [4:0] out_valid_reg;

    assign ram_output = ram_dout;

    // 將輸入的 read_addr 連接到輸出埠，方便觀察
    assign read_addr_out = read_addr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            WorI_pipe <= 2'b00;
        end else begin
            WorI_pipe <= {WorI_pipe[0], WorI};
        end
    end

    // --- 實例化 RAM Wrapper ---
    image_ram_wrapper #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) ram_inst (
        .clka(clk), .ena(ram_write_en), .wea(ram_write_en), .addra(ram_write_addr), .dina(ram_write_data),
        .clkb(clk), .enb(en), .addrb(read_addr), .doutb(ram_dout)
    );
    wire [DATA_WIDTH-1:0] line_buffer_in;
    assign weight_out = (WorI_pipe[1]) ? ram_dout : 0;
    assign line_buffer_in = (~WorI_pipe[1]) ? ram_dout : 0;

    // --- 實例化 Line Buffer ---
    // **重要**: 確保您的 line_buffer 模組有 'en' 埠
    line_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .MAX_WIDTH(MAX_WIDTH),
        .FEATURE_MAP1_SIZE(FEATURE_MAP1_SIZE)
    ) line_buffer_inst (
        .clk(clk),
        .rst_n(rst_n),
        .mode(mode),
        .data_in(line_buffer_in),

        .line_out_0(lb_out0),
        .line_out_1(lb_out1),
        .line_out_2(lb_out2),
        .line_out_3(lb_out3),
        .line_out_4(lb_out4)
    );

    assign l_out_0 = lb_out0;
    assign l_out_1 = lb_out1;

    delay #(
        .DATA_WIDTH(DATA_WIDTH),
        .WAVEFRONT_DELAY(WAVEFRONT_DELAY)
    ) delay_inst (
        .clk(clk),
        .rst_n(rst_n),
        .line_0_in(lb_out0),
        .line_1_in(lb_out1),
        .line_2_in(lb_out2),
        .line_3_in(lb_out3),
        .line_4_in(lb_out4),
        .line_0_out(out0),
        .line_1_out(out1),
        .line_2_out(out2),
        .line_3_out(out3),
        .line_4_out(out4)
    );

    assign out_valid = 0;

endmodule