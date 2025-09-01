`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/18 23:39:56
// Design Name: 
// Module Name: line_buffer
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


module line_buffer#(
    // --- 參數 ---
    parameter DATA_WIDTH          = 8,
    // 將 MAX_WIDTH 設為可能的最大特徵圖尺寸
    parameter MAX_WIDTH           = 32,
    // 不同模式下的特定特徵圖尺寸
    parameter FEATURE_MAP1_SIZE   = 32,
    parameter FEATURE_MAP2_SIZE   = 28,
    parameter FEATURE_MAP3_SIZE   = 14,
    parameter FEATURE_MAP4_SIZE   = 10,
    parameter FEATURE_MAP5_SIZE   = 5,
    parameter WAVEFRONT_DELAY = 4
)(
    // --- 埠 ---
    input  wire                          clk,
    input  wire                          rst_n, // 低電位有效重置
    input  wire  [2:0]                   mode,
    input  wire  [DATA_WIDTH-1:0]        data_in,

    // 5x1 垂直像素列輸出 (共 5 個像素)
    output wire [DATA_WIDTH-1:0]         line_out_0, // 最舊的行 (Top)
    output wire [DATA_WIDTH-1:0]         line_out_1,
    output wire [DATA_WIDTH-1:0]         line_out_2,
    output wire [DATA_WIDTH-1:0]         line_out_3,
    output wire [DATA_WIDTH-1:0]         line_out_4  // 最新的行 (Bottom, 等同於 data_in)
);

    // --- 內部信號和暫存器 ---
    // Part 1: 標準 Line Buffer 邏輯 (此部分完全不變)
    reg [DATA_WIDTH-1:0] line1_buffer [0:MAX_WIDTH-1];
    reg [DATA_WIDTH-1:0] line2_buffer [0:MAX_WIDTH-1];
    reg [DATA_WIDTH-1:0] line3_buffer [0:MAX_WIDTH-1];
    reg [DATA_WIDTH-1:0] line4_buffer [0:MAX_WIDTH-1];

    reg [$clog2(MAX_WIDTH)-1:0] col_counter;
    reg [$clog2(MAX_WIDTH)-1:0] active_line_width;
    integer i;

    // out register
    reg [DATA_WIDTH-1:0] line_out [0:4];

    // --- 邏輯實現 ---

    // 模式解碼器 (不變)
    always @(*) begin
        case (mode)
            3'b000: active_line_width = FEATURE_MAP1_SIZE;
            3'b001: active_line_width = FEATURE_MAP2_SIZE;
            3'b010: active_line_width = FEATURE_MAP3_SIZE;
            3'b011: active_line_width = FEATURE_MAP4_SIZE;
            3'b100: active_line_width = FEATURE_MAP5_SIZE;
            3'b101: active_line_width = FEATURE_MAP2_SIZE;
            3'b110: active_line_width = FEATURE_MAP4_SIZE;
            default: active_line_width = FEATURE_MAP1_SIZE;
        endcase
    end

    // Line Buffer 核心邏輯 (不變)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col_counter <= 0;
            for (i = 0; i < MAX_WIDTH; i = i + 1) begin
                line1_buffer[i] <= 0; line2_buffer[i] <= 0;
                line3_buffer[i] <= 0; line4_buffer[i] <= 0;
            end
        end else begin
            line4_buffer[col_counter] <= line3_buffer[col_counter];
            line3_buffer[col_counter] <= line2_buffer[col_counter];
            line2_buffer[col_counter] <= line1_buffer[col_counter];
            line1_buffer[col_counter] <= data_in;

            if (col_counter == active_line_width - 1) begin
                col_counter <= 0;
            end else begin
                col_counter <= col_counter + 1;
            end
        end
    end

    // 從 Line Buffer 中讀取即時的滑動窗口數據 (不變)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            line_out[0] <= 0;
            line_out[1] <= 0;
            line_out[2] <= 0;
            line_out[3] <= 0;
            line_out[4] <= 0;
        end else begin
            line_out[0] <= line4_buffer[col_counter];
            line_out[1] <= line3_buffer[col_counter];
            line_out[2] <= line2_buffer[col_counter];
            line_out[3] <= line1_buffer[col_counter];
            line_out[4] <= data_in;
        end
    end

    // 輸出埠連接
    assign line_out_0 = line_out[0];
    assign line_out_1 = line_out[1];
    assign line_out_2 = line_out[2];
    assign line_out_3 = line_out[3];
    assign line_out_4 = line_out[4];

endmodule