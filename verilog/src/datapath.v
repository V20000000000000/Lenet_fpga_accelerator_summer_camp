`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/25 00:06:18
// Design Name: 
// Module Name: datapath
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


module datapath #(
    // --- 繼承自 ram_buffer 的參數 ---
    parameter DATA_WIDTH        = 8,
    parameter ADDR_WIDTH        = 11,
    parameter MAX_WIDTH         = 32,
    
    // --- 繼承自 PE_array 的參數 ---
    parameter PE_DATA_WIDTH     = 22,
    parameter PE_PORT_WIDTH     = 8,
    parameter N                 = 5
)(
    // --- Global Clocks and Reset ---
    input wire                      clk,          // 主時鐘 (供給 PE_array 和 ram_buffer 的 r_clk)
    input wire                      rst_n,        // 非同步低電位重置

    // --- Data Inputs (for RAM) ---
    input wire                      ram_write_en,
    input wire [DATA_WIDTH-1:0]     ram_write_data,

    output wire [DATA_WIDTH-1:0]    ram_output_wire,

    // --- Control Signals from Controller ---
    input wire                      ctrl_ram_en,
    input wire                      ctrl_addr_ctrl_en,
    input wire                      ctrl_WorI,
    input wire [2:0]                ctrl_mode,
    input wire [ADDR_WIDTH-1:0]     ctrl_read_addr,
    input wire [$clog2(N*N)-1:0]    ctrl_weight_location,
    input wire [1:0]                  ctrl_mux_sel,

    output wire signed [PE_DATA_WIDTH+3:0] acc_out,
    output wire signed [DATA_WIDTH-1:0] mux_out,
    output wire [ADDR_WIDTH-1:0] ram_write_addr,
    output wire signed [DATA_WIDTH-1:0] weight_data_from_ram,
    output wire signed [DATA_WIDTH-1:0] relu_out,

    // --- Status Signals to Controller (if needed) ---
    output wire [4:0]               out_valid,
    output wire                     valid_out,
    output wire [2:0]               state_out,
    output wire [ADDR_WIDTH-1:0]    row_counter_out,
    output wire [ADDR_WIDTH-1:0]    col_counter_out,

    output wire [DATA_WIDTH-1:0] l_out_0,
    output wire [DATA_WIDTH-1:0] l_out_1
);


    // --- 內部連線 ---
    // ram_buffer -> PE_array 的數據線
    wire [DATA_WIDTH-1:0]   ram_out0, ram_out1, ram_out2, ram_out3, ram_out4;
    // wire [ADDR_WIDTH-1:0]   ram_write_addr;
    wire [PE_DATA_WIDTH-1:0]   y0_out, y1_out, y2_out, y3_out, y4_out;
    // wire [DATA_WIDTH-1:0]   l_out_0, l_out_1;
    wire [DATA_WIDTH-1:0]   ram_out;
    reg [$clog2(N*N)-1:0]    ctrl_weight_location_pipe [0:1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_weight_location_pipe[0] <= 0;
            ctrl_weight_location_pipe[1] <= 0;
        end else begin
            ctrl_weight_location_pipe[0] <= ctrl_weight_location;
            ctrl_weight_location_pipe[1] <= ctrl_weight_location_pipe[0];
        end
    end

    assign ram_output_wire = ram_out;

    // --- 1. 實例化 ram_buffer ---
    ram_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .MAX_WIDTH(MAX_WIDTH)
        // 您可以根據需要繼續傳遞其他 feature map 尺寸參數
    ) ram_buffer_inst (
        .clk(clk),
        .rst_n(rst_n),
        
        // --- 來自 Controller 的控制信號 ---
        .en(ctrl_ram_en),
        .WorI(ctrl_WorI),
        .mode(ctrl_mode),
        .read_addr(ctrl_read_addr),

        // --- 來自上層的 RAM 寫入信號 ---
        .ram_write_en(ram_write_en),
        .ram_write_addr(ram_write_addr),
        .ram_write_data(ram_write_data),
        .ram_output(ram_out),

        // --- Line Buffer ---
        .l_out_0(l_out_0),
        .l_out_1(l_out_1),

        // --- 輸出到 PE_array 或上層 ---
        .out0(ram_out0),
        .out1(ram_out1),
        .out2(ram_out2),
        .out3(ram_out3),
        .out4(ram_out4),
        .out_valid(out_valid),
        .weight_out(weight_data_from_ram),
        .read_addr_out()
    );

    wire [DATA_WIDTH-1:0] max_pool_out;

    max_Pool #(
        .DATA_WIDTH(DATA_WIDTH)
    ) max_pool_inst (
        .clk(clk),
        .rst_n(rst_n),
        .line_1(l_out_0),
        .line_2(l_out_1),
        .max_out(max_pool_out)
    );

    Relu #(
        .DATA_WIDTH(DATA_WIDTH)
    ) relu_inst (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(ram_out),
        .data_out(relu_out) // 直接連接到 mux 的輸入
    );

    // wire [DATA_WIDTH-1:0] mux_out;

    Mux_8bit mux_inst (
        .in0({acc_out[7:0]}),
        .in1(max_pool_out),
        .in2(relu_out),
        .in3(8'd255),
        .sel(ctrl_mux_sel),
        .clk(clk),
        .out(mux_out)
    );

    addr_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) addr_controller_inst (
        .clk(clk),
        .rst_n(rst_n),
        .en(ctrl_addr_ctrl_en),
        .ctrl_mode(ctrl_mode),
        .ctrl_read_addr(ctrl_read_addr),
        .ctrl_write_addr(ram_write_addr),
        .valid_out(valid_out),
        .state_out(state_out),
        .row_counter_out(row_counter_out),
        .col_counter_out(col_counter_out)
    );

    // --- 2. 實例化 PE_array ---
    PE_array #(
        .DATA_WIDTH(PE_DATA_WIDTH),
        .PORT_WIDTH(PE_PORT_WIDTH),
        .N(N)
    ) pe_array_inst (
        .clk(~clk),
        .rst_n(rst_n),

        // --- 來自 Controller 的控制信號 ---
        .WorI(ctrl_WorI),
        .weight_location(ctrl_weight_location_pipe[1]),

        // --- 來自 ram_buffer 的數據輸入 ---
        .a0_in({ram_out0}),
        .a1_in({ram_out1}),
        .a2_in({ram_out2}),
        .a3_in({ram_out3}),
        .a4_in({ram_out4}),
        .weight_in(weight_data_from_ram),

        // --- 來自上層的 Bias 輸入 ---
        .b0_in(8'b0),
        .b1_in(8'b0),
        .b2_in(8'b0),
        .b3_in(8'b0),
        .b4_in(8'b0),

        // --- 最終運算結果輸出 ---
        .y0_out(y0_out),
        .y1_out(y1_out),
        .y2_out(y2_out),
        .y3_out(y3_out),
        .y4_out(y4_out),
        .acc_out(acc_out)
        


    );

endmodule