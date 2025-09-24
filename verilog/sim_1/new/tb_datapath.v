`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/25 00:40:14
// Design Name: 
// Module Name: tb_datapath
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
// Module Name: tb_datapath
// Description: Testbench for the datapath module.
//  - Phase 1: Write feature map and weight data into the internal RAM.
//  - Phase 2: Test the weight preload functionality.
//  - Phase 3: Test the inference functionality.
//////////////////////////////////////////////////////////////////////////////////

module tb_datapath;

    // --- 參數定義 (與 datapath 模組一致) ---
    parameter DATA_WIDTH        = 8;
    parameter ADDR_WIDTH        = 11;
    parameter MAX_WIDTH         = 32;
    parameter PE_DATA_WIDTH     = 22;
    parameter PE_PORT_WIDTH     = 8;
    parameter N                 = 5;
    localparam RAM_DEPTH        = 1 << ADDR_WIDTH;
    localparam WEIGHT_START_ADDR = 11'd1200; // 假設權重存放在位址 1200 之後
    localparam MODE_SIZE        = 3'b000;    // 假設 mode 1 會驅動 line buffer

    // --- Testbench 內部信號 ---
    // Inputs to DUT
    reg                           clk;
    reg                           rst_n;
    reg                           ram_write_en;
    wire [ADDR_WIDTH-1:0]         ram_write_addr;
    reg [DATA_WIDTH-1:0]          ram_write_data;
    reg                           ctrl_ram_en;
    reg                           ctrl_WorI;
    reg [2:0]                     ctrl_mode;
    reg [ADDR_WIDTH-1:0]          ctrl_read_addr;
    reg [$clog2(N*N)-1:0]         ctrl_weight_location;
    reg [1:0]                     ctrl_mux_sel;
    reg                           ctrl_addr_ctrl_en;
    reg                           en; // 確保 en 信號已宣告
    wire [DATA_WIDTH-1:0]         result;
    // wire [DATA_WIDTH-1:0]           mux_out;
    // wire [DATA_WIDTH-1:0]           relu_out;
    // wire                            valid_out;
    // wire [1:0]                      state_out; // 修正寬度
    // wire [ADDR_WIDTH-1:0]           row_counter_out;
    // wire [ADDR_WIDTH-1:0]           col_counter_out;
    // wire signed [DATA_WIDTH - 1:0]  ram_output_wire;
    // wire [DATA_WIDTH-1:0]           l_out_0;
    // wire [DATA_WIDTH-1:0]           l_out_1; 

    integer i;

    // --- 實例化待測模組 (DUT: Datapath) ---
    datapath #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .MAX_WIDTH(MAX_WIDTH),
        .PE_DATA_WIDTH(PE_DATA_WIDTH),
        .PE_PORT_WIDTH(PE_PORT_WIDTH),
        .N(N)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .ram_write_en(ram_write_en),
        // .ram_write_addr(ram_write_addr),
        // .ram_write_data(ram_write_data),
        // .ram_output_wire(ram_output_wire),
        // .weight_data_from_ram(weight_data_from_ram),
        .ctrl_addr_ctrl_en(ctrl_addr_ctrl_en),
        .ctrl_mux_sel(ctrl_mux_sel),
        .ctrl_ram_en(ctrl_ram_en),
        .ctrl_WorI(ctrl_WorI),
        .ctrl_mode(ctrl_mode),
        .ctrl_read_addr(ctrl_read_addr),
        .ctrl_weight_location(ctrl_weight_location),
        .result(result)
        // .acc_out(acc_out),
        // .out_valid(out_valid),
        // // .mux_out(mux_out),
        // // .relu_out(relu_out),
        // .valid_out(valid_out),
        // .state_out(state_out),
        // .row_counter_out(row_counter_out),
        // .col_counter_out(col_counter_out)
        // .l_out_0(l_out_0),
        // .l_out_1(l_out_1)
    );

    // --- 時脈產生器 ---
    initial begin
        clk = 0;
        forever #10 clk = ~clk;
    end

    // --- 測試流程 ---
    initial begin
        // 1. 初始化和重置
        rst_n = 1'b0;
        ram_write_en = 1'b0;
        ram_write_data = 0;
        ctrl_ram_en = 1'b0;
        ctrl_WorI = 1'b0;
        ctrl_mode = MODE_SIZE;
        ctrl_read_addr = 0;
        ctrl_weight_location = 0;
        ctrl_mux_sel = 2'b00;
        ctrl_addr_ctrl_en = 1'b0;
        en = 0;
        #20;
        rst_n = 1'b1;
        en = 1;
        #500;


        // --- 階段 2: 權重預載入測試 ---
        $display("--- [%0t] 階段 2: 權重預載入測試 (Weight Preload) ---", $time);
        ctrl_WorI = 1'b1; // 設定為權重模式
        ctrl_ram_en = 1'b1; // 致能讀取
        #10;
        for (i = 0; i < (N*N); i = i + 1) begin
            @(posedge clk);
            ctrl_read_addr = WEIGHT_START_ADDR + i;
            ctrl_weight_location = i;
        end
        
        @(posedge clk);
        ctrl_ram_en = 1'b1; // 關閉讀取
        #60;
        ctrl_WorI = 1'b0; // 設定為推論模式
        $display("--- [%0t] 階段 2: 權重預載入完成 ---\n", $time);
        #100;
        ctrl_addr_ctrl_en = 1'b0;
        ctrl_read_addr = 0;
        #100;

        // --- 階段 3: 推論模式讀取與運算測試 ---
        $display("--- [%0t] 階段 3: 推論模式讀取測試 (Inference) ---", $time);
        ctrl_mode = 3'b000;
        ctrl_mux_sel = 2'b00;
        ctrl_ram_en = 1'b1;   // 致能讀取
        ctrl_addr_ctrl_en = 1'b1; // 啟用地址控制器
        
        for (i = 0; i < 1200; i = i + 1) begin
            @(posedge clk);
            ctrl_read_addr = i;
        end
        
        @(posedge clk);
        ctrl_ram_en = 1'b0; // 關閉讀取
        $display("--- [%0t] 階段 3: 推論模式讀取完成 ---\n", $time);
        
        #200;

        // 讀取ram 0 - 1023 的數據
        ctrl_addr_ctrl_en = 1'b0;
        ctrl_read_addr = 0;
        ctrl_mode = 3'b000;
        ctrl_mux_sel = 2'b00;
        ctrl_ram_en = 1'b1; // 致能讀取
        for (i = 0; i < 1024; i = i + 1) begin
            @(posedge clk);
            ctrl_read_addr = i;
        end
        @(posedge clk);
        ctrl_ram_en = 1'b0; // 關閉讀取
        #100;


        // --- 階段 4: Max Pooling 測試 1 (Mode 5) ---
        $display("--- [%0t] 階段 4: Max Pooling 測試 1 (Mode 5) ---", $time);
        ctrl_mode = 3'b101;
        ctrl_mux_sel = 2'b01;
        ctrl_ram_en = 1'b1;
        ctrl_addr_ctrl_en = 1'b1; // 啟用地址控制器
        for (i = 0; i < 1200; i = i + 1) begin
            @(posedge clk);
            ctrl_read_addr = i;
        end
        @(posedge clk);
        ctrl_ram_en = 1'b0;
        $display("--- [%0t] 階段 4: Max Pooling 1 測試完成 ---\n", $time);
        #100;

        // --- 階段 5: ReLU 測試 ---
        $display("--- [%0t] 階段 5: ReLU 測試 (sel=10) ---", $time);
        ctrl_mode = 3'b111; // 使用簡單模式讀取
        ctrl_mux_sel = 2'b10;
        ctrl_ram_en = 1'b1;
        for (i = 0; i < 300; i = i + 1) begin
            @(posedge clk);
            ctrl_read_addr = i;
        end
        @(posedge clk);
        ctrl_ram_en = 1'b0;
        $display("--- [%0t] 階段 5: ReLU 測試完成 ---\n", $time);
        #100;

        // // --- 階段 4: Max Pooling 測試 2 (Mode 6) ---
        // $display("--- [%0t] 階段 4: Max Pooling 測試 2 (Mode 6) ---", $time);
        // ctrl_mode <= 3'b110;
        // ctrl_mux_sel <= 2'b01;
        // ctrl_ram_en <= 1'b1;
        // for (i = 0; i < 2048; i = i + 1) begin
        //     @(posedge clk);
        //     ctrl_read_addr <= i;
        // end
        // @(posedge clk);
        // ctrl_ram_en <= 1'b0;
        // $display("--- [%0t] 階段 4: Max Pooling 1 測試完成 ---\n", $time);
        // #100;

        

        $display("--- [%0t] Testbench 所有測試結束 ---", $time);
        $finish;
    end

endmodule