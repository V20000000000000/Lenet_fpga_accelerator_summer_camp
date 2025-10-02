`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/22 18:15:29
// Design Name: 
// Module Name: tb_ram_buffer
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
// Testbench for ram_buffer
// Flow:
// 1. Read from pre-initialized RAM (loaded from .coe file by the simulator).
// 2. After some time, write new data into the RAM.
// 3. Continue reading to verify the newly written data.
//////////////////////////////////////////////////////////////////////////////////

module tb_ram_buffer;

    // --- 參數定義 (與 DUT 同步) ---
    parameter DATA_WIDTH        = 8;
    parameter ADDR_WIDTH        = 11;
    parameter MAX_WIDTH         = 32;
    parameter FEATURE_MAP1_SIZE = 32;
    parameter FEATURE_MAP2_SIZE = 28;
    parameter FEATURE_MAP3_SIZE = 14;
    parameter FEATURE_MAP4_SIZE = 10;
    parameter FEATURE_MAP5_SIZE = 5;
    parameter WAVEFRONT_DELAY   = 4;
    localparam RAM_DEPTH        = 1 << ADDR_WIDTH; // 2048

    // --- Testbench 內部信號 ---
    reg                         clk;
    reg                         rst_n;
    reg                         WorI_tb;
    reg [2:0]                   mode_tb;
    reg [ADDR_WIDTH-1:0]        read_addr_tb;
    reg                         en_tb;
    reg                         ram_write_en_tb;
    reg [ADDR_WIDTH-1:0]        ram_write_addr_tb;
    reg [DATA_WIDTH-1:0]        ram_write_data_tb;

    // --- DUT 輸出信號 ---
    wire [DATA_WIDTH-1:0]       l_out_0_wire, l_out_1_wire;
    wire [DATA_WIDTH-1:0]       out0_wire, out1_wire, out2_wire, out3_wire, out4_wire;
    wire [4:0]                  out_valid_wire;
    wire [ADDR_WIDTH-1:0]       read_addr_out_wire;
    wire [DATA_WIDTH-1:0]       weight_out_wire;
    wire [DATA_WIDTH-1:0]       ram_output_wire;

    integer i;

    // --- 實例化待測模組 (DUT) ---
    ram_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .MAX_WIDTH(MAX_WIDTH),
        .FEATURE_MAP1_SIZE(FEATURE_MAP1_SIZE),
        .FEATURE_MAP2_SIZE(FEATURE_MAP2_SIZE),
        .FEATURE_MAP3_SIZE(FEATURE_MAP3_SIZE),
        .FEATURE_MAP4_SIZE(FEATURE_MAP4_SIZE),
        .FEATURE_MAP5_SIZE(FEATURE_MAP5_SIZE),
        .WAVEFRONT_DELAY(WAVEFRONT_DELAY)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .WorI(WorI_tb),
        .mode(mode_tb),
        .read_addr(read_addr_tb),
        .en(en_tb),
        .ram_write_en(ram_write_en_tb),
        .ram_write_addr(ram_write_addr_tb),
        .ram_write_data(ram_write_data_tb),
        .l_out_0(l_out_0_wire),
        .l_out_1(l_out_1_wire),
        .out0(out0_wire),
        .out1(out1_wire),
        .out2(out2_wire),
        .out3(out3_wire),
        .out4(out4_wire),
        .out_valid(out_valid_wire),
        .read_addr_out(read_addr_out_wire),
        .weight_out(weight_out_wire),
        .ram_output(ram_output_wire)
    );

    // --- 時脈產生器 ---
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz 時脈
    end

    // --- 測試流程 ---
    initial begin
        // 1. 初始化和重置
        $display("--- 系統初始化與重置 ---");
        rst_n <= 1'b0;
        WorI_tb <= 1'b0;
        mode_tb <= 3'b000;
        read_addr_tb <= 0;
        en_tb <= 1'b0;
        ram_write_en_tb <= 1'b0;
        ram_write_addr_tb <= 0;
        ram_write_data_tb <= 0;
        #20;
        rst_n <= 1'b1;
        #20;

        // --- 階段 1: RAM 初始資料寫入 ---
        $display("\n--- 階段 1: RAM 初始資料寫入 ---");
        ram_write_en_tb <= 1'b1;
        // 寫入圖像和權重資料
        for (i = 0; i < 1024; i = i + 1) begin
            @(posedge clk);
            ram_write_addr_tb <= i;
            // 寫入一個可預測的 pattern，例如: i 的低8位
            ram_write_data_tb <= i[7:0]; 
        end
        @(posedge clk);
        ram_write_en_tb <= 1'b0;
        $display("--- 階段 1: 資料寫入完成 ---\n");
        #100;

        // --- 階段 2: 權重預載入測試 (Weight Preload Read Test) ---
        $display("--- 階段 2: 權重讀取模式測試 ---");
        WorI_tb <= 1'b1; // 設定為權重模式
        en_tb <= 1'b1;   // 致能讀取
        
        // 從位址 800 開始讀取 25 個權重 (模擬 5x5 核心)
        for (i = 0; i < 25; i = i + 1) begin
            @(posedge clk);
            read_addr_tb <= 800 + i;
        end
        
        @(posedge clk);
        read_addr_tb <= 0;
        en_tb <= 1'b0;    // 關閉讀取
        WorI_tb <= 1'b0;  // 切回推論模式
        $display("--- 階段 2: 權重讀取測試完成 ---\n");
        #100;

        // --- 階段 3: 推論模式讀取測試 (Inference Mode Read Test) ---
        // 測試不同的 mode
        run_inference_test(3'b000, FEATURE_MAP1_SIZE * FEATURE_MAP1_SIZE); // 32x32
        run_inference_test(3'b001, FEATURE_MAP2_SIZE * FEATURE_MAP2_SIZE); // 28x28
        run_inference_test(3'b010, FEATURE_MAP3_SIZE * FEATURE_MAP3_SIZE); // 14x14
        run_inference_test(3'b011, FEATURE_MAP4_SIZE * FEATURE_MAP4_SIZE); // 10x10
        run_inference_test(3'b100, FEATURE_MAP5_SIZE * FEATURE_MAP5_SIZE); // 5x5

        $display("--- 所有測試完成 ---");
        $finish;
    end

    // --- 推論測試任務 ---
    task run_inference_test;
        input [2:0] current_mode;
        input integer read_count;
        begin
            $display("--- 階段 3: 推論模式測試 (mode = %b) ---", current_mode);
            #100;
            WorI_tb <= 1'b0;      // 確保為推論模式
            mode_tb <= current_mode;
            en_tb <= 1'b1;        // 致能讀取
            
            // 從位址 0 開始讀取特徵圖資料
            for (i = 0; i < read_count; i = i + 1) begin
                @(posedge clk);
                read_addr_tb <= i;
            end
            
            @(posedge clk);
            en_tb <= 1'b0;        // 關閉讀取
            read_addr_tb <= 0;
            $display("--- 推論模式測試 (mode = %b) 完成 ---\n", current_mode);
            #100;
        end
    endtask

    // --- 監控輸出 ---
    initial begin
        $monitor("Time=%0t | rst=%b | WorI=%b | mode=%b | en=%b | rd_addr=%4d | ram_out=%3d | weight_out=%3d | l_out0=%3d | out0=%3d | out1=%3d | out2=%3d | out3=%3d | out4=%3d",
                $time, rst_n, WorI_tb, mode_tb, en_tb, read_addr_out_wire, ram_output_wire, weight_out_wire, 
                l_out_0_wire, out0_wire, out1_wire, out2_wire, out3_wire, out4_wire);
    end

endmodule