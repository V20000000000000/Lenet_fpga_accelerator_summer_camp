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

    // --- 參數定義 ---
    parameter DATA_WIDTH        = 8;
    parameter ADDR_WIDTH        = 10;
    parameter MAX_WIDTH         = 28;
    parameter FEATURE_MAP1_SIZE = 28;
    localparam RAM_DEPTH        = 1 << ADDR_WIDTH; // 1024
    wire [4:0] out_valid;

    // --- Testbench 內部信號 ---
    reg                         clk;
    reg                         rst_n;
    reg                         WorI_tb; // 新增 WorI 控制信號
    reg [2:0]                   mode;
    reg [ADDR_WIDTH-1:0]        tb_read_addr;
    reg                         tb_en; // 重新命名 en 以更清晰
    reg                         tb_ram_write_en;
    reg [ADDR_WIDTH-1:0]        tb_ram_write_addr;
    reg [DATA_WIDTH-1:0]        tb_ram_write_data;

    wire [DATA_WIDTH-1:0]       out0, out1, out2, out3, out4;
    wire [ADDR_WIDTH-1:0]       read_addr_out;
    wire [DATA_WIDTH-1:0]       weight_out_wire;
    wire [DATA_WIDTH-1:0]       ram_output_wire;
    wire [DATA_WIDTH-1:0]       line_buffer_in_wire;
    integer i;

    // --- 實例化待測模組 (DUT) ---
    ram_buffer #(
        .DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .MAX_WIDTH(MAX_WIDTH),
        .FEATURE_MAP1_SIZE(FEATURE_MAP1_SIZE)
    ) dut (
        .r_clk(clk), 
        .w_clk(clk), 
        .rst_n(rst_n), 
        .WorI(WorI_tb), // 連接 WorI 控制信號
        .mode(mode),
        .read_addr(tb_read_addr), 
        .en(tb_en), // 連接致能信號
        .ram_write_en(tb_ram_write_en), 
        .ram_write_addr(tb_ram_write_addr), 
        .ram_write_data(tb_ram_write_data),
        .out0(out0), .out1(out1), .out2(out2), .out3(out3), .out4(out4),
        .read_addr_out(read_addr_out),
        .weight_out(weight_out_wire),
        .ram_output(ram_output_wire),
        .out_valid(out_valid)
    );

    // --- 時脈產生器 ---
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz 時脈
    end

    // --- 測試流程 ---
    initial begin
        // 1. 初始化和重置
        rst_n <= 1'b0;
        WorI_tb <= 1'b0;
        mode <= 3'b000;
        tb_read_addr <= 0;
        tb_en <= 1'b0;
        tb_ram_write_en <= 1'b0;
        tb_ram_write_addr <= 0;
        tb_ram_write_data <= 0;
        #20;
        rst_n <= 1'b1;
        #200;

        // --- 階段 1: 權重預載入測試 ---
        $display("--- 階段 1: 權重預載入測試 (Weight Preload) ---");
        WorI_tb <= 1'b1; // 設定為權重模式
        tb_en <= 1'b1;   // 致能讀取
        
        // 從位址 800 開始讀取 25 個權重 (假設為 5x5 核心)
        for (i = 0; i < 25; i = i + 1) begin
            @(posedge clk);
            tb_read_addr <= 800 + i;
        end
        tb_read_addr <= 0;

        @(posedge clk);
        tb_en <= 1'b1; // 關閉讀取
        WorI_tb <= 1'b0;
        $display("--- 階段 1: 權重預載入完成 ---\n");
        
        #100;

        // --- 階段 2: 推論模式讀取測試 ---
        $display("--- 階段 2: 推論模式讀取測試 (Inference) ---");
        WorI_tb <= 1'b0; // 設定為推論模式
        tb_en <= 1'b1;   // 致能讀取
        
        // 從位址 0 開始讀取所有影像資料
        for (i = 0; i < RAM_DEPTH; i = i + 1) begin
            @(posedge clk);
            tb_read_addr <= i;
        end
        
        @(posedge clk);
        tb_en <= 1'b0; // 關閉讀取
        $display("--- 階段 2: 推論模式讀取完成 ---\n");
        #100;

    // feature map size = 24
        // 1. 初始化和重置
        rst_n <= 1'b0;
        WorI_tb <= 1'b0;
        mode <= 3'b001;
        tb_read_addr <= 0;
        tb_en <= 1'b0;
        tb_ram_write_en <= 1'b0;
        tb_ram_write_addr <= 0;
        tb_ram_write_data <= 0;
        #20;
        rst_n <= 1'b1;
        #200;

        // --- 階段 2: 推論模式讀取測試 ---
        $display("--- 階段 2: 推論模式讀取測試 (Inference) ---");
        WorI_tb <= 1'b0; // 設定為推論模式
        tb_en <= 1'b1;   // 致能讀取
        
        // 從位址 0 開始讀取所有影像資料
        for (i = 0; i < RAM_DEPTH; i = i + 1) begin
            @(posedge clk);
            tb_read_addr <= i;
        end
        
        @(posedge clk);
        tb_en <= 1'b0; // 關閉讀取
        $display("--- 階段 2: 推論模式讀取完成 ---\n");
        #100;

    // feature map size = 12
        // 1. 初始化和重置
        rst_n <= 1'b0;
        WorI_tb <= 1'b0;
        mode <= 3'b010;
        tb_read_addr <= 0;
        tb_en <= 1'b0;
        tb_ram_write_en <= 1'b0;
        tb_ram_write_addr <= 0;
        tb_ram_write_data <= 0;
        #20;
        rst_n <= 1'b1;
        #200;

        // --- 階段 2: 推論模式讀取測試 ---
        $display("--- 階段 2: 推論模式讀取測試 (Inference) ---");
        WorI_tb <= 1'b0; // 設定為推論模式
        tb_en <= 1'b1;   // 致能讀取
        
        // 從位址 0 開始讀取所有影像資料
        for (i = 0; i < RAM_DEPTH; i = i + 1) begin
            @(posedge clk);
            tb_read_addr <= i;
        end
        
        @(posedge clk);
        tb_en <= 1'b0; // 關閉讀取
        $display("--- 階段 2: 推論模式讀取完成 ---\n");
        #100;

    // feature map size = 8
        // 1. 初始化和重置
        rst_n <= 1'b0;
        WorI_tb <= 1'b0;
        mode <= 3'b011;
        tb_read_addr <= 0;
        tb_en <= 1'b0;
        tb_ram_write_en <= 1'b0;
        tb_ram_write_addr <= 0;
        tb_ram_write_data <= 0;
        #20;
        rst_n <= 1'b1;
        #200;

        // --- 階段 2: 推論模式讀取測試 ---
        $display("--- 階段 2: 推論模式讀取測試 (Inference) ---");
        WorI_tb <= 1'b0; // 設定為推論模式
        tb_en <= 1'b1;   // 致能讀取
        
        // 從位址 0 開始讀取所有影像資料
        for (i = 0; i < RAM_DEPTH; i = i + 1) begin
            @(posedge clk);
            tb_read_addr <= i;
        end
        
        @(posedge clk);
        tb_en <= 1'b0; // 關閉讀取
        $display("--- 階段 2: 推論模式讀取完成 ---\n");
        #100;

    // feature map size = 4
        // 1. 初始化和重置
        rst_n <= 1'b0;
        WorI_tb <= 1'b0;
        mode <= 3'b100;
        tb_read_addr <= 0;
        tb_en <= 1'b0;
        tb_ram_write_en <= 1'b0;
        tb_ram_write_addr <= 0;
        tb_ram_write_data <= 0;
        #20;
        rst_n <= 1'b1;
        #200;

        // --- 階段 2: 推論模式讀取測試 ---
        $display("--- 階段 2: 推論模式讀取測試 (Inference) ---");
        WorI_tb <= 1'b0; // 設定為推論模式
        tb_en <= 1'b1;   // 致能讀取
        
        // 從位址 0 開始讀取所有影像資料
        for (i = 0; i < RAM_DEPTH; i = i + 1) begin
            @(posedge clk);
            tb_read_addr <= i;
        end
        
        @(posedge clk);
        tb_en <= 1'b0; // 關閉讀取
        $display("--- 階段 2: 推論模式讀取完成 ---\n");
        #100;

        // --- 階段 3: 寫入測試 ---
        $display("--- 階段 3: 寫入測試 ---");
        tb_ram_write_en <= 1'b1;
        for (i = 0; i < 10; i = i + 1) begin // 寫入前10個地址
            @(posedge clk);
            tb_ram_write_addr <= i;
            tb_ram_write_data <= 8'hAA + i; // 寫入不同的數據
        end
        @(posedge clk);
        tb_ram_write_en <= 1'b0;
        $display("--- 階段 3: 寫入測試完成 ---\n");

        $display("--- Testbench 結束 ---");
        $finish;
    end

    // --- 監控輸出 ---
    initial begin
        // 更新 $monitor 以包含 WorI
        $monitor("時間=%0t | rst=%b | WorI=%b | en=%b | rd_addr=%d | wr_en=%b | out0=%d, weight_out=%d",
                $time, rst_n, WorI_tb, tb_en, read_addr_out, tb_ram_write_en, out0, weight_out_wire);
    end

endmodule