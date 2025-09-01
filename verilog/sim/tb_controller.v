`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/25 19:50:35
// Design Name: 
// Module Name: tb_controller
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


module tb_controller;

    // --- 參數定義 (與 controller 模組一致) ---
    parameter DATA_WIDTH        = 8;
    parameter ADDR_WIDTH        = 10;
    parameter N                 = 5;
    parameter WEIGHT_START_ADDR = 10'd800;
    parameter IMG_START_ADDR    = 10'd0;
    parameter IMG_DATA_COUNT    = 10'd784;

    // --- Testbench 內部信號 ---
    // Inputs to DUT
    reg                      clk;
    reg                      rst_n;
    reg                      start;

    // Outputs from DUT
    wire                     busy;
    wire                     done;
    wire                     ctrl_ram_en;
    wire                     ctrl_WorI;
    wire [2:0]               ctrl_mode;
    wire [ADDR_WIDTH-1:0]    ctrl_read_addr;
    wire [$clog2(N*N)-1:0]   ctrl_weight_location;

    // --- 實例化待測模組 (DUT: Controller) ---
    controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N(N),
        .WEIGHT_START_ADDR(WEIGHT_START_ADDR),
        .IMG_START_ADDR(IMG_START_ADDR),
        .IMG_DATA_COUNT(IMG_DATA_COUNT)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .busy(busy),
        .done(done),
        .ctrl_ram_en(ctrl_ram_en),
        .ctrl_WorI(ctrl_WorI),
        .ctrl_mode(ctrl_mode),
        .ctrl_read_addr(ctrl_read_addr),
        .ctrl_weight_location(ctrl_weight_location)
    );
    
    // --- 讓內部 FSM 狀態可被 testbench 觀察 (可選) ---
    // 為了在 $monitor 中顯示狀態，可以使用 Verilog 的 cross-module reference
    // 注意：這在某些 simulator 中可能需要特定 flag
    wire [2:0] current_state = uut.current_state;
    wire [1:0] conv_state = uut.conv_state;

    // --- 時脈產生器 (100MHz) ---
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 5ns high, 5ns low -> 10ns period
    end

    // --- 測試流程 ---
    initial begin
        // 1. 初始化和重置
        $display("--- [%0t] Testbench Started: Initializing and Resetting ---", $time);
        rst_n <= 1'b0;
        start <= 1'b0;
        #20; // 維持重置 20ns
        rst_n <= 1'b1;
        $display("--- [%0t] Reset Released. Controller is in IDLE state. ---", $time);
        #10;

        // 2. 啟動 Controller
        $display("--- [%0t] Asserting 'start' for one cycle. ---", $time);
        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        
        // 3. 等待運算完成
        $display("--- [%0t] Controller started. Waiting for 'done' signal... ---", $time);
        wait (done == 1'b1);
        
        $display("--- [%0t] 'done' signal received. Controller has finished the task. ---", $time);

        // 4. 結束模擬
        #100; // 等待一小段時間觀察結束後的狀態
        $display("--- [%0t] Testbench Finished. ---", $time);
        $finish;
    end

    // --- 監控輸出 ---
    initial begin
        // (可選) 產生 VCD 波形文件
        $dumpfile("tb_controller.vcd");
        $dumpvars(0, tb_controller);
    end

    always @(posedge clk) begin
        if (rst_n) begin
            $monitor("Time=%0t | State: %d | ConvState: %d | busy=%b | done=%b | WorI=%b | ram_en=%b | rd_addr=%d | weight_loc=%d",
                    $time, current_state, conv_state, busy, done, ctrl_WorI, ctrl_ram_en, ctrl_read_addr, ctrl_weight_location);
        end
    end

endmodule
