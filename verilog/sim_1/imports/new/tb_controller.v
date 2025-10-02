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
    parameter DATA_WIDTH = 8;
    parameter ADDR_WIDTH = 11;
    parameter N          = 5;

    // --- Testbench 內部信號 ---
    reg  clk;
    reg  rst_n;
    reg  start;

    wire busy;
    wire done;

    wire ctrl_write_en; // 從 controller 增加的 output
    wire ctrl_ram_en;
    wire ctrl_addr_ctrl_en;
    wire ctrl_WorI;
    wire [2:0] ctrl_mode;
    wire [ADDR_WIDTH-1:0] ctrl_read_addr;
    wire [$clog2(N*N)-1:0] ctrl_weight_location;
    wire [1:0] ctrl_mux_sel;

    // --- 實例化待測模組 (DUT: Controller) ---
    controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N(N)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .busy(busy),
        .done(done),
        .ctrl_write_en(ctrl_write_en),
        .ctrl_ram_en(ctrl_ram_en),
        .ctrl_addr_ctrl_en(ctrl_addr_ctrl_en),
        .ctrl_WorI(ctrl_WorI),
        .ctrl_mode(ctrl_mode),
        .ctrl_read_addr(ctrl_read_addr),
        .ctrl_weight_location(ctrl_weight_location),
        .ctrl_mux_sel(ctrl_mux_sel)
    );

    // --- 時脈產生器 ---
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 產生 100MHz 的時脈 (週期 10ns)
    end

    // --- 測試流程 ---
    initial begin
        // 1. 初始化和重置
        // 使用阻塞賦值 (=) 來避免 t=0 時的 race condition
        $display("--- [%0t] 測試開始: 初始化與重置 ---", $time);
        rst_n = 1'b0;
        start = 1'b0;
        #20; // 維持重置狀態 20ns (2 個 clock cycle)
        rst_n = 1'b1;
        $display("--- [%0t] 重置結束 ---", $time);
        
        // 2. 等待一個時脈週期後，發送 start 脈衝
        @(posedge clk);
        $display("--- [%0t] 發送 Start 脈衝 ---", $time);
        start = 1'b1;
        
        @(posedge clk);
        start = 1'b0;

        // 3. 等待 PRELOAD 狀態並驗證
        wait (dut.state == dut.PRELOAD);
        $display("--- [%0t] 進入 PRELOAD 狀態 ---", $time);
        if (dut.ctrl_WorI !== 1'b1) $error("PRELOAD 狀態錯誤: ctrl_WorI 應為 1");
        
        // 4. 等待 INFERENCE 狀態並驗證
        wait (dut.state == dut.INFERENCE);
        $display("--- [%0t] 進入 INFERENCE 狀態 ---", $time);
        if (dut.ctrl_mode !== 3'b000) $error("INFERENCE 狀態錯誤: ctrl_mode 應為 000");

        // 5. 等待 MAXPOOL 狀態並驗證
        wait (dut.state == dut.MAXPOOL);
        $display("--- [%0t] 進入 MAXPOOL 狀態 ---", $time);
        if (dut.ctrl_mode !== 3'b101) $error("MAXPOOL 狀態錯誤: ctrl_mode 應為 101");
        
        // 6. 等待 RELU 狀態並驗證
        wait (dut.state == dut.RELU);
        $display("--- [%0t] 進入 RELU 狀態 ---", $time);
        if (dut.ctrl_mux_sel !== 2'b10) $error("RELU 狀態錯誤: ctrl_mux_sel 應為 10");

        // 7. 等待 done 信號變為高電位
        wait (done);
        $display("--- [%0t] Controller 完成運作 (done = 1) ---", $time);

        // 8. 驗證 FSM 是否回到 IDLE
        wait (dut.state == dut.IDLE);
        $display("--- [%0t] Controller 回到 IDLE 狀態 ---", $time);
        
        #100;

        // 9. 結束測試
        $display("--- [%0t] Testbench 所有測試結束 ---", $time);
        $finish;
    end

    // --- 監控信號變化 ---
    initial begin
        // 使用 dut.state 可以更清晰地看到當前狀態名稱
        $monitor("Time=%0t | state=%3s | busy=%b, done=%b | WorI=%b, mode=%3b, mux=%2b | rd_addr=%4d",
                 $time, dut.state, busy, done, ctrl_WorI, ctrl_mode, ctrl_mux_sel, ctrl_read_addr);
    end

endmodule