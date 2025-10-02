`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/10 03:02:48
// Design Name: 
// Module Name: tb_PE
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


`timescale 1ns / 1ps


module tb_PE;

    // -- 參數定義 --
    // 與待測模組 (DUT) 保持一致
    localparam DATA_WIDTH = 22;
    localparam PORT_WIDTH = 9;

    // -- 訊號宣告 --
    reg                           clk;
    reg                           rst_n;
    reg                           mode;
    reg  signed [PORT_WIDTH-1:0]  a;
    reg  signed [DATA_WIDTH-1:0]  b;
    reg  signed [PORT_WIDTH-1:0]  weight;

    wire signed [PORT_WIDTH-1:0]  a_out;
    wire signed [DATA_WIDTH-1:0]  b_out;

    // -- 例化待測模組 (DUT) --
    // **請確保這裡例化的是您最終使用的 PE 模組**
    // 例如 pe_module_ip 或 pe_module_pipelined
    PE #(
        .DATA_WIDTH(DATA_WIDTH),
        .PORT_WIDTH(PORT_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .mode(mode),
        .a(a),
        .b(b),
        .weight(weight),
        .a_out(a_out),
        .b_out(b_out)
    );

    // -- 1. 時脈產生器 --
    initial begin
        clk = 0;
        forever #20 clk = ~clk; // 產生一個 40ns 週期的時脈 (25MHz)
    end

    // -- 2. 測試腳本 --
    initial begin
        $display("========================================");
        $display("T=%0t | 測試平台啟動 (3級管線PE, 負緣觸發)", $time);
        
        // -- 情境 1: 系統重置 --
        $display("--- 情境 1: 系統重置 ---");
        a      <= 0;
        b      <= 0;
        weight <= 0;
        mode   <= 0;
        rst_n  <= 1'b0;
        #40;
        rst_n  <= 1'b1;
        @(posedge clk);
        $display("T=%0t | 重置已釋放. a_out=%d, b_out=%d", $time, a_out, b_out);

        // -- 情境 2: 權重預載 (mode = 1) --
        $display("\n--- 情境 2: 權重預載 ---");
        @(negedge clk); // **修改**: 在負緣更新
        mode   <= 1'b1;
        weight <= 5;
        @(posedge clk);
        $display("T=%0t | 模式=1, 權重=5 已載入. 輸出應為 0.", $time);
        
        // -- 情境 3: 推理運算 (mode = 0) --
        $display("\n--- 情境 3: 推理運算 ---");
        @(negedge clk); // **修改**: 在負緣更新
        mode <= 1'b0;
        weight <= 0;
        @(posedge clk); // 等待一個週期讓 mode 生效

        // --- Cycle 1: 輸入第一組數據 ---
        $display("\nCycle 1: 輸入向量 1 (a=10, b=100)");
        @(negedge clk); // **修改**: 在負緣更新
        a <= 10;
        b <= 100;
        @(posedge clk);
        $display("T=%0t | 輸出: a_out=%d, b_out=%d (來自重置)", $time, a_out, b_out);

        // --- Cycle 2: 輸入第二組數據 ---
        $display("\nCycle 2: 輸入向量 2 (a=20, b=-30)");
        @(negedge clk); // **修改**: 在負緣更新
        a <= 20;
        b <= -30;
        @(posedge clk);
        $display("T=%0t | 輸出: a_out=%d, b_out=%d (來自重置)", $time, a_out, b_out);

        // --- Cycle 3: 輸入第三組數據 ---
        $display("\nCycle 3: 輸入向量 3 (a=-5, b=15)");
        @(negedge clk); // **修改**: 在負緣更新
        a <= -5;
        b <= 15;
        @(posedge clk);
        $display("T=%0t | 輸出: a_out=%d, b_out=%d (來自重置)", $time, a_out, b_out);

        // --- Cycle 4: 觀察第一組數據的結果 ---
        $display("\nCycle 4: 應出現向量 1 的結果");
        @(negedge clk); // **修改**: 在負緣更新
        a <= 0; // 停止輸入
        b <= 0;
        @(posedge clk);
        // **結果出現**: 這是 Cycle 1 輸入的 a=10, b=100 的結果
        // 預期 a_out = 10, b_out = 10 * 5 + 100 = 150
        $display("T=%0t | 輸出: a_out=%d, b_out=%d (a=10, b=100 的結果)", $time, a_out, b_out);

        // --- Cycle 5: 觀察第二組數據的結果 ---
        $display("\nCycle 5: 應出現向量 2 的結果");
        @(posedge clk);
        // **結果出現**: 這是 Cycle 2 輸入的 a=20, b=-30 的結果
        // 預期 a_out = 20, b_out = 20 * 5 + (-30) = 70
        $display("T=%0t | 輸出: a_out=%d, b_out=%d (a=20, b=-30 的結果)", $time, a_out, b_out);
        
        // --- Cycle 6: 觀察第三組數據的結果 ---
        $display("\nCycle 6: 應出現向量 3 的結果");
        @(posedge clk);
        // **結果出現**: 這是 Cycle 3 輸入的 a=-5, b=15 的結果
        // 預期 a_out = -5, b_out = (-5) * 5 + 15 = -10
        $display("T=%0t | 輸出: a_out=%d, b_out=%d (a=-5, b=15 的結果)", $time, a_out, b_out);

        // -- 結束模擬 --
        #100;
        $display("\n========================================");
        $display("T=%0t | 測試平台結束.", $time);
        $finish;
    end
endmodule