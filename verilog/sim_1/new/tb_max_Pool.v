`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/20 14:52:41
// Design Name: 
// Module Name: tb_max_Pool
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


module tb_max_Pool;

    // --- 參數定義 ---
    localparam DATA_WIDTH = 8;
    localparam CLK_PERIOD = 10; // 10ns, 100MHz

    // --- Testbench 內部信號 ---
    reg                      rst_n;
    reg                      clk;
    reg [DATA_WIDTH-1:0]     line_1;
    reg [DATA_WIDTH-1:0]     line_2;

    wire [DATA_WIDTH-1:0]    max_out;

    // --- 測試輔助變數 ---
    integer i;
    integer error_count = 0;

    // --- "黃金模型": 用於計算期望值 ---
    // 這些暫存器模擬 DUT 內部的管線延遲
    reg [DATA_WIDTH-1:0] line_1_d1, line_1_d2; // line_1 延遲 1, 2 週期
    reg [DATA_WIDTH-1:0] line_2_d1, line_2_d2; // line_2 延遲 1, 2 週期
    reg [DATA_WIDTH-1:0] expected_max;

    // --- 例化待測模組 (DUT) ---
    max_Pool #(
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .rst_n(rst_n),
        .clk(clk),
        .line_1(line_1),
        .line_2(line_2),
        .max_out(max_out)
    );

    // --- 時脈產生器 ---
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2) clk = ~clk;
    end

    // --- 黃金模型: 模擬 DUT 的管線延遲 ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            line_1_d1 <= 0;
            line_1_d2 <= 0;
            line_2_d1 <= 0;
            line_2_d2 <= 0;
        end else begin
            line_1_d1 <= line_1;
            line_1_d2 <= line_1_d1;
            line_2_d1 <= line_2;
            line_2_d2 <= line_2_d1;
        end
    end

    // --- 黃金模型: 計算期望的最大值 ---
    always @(*) begin
        // 根據 DUT 的兩級管線延遲，
        // 當前週期的輸出 max_out 應該是 T-2 時刻 2x2 窗口的最大值。
        // 該窗口由 line_1_d2, line_1_d1, line_2_d2, line_2_d1 組成。
        expected_max = (line_1_d1 > line_1_d2) ? line_1_d1 : line_1_d2;
        expected_max = (expected_max > line_2_d1) ? expected_max : line_2_d1;
        expected_max = (expected_max > line_2_d2) ? expected_max : line_2_d2;
    end

    // --- 主測試流程 ---
    initial begin
        #200;
        $display("========================================");
        $display("T=%0t | Max Pool Testbench 啟動", $time);
        
        // 1. 初始化並施加重置
        line_1 <= 8'h00;
        line_2 <= 8'h00;
        rst_n  <= 1'b0; // 施加重置
        repeat(5) @(posedge clk);
        rst_n  <= 1'b1; // 釋放重置
        $display("T=%0t | 系統重置完成", $time);
        
        // --- 串流輸入測試資料並進行驗證 ---
        $display("\n--- 開始串流資料並進行驗證 ---");
        for (i = 0; i < 30; i = i + 1) begin
            @(posedge clk);
            // 產生兩組交錯的、可預測的資料流
            line_1 <= i + 10; // e.g., 10, 11, 12, ...
            line_2 <= 50 - i; // e.g., 50, 49, 48, ...

            // 等待 DUT 的管線填滿 (2 個週期) 後，才開始驗證
            if (i >= 2) begin
                if (max_out === expected_max) begin
                    $display("T=%0t | 驗證成功: 2x2_Window={%d,%d,%d,%d}, Max_Out=%d, Expected=%d", 
                            $time, line_1_d1, line_1_d2, line_2_d1, line_2_d2, max_out, expected_max);
            end else begin
                $display("T=%0t | ***** 驗證失敗 *****: 2x2_Window={%d,%d,%d,%d}, Max_Out=%d, Expected=%d", 
                            $time, line_1_d1, line_1_d2, line_2_d1, line_2_d2, max_out, expected_max);
                    error_count = error_count + 1;
                end
            end
        end

        // --- 測試結束 ---
        $display("\n========================================");
        if (error_count == 0) begin
            $display("T=%0t | 所有測試通過!", $time);
        end else begin
            $display("T=%0t | 測試完成，發現 %d 個錯誤!", $time, error_count);
        end
        $display("========================================");
        $finish;
    end

endmodule
