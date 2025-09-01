`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/19 01:08:19
// Design Name: 
// Module Name: tb_line_buffer
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


module tb_line_buffer;

    // --- 參數定義 ---
    localparam DATA_WIDTH = 8;
    localparam MAX_WIDTH  = 28;
    localparam CLK_PERIOD = 10; // 10ns, 100MHz

    // --- Testbench 內部信號 ---
    reg                        clk;
    reg                        rst_n;
    reg [2:0]                  mode;
    reg [DATA_WIDTH-1:0]       data_in;

    wire [DATA_WIDTH-1:0]      line_out_0;
    wire [DATA_WIDTH-1:0]      line_out_1;
    wire [DATA_WIDTH-1:0]      line_out_2;
    wire [DATA_WIDTH-1:0]      line_out_3;
    wire [DATA_WIDTH-1:0]      line_out_4;

    // --- 測試輔助變數 ---
    integer pixel_count = 0;
    integer error_count = 0;
    integer line_width;

    // "黃金模型": 儲存輸入資料的歷史紀錄，用於比對
    reg [DATA_WIDTH-1:0] history_buffer [0:MAX_WIDTH*6-1];

    // --- 例化待測模組 (DUT) ---
    line_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .MAX_WIDTH(MAX_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .mode(mode),
        .data_in(data_in),
        .line_out_0(line_out_0),
        .line_out_1(line_out_1),
        .line_out_2(line_out_2),
        .line_out_3(line_out_3),
        .line_out_4(line_out_4)
    );

    // --- 時脈產生器 ---
    initial begin
        clk = 1;
        forever #(CLK_PERIOD / 2) clk = ~clk;
    end

    // --- 主測試流程 ---
    initial begin
        $display("===================================================");
        $display("T=%0t | Line Buffer Testbench 啟動", $time);
        #200
        // 1. 初始化並施加重置
        mode    <= 3'b001; // 從 mode 0 (寬度 28) 開始測試
        data_in <= 8'h00;
        rst_n   <= 1'b0; // 施加重置
        repeat(5) @(posedge clk);
        rst_n   <= 1'b1; // 釋放重置
        $display("T=%0t | 系統重置完成", $time);
        
        @(posedge clk);

        // --- 場景 1: 測試 Mode 0 (寬度 28) ---
        $display("\n--- 場景 1: 測試 Mode 0 (寬度 = 28) ---");
        mode <= 3'b000;
        line_width = 28;
        
        // 串流輸入 6 行像素資料
        for (pixel_count = 0; pixel_count < line_width * 6; pixel_count = pixel_count + 1) begin
            @(posedge clk);
            data_in <= pixel_count + 1; // 送入從 1 開始遞增的資料
            history_buffer[pixel_count] <= pixel_count + 1;

            // 等待 4 行資料填滿 Line Buffer 後，開始驗證
            if (pixel_count >= line_width * 4) begin
                // 驗證 line_out_4 (應該等於當前的 data_in)
                if (line_out_4 !== data_in) begin
                    $display("T=%0t | ***** 錯誤 (line_out_4) *****: 像素 %d, 輸出 = %h, 預期 = %h", $time, pixel_count, line_out_4, data_in);
                    error_count = error_count + 1;
                end

                // 驗證 line_out_3 (應該是 1 行前的資料)
                if (line_out_3 !== history_buffer[pixel_count - line_width]) begin
                    $display("T=%0t | ***** 錯誤 (line_out_3) *****: 像素 %d, 輸出 = %h, 預期 = %h", $time, pixel_count, line_out_3, history_buffer[pixel_count - line_width]);
                    error_count = error_count + 1;
                end

                // 驗證 line_out_2 (應該是 2 行前的資料)
                if (line_out_2 !== history_buffer[pixel_count - 2*line_width]) begin
                    $display("T=%0t | ***** 錯誤 (line_out_2) *****: 像素 %d, 輸出 = %h, 預期 = %h", $time, pixel_count, line_out_2, history_buffer[pixel_count - 2*line_width]);
                    error_count = error_count + 1;
                end

                // 驗證 line_out_1 (應該是 3 行前的資料)
                if (line_out_1 !== history_buffer[pixel_count - 3*line_width]) begin
                    $display("T=%0t | ***** 錯誤 (line_out_1) *****: 像素 %d, 輸出 = %h, 預期 = %h", $time, pixel_count, line_out_1, history_buffer[pixel_count - 3*line_width]);
                    error_count = error_count + 1;
                end

                // 驗證 line_out_0 (應該是 4 行前的資料)
                if (line_out_0 !== history_buffer[pixel_count - 4*line_width]) begin
                    $display("T=%0t | ***** 錯誤 (line_out_0) *****: 像素 %d, 輸出 = %h, 預期 = %h", $time, pixel_count, line_out_0, history_buffer[pixel_count - 4*line_width]);
                    error_count = error_count + 1;
                end
            end
        end
        $display("T=%0t | Mode 0 測試資料串流完畢", $time);

        // --- 測試結束 ---
        $display("\n===================================================");
        if (error_count == 0) begin
            $display("T=%0t | 所有測試通過!", $time);
        end else begin
            $display("T=%0t | 測試完成，發現 %d 個錯誤!", $time, error_count);
        end
        $display("===================================================");
        $finish;
    end

endmodule

