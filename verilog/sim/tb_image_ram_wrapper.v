`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/18 01:04:14
// Design Name: 
// Module Name: tb_image_ram_wrapper
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


module tb_image_ram_wrapper;

    // -- 參數定義 --
    localparam DATA_WIDTH = 8;
    localparam ADDR_WIDTH = 10;
    localparam MEM_DEPTH  = 1 << ADDR_WIDTH; // 1024
    localparam CLK_PERIOD = 10; // 10ns, 100MHz

    // -- Testbench 內部信號 --
    reg                            clk;
    
    // Port A 驅動信號
    reg                            ena;
    reg                            wea;
    reg        [ADDR_WIDTH-1:0]    addra;
    reg        [DATA_WIDTH-1:0]    dina;

    // Port B 驅動信號
    reg                            enb;
    reg        [ADDR_WIDTH-1:0]    addrb;
    
    // Port B 接收信號
    wire       [DATA_WIDTH-1:0]    doutb;
    
    // 其他測試用變數
    integer i;
    integer error_count = 0;
    reg [DATA_WIDTH-1:0] expected_data;

    // -- 例化待測模組 (DUT: Device Under Test) --
    image_ram_wrapper #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        // Port A
        .clka(clk),
        .ena(ena),
        .wea(wea),
        .addra(addra),
        .dina(dina),

        // Port B
        .clkb(clk), // 在此測試中，我們將兩個時脈連到同一個來源
        .enb(enb),
        .addrb(addrb),
        .doutb(doutb)
    );

    // -- 時脈產生器 --
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2) clk = ~clk;
    end

    // -- 主測試流程 --
    initial begin
        $display("========================================");
        $display("T=%0t | BRAM Testbench (先讀後寫) 啟動", $time);
        
        // 1. 初始化所有輸入信號
        ena   <= 1'b0;
        wea   <= 1'b0;
        addra <= 'h0;
        dina  <= 'h0;
        enb   <= 1'b0;
        addrb <= 'h0;

        repeat(200) @(posedge clk); // 等待幾個週期讓系統穩定
        
        // --- 場景 1: 讀取並驗證 BRAM 的初始內容 (.coe 檔案) ---
        $display("\n--- 場景 1: 讀取並驗證初始內容 (0 到 1023) ---");
        enb <= 1'b1; // 致能 Port B 進行讀取

        for (i = 0; i < MEM_DEPTH; i = i + 1) begin
            @(posedge clk);
            addrb <= i; // 提供要讀取的位址
            
            // 計算期望值 (根據 coe 檔案的規律)
            if (i < 784) begin
                expected_data = (i % 28) + (i / 28);
            end else begin
                expected_data = 8'h00;
            end
            
            // BRAM 的讀取有 1 個時脈週期的延遲 (latency)
            // 所以期望值會在下一個週期與 doutb 比較
            @(posedge clk);
            
            // 驗證讀出的資料
            if (doutb === expected_data) begin
                // 為了避免洗版，只在特定位址或發現錯誤時顯示訊息
                if (i < 10 || i % 100 == 0) begin
                    $display("T=%0t | 初始值驗證成功: Addr = %d, Data = %h", $time, addrb, doutb);
                end
            end else begin
                $display("T=%0t | ***** 初始值驗證失敗 *****: Addr = %d, 讀出 = %h (預期 = %h)", $time, addrb, doutb, expected_data);
                error_count = error_count + 1;
            end
        end

        for (i = 0; i < MEM_DEPTH; i = i + 1) begin
            @(posedge clk);
            addrb <= i; // 提供要讀取的位址
            
            // 計算期望值 (根據 coe 檔案的規律)
            if (i < 784) begin
                expected_data = (i % 28) + (i / 28);
            end else begin
                expected_data = 8'h00;
            end
            
            // BRAM 的讀取有 1 個時脈週期的延遲 (latency)
            // 所以期望值會在下一個週期與 doutb 比較
            @(posedge clk);
            
            // 驗證讀出的資料
            if (doutb === expected_data) begin
                // 為了避免洗版，只在特定位址或發現錯誤時顯示訊息
                if (i < 10 || i % 100 == 0) begin
                    $display("T=%0t | 初始值驗證成功: Addr = %d, Data = %h", $time, addrb, doutb);
                end
            end else begin
                $display("T=%0t | ***** 初始值驗證失敗 *****: Addr = %d, 讀出 = %h (預期 = %h)", $time, addrb, doutb, expected_data);
                error_count = error_count + 1;
            end
        end

        @(posedge clk);
        enb <= 1'b0; // 讀取完成後，取消 Port B 的致能
        $display("T=%0t | 初始內容驗證完成", $time);
        
        repeat(2) @(posedge clk);

        // --- 場景 2: 透過 Port A 寫入新資料 ---
        $display("\n--- 場景 2: 寫入新資料 ---");
        ena <= 1'b1; // 致能 Port A
        wea <= 1'b1; // 設為寫入模式

        // 寫入位址 0
        @(posedge clk);
        addra <= 0; dina <= 8'hAA;
        $display("T=%0t | 寫入: Addr = %d, Data = %h", $time, addra, dina);
        
        // 寫入位址 1
        @(posedge clk);
        addra <= 1; dina <= 8'hBB;
        $display("T=%0t | 寫入: Addr = %d, Data = %h", $time, addra, dina);

        // 寫入位址 784 (原本是 0 的區域)
        @(posedge clk);
        addra <= 784; dina <= 8'hCC;
        $display("T=%0t | 寫入: Addr = %d, Data = %h", $time, addra, dina);

        // 寫入位址 1023 (最後一個位址)
        @(posedge clk);
        addra <= 1023; dina <= 8'hDD;
        $display("T=%0t | 寫入: Addr = %d, Data = %h", $time, addra, dina);

        @(posedge clk);
        ena <= 1'b0; wea <= 1'b0; // 寫入完成後取消致能
        $display("T=%0t | 新資料寫入完成", $time);

        repeat(2) @(posedge clk);

        // --- 場景 3: 驗證寫入的結果 ---
        $display("\n--- 場景 3: 驗證寫入結果 ---");
        enb <= 1'b1; // 致能 Port B
        
        // 驗證位址 0
        @(posedge clk); addrb <= 0; @(posedge clk);
        if (doutb === 8'hAA) $display("T=%0t | 寫入驗證成功: Addr = %d, Data = %h", $time, addrb, doutb);
        else begin $display("T=%0t | ***** 寫入驗證失敗 *****: Addr = %d, 讀出 = %h (預期 = %h)", $time, addrb, doutb, 8'hAA); error_count = error_count + 1; end
        
        // 驗證位址 1
        @(posedge clk); addrb <= 1; @(posedge clk);
        if (doutb === 8'hBB) $display("T=%0t | 寫入驗證成功: Addr = %d, Data = %h", $time, addrb, doutb);
        else begin $display("T=%0t | ***** 寫入驗證失敗 *****: Addr = %d, 讀出 = %h (預期 = %h)", $time, addrb, doutb, 8'hBB); error_count = error_count + 1; end

        // 驗證位址 784
        @(posedge clk); addrb <= 784; @(posedge clk);
        if (doutb === 8'hCC) $display("T=%0t | 寫入驗證成功: Addr = %d, Data = %h", $time, addrb, doutb);
        else begin $display("T=%0t | ***** 寫入驗證失敗 *****: Addr = %d, 讀出 = %h (預期 = %h)", $time, addrb, doutb, 8'hCC); error_count = error_count + 1; end

        // 驗證位址 1023
        @(posedge clk); addrb <= 1023; @(posedge clk);
        if (doutb === 8'hDD) $display("T=%0t | 寫入驗證成功: Addr = %d, Data = %h", $time, addrb, doutb);
        else begin $display("T=%0t | ***** 寫入驗證失敗 *****: Addr = %d, 讀出 = %h (預期 = %h)", $time, addrb, doutb, 8'hDD); error_count = error_count + 1; end
        
        @(posedge clk);
        enb <= 1'b0;

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
