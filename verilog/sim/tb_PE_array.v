`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/10 19:20:44
// Design Name: 
// Module Name: tb_PE_array
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
//////////////////////////////////////////////////////////////////////////////////
// 模組: tb_PE_array
// 描述: 一個為 5x5 PE_array 設計的測試平台，
//       提供 5 組獨立的、以對角線波前方式連續輸入的測試數據。
//////////////////////////////////////////////////////////////////////////////////
module tb_PE_array;

    // -- 與待測模組匹配的參數 --
    localparam DATA_WIDTH = 22;
    localparam PORT_WIDTH = 8;
    localparam N = 5;
    localparam STAGGER_DELAY = 4; // 定義交錯延遲為 4 週期

    // -- 用於驅動待測模組輸入的 Testbench 暫存器 --
    reg                         clk;
    reg                         rst_n;
    reg                         mode;

    // DUT 中定義的獨立輸入埠
    reg signed [PORT_WIDTH-1:0] a0_in, a1_in, a2_in, a3_in, a4_in;
    reg signed [PORT_WIDTH-1:0] b0_in, b1_in, b2_in, b3_in, b4_in;
    
    // *** 修改: 移除舊的 w_inputs 陣列 ***
    // reg signed [PORT_WIDTH-1:0] w_inputs [0:N*N-1];

    // *** 新增: 用於循序載入權重的信號 ***
    reg signed [PORT_WIDTH-1:0] weight_in;
    // 注意: 寬度需與 PE_array 的 weight_location 埠完全匹配
    reg [$clog2(N*N)-1:0] weight_location;


    // -- 用於接收待測模組輸出的 Testbench 線網 --
    wire signed [DATA_WIDTH-1:0] y0_out, y1_out, y2_out, y3_out, y4_out;
    wire signed [DATA_WIDTH+3:0] acc_out;
    // 為了匹配您的 DUT，我假設 s0-s3 輸出仍然存在
    wire signed [DATA_WIDTH:0] s0;
    wire signed [DATA_WIDTH+1:0] s1;
    wire signed [DATA_WIDTH+2:0] s2;
    wire signed [DATA_WIDTH+3:0] s3;


    // -- 用於產生激勵訊號的 Testbench 專用變數 --
    integer i, j, k;
    
    // 為 5 組測試數據宣告矩陣陣列
    reg signed [PORT_WIDTH-1:0] test_A [0:N-1][0:N-1][0:N-1]; // 5x5x5 matrix for A
    reg signed [PORT_WIDTH-1:0] test_B [0:N-1][0:N-1][0:N-1]; // 5x5x5 matrix for B
    wire signed [PORT_WIDTH-1:0] a00_out, a01_out, a02_out, a03_out, a04_out;
    wire signed [PORT_WIDTH-1:0] a10_out, a11_out, a12_out, a13_out, a14_out;
    wire signed [PORT_WIDTH-1:0] a20_out, a21_out, a22_out, a23_out, a24_out;
    wire signed [PORT_WIDTH-1:0] a30_out, a31_out, a32_out, a33_out, a34_out;
    wire signed [PORT_WIDTH-1:0] a40_out, a41_out, a42_out, a43_out, a44_out;
    wire signed [DATA_WIDTH-1:0] b00_out, b01_out, b02_out, b03_out, b04_out;
    wire signed [DATA_WIDTH-1:0] b10_out, b11_out, b12_out, b13_out, b14_out;
    wire signed [DATA_WIDTH-1:0] b20_out, b21_out, b22_out, b23_out, b24_out;
    wire signed [DATA_WIDTH-1:0] b30_out, b31_out, b32_out, b33_out, b34_out;
    wire signed [DATA_WIDTH-1:0] b40_out, b41_out, b42_out, b43_out, b44_out;
    wire signed [PORT_WIDTH-1:0] w00_out, w01_out, w02_out, w03_out, w04_out;
    wire signed [PORT_WIDTH-1:0] w10_out, w11_out, w12_out, w13_out, w14_out;
    wire signed [PORT_WIDTH-1:0] w20_out, w21_out, w22_out, w23_out, w24_out;
    wire signed [PORT_WIDTH-1:0] w30_out, w31_out, w32_out, w33_out, w34_out;
    wire signed [PORT_WIDTH-1:0] w40_out, w41_out, w42_out, w43_out, w44_out;

    // -- 例化待測模組 (DUT) --
    PE_array #(
        .DATA_WIDTH(DATA_WIDTH),
        .PORT_WIDTH(PORT_WIDTH),
        .N(N)
        // 假設您的 PE_array 有 PE_LATENCY 參數
        // .PE_LATENCY(STAGGER_DELAY) 
    ) dut (
        .clk(clk), .rst_n(rst_n), .WorI(mode),
        .a0_in(a0_in), .a1_in(a1_in), .a2_in(a2_in), .a3_in(a3_in), .a4_in(a4_in),
        .b0_in(b0_in), .b1_in(b1_in), .b2_in(b2_in), .b3_in(b3_in), .b4_in(b4_in),
        
        // *** 修改: 連接新的權重埠 ***
        .weight_in(weight_in),
        .weight_location(weight_location),

        .y0_out(y0_out), .y1_out(y1_out), .y2_out(y2_out), .y3_out(y3_out), .y4_out(y4_out), 
        .acc_out(acc_out),
        .a00_out(a00_out), .a01_out(a01_out), .a02_out(a02_out), .a03_out(a03_out), .a04_out(a04_out),
        .a10_out(a10_out), .a11_out(a11_out), .a12_out(a12_out), .a13_out(a13_out), .a14_out(a14_out),
        .a20_out(a20_out), .a21_out(a21_out), .a22_out(a22_out), .a23_out(a23_out), .a24_out(a24_out),
        .a30_out(a30_out), .a31_out(a31_out), .a32_out(a32_out), .a33_out(a33_out), .a34_out(a34_out),
        .a40_out(a40_out), .a41_out(a41_out), .a42_out(a42_out), .a43_out(a43_out), .a44_out(a44_out),
        .b00_out(b00_out), .b01_out(b01_out), .b02_out(b02_out), .b03_out(b03_out), .b04_out(b04_out),
        .b10_out(b10_out), .b11_out(b11_out), .b12_out(b12_out), .b13_out(b13_out), .b14_out(b14_out),
        .b20_out(b20_out), .b21_out(b21_out), .b22_out(b22_out), .b23_out(b23_out), .b24_out(b24_out),
        .b30_out(b30_out), .b31_out(b31_out), .b32_out(b32_out), .b33_out(b33_out), .b34_out(b34_out),
        .b40_out(b40_out), .b41_out(b41_out), .b42_out(b42_out), .b43_out(b43_out), .b44_out(b44_out),
        .w00_out(w00_out), .w01_out(w01_out), .w02_out(w02_out), .w03_out(w03_out), .w04_out(w04_out),
        .w10_out(w10_out), .w11_out(w11_out), .w12_out(w12_out), .w13_out(w13_out), .w14_out(w14_out),
        .w20_out(w20_out), .w21_out(w21_out), .w22_out(w22_out), .w23_out(w23_out), .w24_out(w24_out),
        .w30_out(w30_out), .w31_out(w31_out), .w32_out(w32_out), .w33_out(w33_out), .w34_out(w34_out),
        .w40_out(w40_out), .w41_out(w41_out), .w42_out(w42_out), .w43_out(w43_out), .w44_out(w44_out)
    );

    // 1. 時脈產生
    initial begin
        clk = 0;
        forever #20 clk = ~clk; // 40ns 週期 (25MHz)
    end

    // 2. 主測試流程
    initial begin
        $display("========================================");
        $display("T=%0t | 測試平台啟動 (交錯式輸入)", $time);

        // 初始化 5 組更多樣化的測試數據
        for (i = 0; i < N; i = i + 1) begin
            for (j = 0; j < N; j = j + 1) begin
                test_A[0][i][j] = i + j; test_B[0][i][j] = i - j;
                test_A[1][i][j] = j * 2; test_B[1][i][j] = i * 3;
                test_A[2][i][j] = -(i+j); test_B[2][i][j] = -(j*2);
                test_A[3][i][j] = 127 - i - j; test_B[3][i][j] = -128 + i + j;
                test_A[4][i][j] = j; test_B[4][i][j] = i;
            end
        end

        // --- 場景 1: 系統重置 ---
        rst_n <= 1'b0;
        mode  <= 0;
        {a4_in, a3_in, a2_in, a1_in, a0_in} <= 0;
        {b4_in, b3_in, b2_in, b1_in, b0_in} <= 0;
        weight_in <= 0;
        weight_location <= 0;
        #200;
        rst_n <= 1'b1;
        @(posedge clk);
        $display("T=%0t | 重置已釋放", $time);

        // --- 場景 2: 權重預載 (循序載入版本) ---
        $display("\n--- 場景 2: 權重循序預載 ---");
        mode <= 1'b1; // 進入預載入模式
        
        // 使用 for 迴圈，花費 25 個週期循序載入權重
        for (k = 0; k < N*N; k = k + 1) begin
            @(negedge clk);
            weight_location <= k;
            weight_in       <= 1; // 將所有權重都設為 1
            $display("T=%0t | 正在載入權重到位置 %d", $time, k);
        end
        #20;
        @(negedge clk);
        mode <= 1'b0; // 切換到推斷模式
        weight_in <= 0;
        weight_location <= 0;
        $display("T=%0t | 模式=0, 權重載入完畢", $time);
        
        // --- 場景 3: 交錯式推斷 (保持不變) ---
        $display("\n--- 場景 3: 交錯式推斷 ---");

        for (k = 0; k < (N-1)*STAGGER_DELAY + 5*N + 2*N; k = k + 1) begin
            @(negedge clk);

            // --- 驅動 a_in[i] ---
            a0_in <= (k >= 0*STAGGER_DELAY && k < 0*STAGGER_DELAY + 5*N) ? test_A[(k-0*STAGGER_DELAY)/N][0][(k-0*STAGGER_DELAY)%N] : 0;
            a1_in <= (k >= 1*STAGGER_DELAY && k < 1*STAGGER_DELAY + 5*N) ? test_A[(k-1*STAGGER_DELAY)/N][1][(k-1*STAGGER_DELAY)%N] : 0;
            a2_in <= (k >= 2*STAGGER_DELAY && k < 2*STAGGER_DELAY + 5*N) ? test_A[(k-2*STAGGER_DELAY)/N][2][(k-2*STAGGER_DELAY)%N] : 0;
            a3_in <= (k >= 3*STAGGER_DELAY && k < 3*STAGGER_DELAY + 5*N) ? test_A[(k-3*STAGGER_DELAY)/N][3][(k-3*STAGGER_DELAY)%N] : 0;
            a4_in <= (k >= 4*STAGGER_DELAY && k < 4*STAGGER_DELAY + 5*N) ? test_A[(k-4*STAGGER_DELAY)/N][4][(k-4*STAGGER_DELAY)%N] : 0;

            // --- 驅動 b_in[j] ---
            b0_in <= (k >= 0*STAGGER_DELAY && k < 0*STAGGER_DELAY + 5*N) ? test_B[(k-0*STAGGER_DELAY)/N][(k-0*STAGGER_DELAY)%N][0] : 0;
            b1_in <= (k >= 1*STAGGER_DELAY && k < 1*STAGGER_DELAY + 5*N) ? test_B[(k-1*STAGGER_DELAY)/N][(k-1*STAGGER_DELAY)%N][1] : 0;
            b2_in <= (k >= 2*STAGGER_DELAY && k < 2*STAGGER_DELAY + 5*N) ? test_B[(k-2*STAGGER_DELAY)/N][(k-2*STAGGER_DELAY)%N][2] : 0;
            b3_in <= (k >= 3*STAGGER_DELAY && k < 3*STAGGER_DELAY + 5*N) ? test_B[(k-3*STAGGER_DELAY)/N][(k-3*STAGGER_DELAY)%N][3] : 0;
            b4_in <= (k >= 4*STAGGER_DELAY && k < 4*STAGGER_DELAY + 5*N) ? test_B[(k-4*STAGGER_DELAY)/N][(k-4*STAGGER_DELAY)%N][4] : 0;
            
            @(posedge clk);
            #1;
            $display("T=%0t | k=%d | a_in={%d,%d,%d,%d,%d} | b_in={%d,%d,%d,%d,%d} | y_out={%d,%d,%d,%d,%d} | acc_out=%d",
                    $time, k, a4_in, a3_in, a2_in, a1_in, a0_in,
                    b4_in, b3_in, b2_in, b1_in, b0_in,
                    y4_out, y3_out, y2_out, y1_out, y0_out, acc_out);
        end

        // --- 結束模擬 ---
        $display("\n========================================");
        $display("T=%0t | 測試平台結束", $time);
        $finish;
    end

endmodule
