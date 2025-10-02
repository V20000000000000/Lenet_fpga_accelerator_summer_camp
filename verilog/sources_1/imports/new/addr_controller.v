`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/31 12:05:12
// Design Name: 
// Module Name: addr_controller
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

module addr_controller#(
    // --- 頂層參數 ---
    parameter DATA_WIDTH            = 8,
    parameter ADDR_WIDTH            = 11,
    
    // --- 特徵圖尺寸參數 ---
    parameter FEATURE_MAP1_SIZE       = 32,
    parameter FEATURE_MAP2_SIZE       = 28,
    parameter FEATURE_MAP3_SIZE       = 14,
    parameter FEATURE_MAP4_SIZE       = 10,
    parameter FEATURE_MAP5_SIZE       = 5,
    parameter FEATURE_MAP1_START_ADDR = 173,
    parameter FEATURE_MAP2_START_ADDR = 157,
    parameter FEATURE_MAP3_START_ADDR = 101,
    parameter FEATURE_MAP4_START_ADDR = 85,
    parameter FEATURE_MAP5_START_ADDR = 60,
    parameter MAX_POOLING1_START_ADDR = 87,
    parameter MAX_POOLING2_START_ADDR = 33,
    parameter RELU_START_ADDR_1       = 3,
    parameter KERNEL_SIZE           = 5
)(
    input wire clk,
    input wire rst_n,
    input wire [2:0] ctrl_mode,
    input wire [ADDR_WIDTH-1:0] ctrl_read_addr,
    output reg [ADDR_WIDTH-1:0] ctrl_write_addr,
    input wire en,

    // --- 為了方便驗證而拉出的內部信號 ---
    output wire valid_out,
    output wire [2:0] state_out,
    output wire [ADDR_WIDTH-1:0] row_counter_out,
    output wire [ADDR_WIDTH-1:0] col_counter_out
);

    // --- FSM 狀態定義 ---
    localparam S_IDLE         = 2'b00;
    localparam S_VALID        = 2'b01;
    localparam S_INVALID_EDGE = 2'b10;
    localparam S_FINISH       = 2'b11;
    localparam S_FETCH_DATA   = 2'b01; // For Pooling FSM

    // --- 新增: 註冊輸入信號以改善時序 ---
    reg [ADDR_WIDTH-1:0] ctrl_read_addr_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_read_addr_reg <= 0;
        end else begin
            ctrl_read_addr_reg <= ctrl_read_addr;
        end
    end

    // --- 為 5 個 FSM 宣告各自的內部信號 ---
    // 當前狀態 (時序邏輯)
    reg [1:0] current_map1_state, current_map2_state, current_map3_state, current_map4_state, current_map5_state, current_map6_state, current_map7_state, current_map8_state;
    reg map1_valid_reg, map2_valid_reg, map3_valid_reg, map4_valid_reg, map5_valid_reg, map6_valid_reg, map7_valid_reg, map8_valid_reg;
    reg [ADDR_WIDTH-1:0] row_counter1, row_counter2, row_counter3, row_counter4, row_counter5, row_counter6, row_counter7, row_counter8;
    reg [ADDR_WIDTH-1:0] col_counter1, col_counter2, col_counter3, col_counter4, col_counter5, col_counter6, col_counter7, col_counter8;
    reg [ADDR_WIDTH-1:0] edge_counter1, edge_counter2, edge_counter3, edge_counter4, edge_counter5, edge_counter6, edge_counter7, edge_counter8;

    // 下一個狀態 (組合邏輯)
    reg [1:0] next_map1_state, next_map2_state, next_map3_state, next_map4_state, next_map5_state, next_map6_state, next_map7_state, next_map8_state;
    reg next_map1_valid, next_map2_valid, next_map3_valid, next_map4_valid, next_map5_valid, next_map6_valid, next_map7_valid, next_map8_valid;
    reg [ADDR_WIDTH-1:0] next_row_counter1, next_row_counter2, next_row_counter3, next_row_counter4, next_row_counter5, next_row_counter6, next_row_counter7, next_row_counter8;
    reg [ADDR_WIDTH-1:0] next_col_counter1, next_col_counter2, next_col_counter3, next_col_counter4, next_col_counter5, next_col_counter6, next_col_counter7, next_col_counter8;
    reg [ADDR_WIDTH-1:0] next_edge_counter1, next_edge_counter2, next_edge_counter3, next_edge_counter4, next_edge_counter5, next_edge_counter6, next_edge_counter7, next_edge_counter8;
    wire      mode_changed;

    // FSM for Map 1: 組合邏輯部分
    always @(*) begin
        // 預設下一狀態等於當前狀態
        next_map1_state    = current_map1_state;
        next_map1_valid    = map1_valid_reg;
        next_row_counter1  = row_counter1;
        next_col_counter1  = col_counter1;
        next_edge_counter1 = edge_counter1;

        case (current_map1_state)
            S_IDLE: begin
                next_map1_valid = 1'b0;
                if (ctrl_read_addr_reg >= FEATURE_MAP1_START_ADDR) begin
                    next_map1_state = S_VALID;
                end else begin
                    next_map1_state = S_IDLE;
                end
            end
            S_VALID: begin
                next_map1_valid = 1'b1;
                if (col_counter1 < (FEATURE_MAP1_SIZE - KERNEL_SIZE)) begin
                    next_col_counter1 = col_counter1 + 1;
                end else begin
                    next_map1_state    = S_INVALID_EDGE;
                    next_edge_counter1 = 0;
                    next_col_counter1  = 0;
                end
            end
            S_INVALID_EDGE: begin
                next_map1_valid = 1'b0;
                if (edge_counter1 < (KERNEL_SIZE - 2)) begin
                    next_edge_counter1 = edge_counter1 + 1;
                end else begin
                    next_row_counter1 = row_counter1 + 1;
                    if (row_counter1 < (FEATURE_MAP1_SIZE - KERNEL_SIZE)) begin
                        next_map1_state = S_VALID;
                    end else begin
                        next_map1_state = S_FINISH;
                    end
                end
            end
            S_FINISH: begin
                next_map1_valid = 1'b0;
            end
            default: begin
                next_map1_state = S_IDLE;
            end
        endcase
    end
    
    // FSM for Map 1: 時序邏輯部分
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_map1_state <= S_IDLE;
            map1_valid_reg     <= 1'b0;
            row_counter1       <= 0;
            col_counter1       <= 0;
            edge_counter1      <= 0;
        end else begin
            if (!en || ctrl_mode != 3'b000 || mode_changed) begin
                current_map1_state <= S_IDLE;
                map1_valid_reg     <= 1'b0;
                row_counter1       <= 0;
                col_counter1       <= 0;
                edge_counter1      <= 0;
            end else begin
                current_map1_state <= next_map1_state;
                map1_valid_reg     <= next_map1_valid;
                row_counter1       <= next_row_counter1;
                col_counter1       <= next_col_counter1;
                edge_counter1      <= next_edge_counter1;
            end
        end
    end

    // --- FSM for Map 2: Combinational Part ---
    always @(*) begin
        next_map2_state    = current_map2_state;
        next_map2_valid    = map2_valid_reg;
        next_row_counter2  = row_counter2;
        next_col_counter2  = col_counter2;
        next_edge_counter2 = edge_counter2;

        case (current_map2_state)
            S_IDLE: begin
                next_map2_valid = 1'b0;
                if (ctrl_read_addr_reg >= FEATURE_MAP2_START_ADDR) begin
                    next_map2_state = S_VALID;
                end else begin
                    next_map2_state = S_IDLE;
                end
            end
            S_VALID: begin
                next_map2_valid = 1'b1;
                if (col_counter2 < (FEATURE_MAP2_SIZE - KERNEL_SIZE))
                    next_col_counter2 = col_counter2 + 1;
                else begin
                    next_map2_state    = S_INVALID_EDGE;
                    next_edge_counter2 = 0;
                    next_col_counter2  = 0;
                end
            end
            S_INVALID_EDGE: begin
                next_map2_valid = 1'b0;
                if (edge_counter2 < (KERNEL_SIZE - 2))
                    next_edge_counter2 = edge_counter2 + 1;
                else begin
                    next_row_counter2 = row_counter2 + 1;
                    if (row_counter2 < (FEATURE_MAP2_SIZE - KERNEL_SIZE))
                        next_map2_state = S_VALID;
                    else
                        next_map2_state = S_FINISH;
                end
            end
            S_FINISH: begin
                next_map2_valid = 1'b0;
            end
            default: next_map2_state = S_IDLE;
        endcase
    end

    // --- FSM for Map 2: Sequential Part ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_map2_state <= S_IDLE;
            map2_valid_reg     <= 1'b0;
            row_counter2       <= 0;
            col_counter2       <= 0;
            edge_counter2      <= 0;
        end else begin
            if (!en || ctrl_mode != 3'b001 || mode_changed) begin
                current_map2_state <= S_IDLE;
                map2_valid_reg     <= 1'b0;
                row_counter2       <= 0;
                col_counter2       <= 0;
                edge_counter2      <= 0;
            end else begin
                current_map2_state <= next_map2_state;
                map2_valid_reg     <= next_map2_valid;
                row_counter2       <= next_row_counter2;
                col_counter2       <= next_col_counter2;
                edge_counter2      <= next_edge_counter2;
            end
        end
    end

    // --- FSM for Map 3: Combinational Part ---
    always @(*) begin
        next_map3_state    = current_map3_state;
        next_map3_valid    = map3_valid_reg;
        next_row_counter3  = row_counter3;
        next_col_counter3  = col_counter3;
        next_edge_counter3 = edge_counter3;

        case (current_map3_state)
            S_IDLE: begin
                next_map3_valid = 1'b0;
                if (ctrl_read_addr_reg >= FEATURE_MAP3_START_ADDR) begin
                    next_map3_state = S_VALID;
                end else begin
                    next_map3_state = S_IDLE;
                end
            end
            S_VALID: begin
                next_map3_valid = 1'b1;
                if (col_counter3 < (FEATURE_MAP3_SIZE - KERNEL_SIZE))
                    next_col_counter3 = col_counter3 + 1;
                else begin
                    next_map3_state    = S_INVALID_EDGE;
                    next_edge_counter3 = 0;
                    next_col_counter3  = 0;
                end
            end
            S_INVALID_EDGE: begin
                next_map3_valid = 1'b0;
                if (edge_counter3 < (KERNEL_SIZE - 2))
                    next_edge_counter3 = edge_counter3 + 1;
                else begin
                    next_row_counter3 = row_counter3 + 1;
                    if (row_counter3 < (FEATURE_MAP3_SIZE - KERNEL_SIZE))
                        next_map3_state = S_VALID;
                    else
                        next_map3_state = S_FINISH;
                end
            end
            S_FINISH: begin
                next_map3_valid = 1'b0;
            end
            default: next_map3_state = S_IDLE;
        endcase
    end

    // --- FSM for Map 3: Sequential Part ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_map3_state <= S_IDLE;
            map3_valid_reg     <= 1'b0;
            row_counter3       <= 0;
            col_counter3       <= 0;
            edge_counter3      <= 0;
        end else begin
            if (!en || ctrl_mode != 3'b010 || mode_changed) begin
                current_map3_state <= S_IDLE;
                map3_valid_reg     <= 1'b0;
                row_counter3       <= 0;
                col_counter3       <= 0;
                edge_counter3      <= 0;
            end else begin
                current_map3_state <= next_map3_state;
                map3_valid_reg     <= next_map3_valid;
                row_counter3       <= next_row_counter3;
                col_counter3       <= next_col_counter3;
                edge_counter3      <= next_edge_counter3;
            end
        end
    end

    // --- FSM for Map 4: Combinational Part ---
    always @(*) begin
        next_map4_state    = current_map4_state;
        next_map4_valid    = map4_valid_reg;
        next_row_counter4  = row_counter4;
        next_col_counter4  = col_counter4;
        next_edge_counter4 = edge_counter4;

        case (current_map4_state)
            S_IDLE: begin
                next_map4_valid = 1'b0;
                if (ctrl_read_addr_reg >= FEATURE_MAP4_START_ADDR) begin
                    next_map4_state = S_VALID;
                end else begin
                    next_map4_state = S_IDLE;
                end
            end
            S_VALID: begin
                next_map4_valid = 1'b1;
                if (col_counter4 < (FEATURE_MAP4_SIZE - KERNEL_SIZE))
                    next_col_counter4 = col_counter4 + 1;
                else begin
                    next_map4_state    = S_INVALID_EDGE;
                    next_edge_counter4 = 0;
                    next_col_counter4  = 0;
                end
            end
            S_INVALID_EDGE: begin
                next_map4_valid = 1'b0;
                if (edge_counter4 < (KERNEL_SIZE - 2))
                    next_edge_counter4 = edge_counter4 + 1;
                else begin
                    next_row_counter4 = row_counter4 + 1;
                    if (row_counter4 < (FEATURE_MAP4_SIZE - KERNEL_SIZE))
                        next_map4_state = S_VALID;
                    else
                        next_map4_state = S_FINISH;
                end
            end
            S_FINISH: begin
                next_map4_valid = 1'b0;
            end
            default: next_map4_state = S_IDLE;
        endcase
    end

    // --- FSM for Map 4: Sequential Part ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_map4_state <= S_IDLE;
            map4_valid_reg     <= 1'b0;
            row_counter4       <= 0;
            col_counter4       <= 0;
            edge_counter4      <= 0;
        end else begin
            if (!en || ctrl_mode != 3'b011 || mode_changed) begin
                current_map4_state <= S_IDLE;
                map4_valid_reg     <= 1'b0;
                row_counter4       <= 0;
                col_counter4       <= 0;
                edge_counter4      <= 0;
            end else begin
                current_map4_state <= next_map4_state;
                map4_valid_reg     <= next_map4_valid;
                row_counter4       <= next_row_counter4;
                col_counter4       <= next_col_counter4;
                edge_counter4      <= next_edge_counter4;
            end
        end
    end
    
    // --- FSM for Map 5: Combinational Part ---
    always @(*) begin
        next_map5_state    = current_map5_state;
        next_map5_valid    = map5_valid_reg;
        next_row_counter5  = row_counter5;
        next_col_counter5  = col_counter5;
        next_edge_counter5 = edge_counter5;

        case (current_map5_state)
            S_IDLE: begin
                next_map5_valid = 1'b0;
                if (ctrl_read_addr_reg >= FEATURE_MAP5_START_ADDR) begin
                    next_map5_state = S_VALID;
                end else begin
                    next_map5_state = S_IDLE;
                end
            end
            S_VALID: begin
                next_map5_valid = 1'b1;
                if (col_counter5 < (FEATURE_MAP5_SIZE - KERNEL_SIZE))
                    next_col_counter5 = col_counter5 + 1;
                else begin
                    next_map5_state    = S_INVALID_EDGE;
                    next_edge_counter5 = 0;
                    next_col_counter5  = 0;
                end
            end
            S_INVALID_EDGE: begin
                next_map5_valid = 1'b0;
                if (edge_counter5 < (KERNEL_SIZE - 2))
                    next_edge_counter5 = edge_counter5 + 1;
                else begin
                    next_row_counter5 = row_counter5 + 1;
                    if (row_counter5 < (FEATURE_MAP5_SIZE - KERNEL_SIZE))
                        next_map5_state = S_VALID;
                    else
                        next_map5_state = S_FINISH;
                end
            end
            S_FINISH: begin
                next_map5_valid = 1'b0;
            end
            default: next_map5_state = S_IDLE;
        endcase
    end

    // --- FSM for Map 5: Sequential Part ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_map5_state <= S_IDLE;
            map5_valid_reg     <= 1'b0;
            row_counter5       <= 0;
            col_counter5       <= 0;
            edge_counter5      <= 0;
        end else begin
            if (!en || ctrl_mode != 3'b100 || mode_changed) begin
                current_map5_state <= S_IDLE;
                map5_valid_reg     <= 1'b0;
                row_counter5       <= 0;
                col_counter5       <= 0;
                edge_counter5      <= 0;
            end else begin
                current_map5_state <= next_map5_state;
                map5_valid_reg     <= next_map5_valid;
                row_counter5       <= next_row_counter5;
                col_counter5       <= next_col_counter5;
                edge_counter5      <= next_edge_counter5;
            end
        end
    end

    

    // --- FSM for Map 6 (Pooling 1, 14x14 out): Combinational Part ---
    always @(*) begin
        // 預設下一狀態等於當前狀態
        next_map6_state   = current_map6_state;
        next_map6_valid   = 1'b0; // 預設 valid 為 0
        next_row_counter6 = row_counter6;
        next_col_counter6 = col_counter6;

        case (current_map6_state)
            S_IDLE: begin
                // 只有當 read address 達到起始值時才轉換狀態
                if ((~mode_changed) && ctrl_read_addr_reg >= MAX_POOLING1_START_ADDR) begin
                    next_map6_state = S_FETCH_DATA;
                end else begin
                    next_map6_state = S_IDLE;
                end
            end
            S_FETCH_DATA: begin
                // 2x2 Pooling with stride 2
                // 只有在 2x2 窗口的右下角 (row, col 都是奇數) 才輸出 valid
                if (row_counter6[0] == 1'b1 && col_counter6[0] == 1'b1) begin
                    next_map6_valid = 1'b1;
                end
                
                // 遍歷輸入的 Feature Map (Map 2: 28x28)
                if (col_counter6 < (FEATURE_MAP2_SIZE - 1)) begin
                    next_col_counter6 = col_counter6 + 1;
                end else begin 
                    next_col_counter6 = 0;
                    if (row_counter6 < (FEATURE_MAP2_SIZE - 1)) begin
                        next_row_counter6 = row_counter6 + 1;
                    end else begin
                        // 遍歷完成，進入 FINISH 狀態
                        next_map6_state = S_FINISH;
                    end
                end
            end
            S_FINISH: begin
                next_map6_valid = 1'b0;
                // 停在 FINISH 狀態，直到模式切換被重置
                next_map6_state = S_FINISH;
            end
            default: begin
                next_map6_state = S_IDLE;
            end
        endcase
    end

    // --- FSM for Map 6 (Pooling 1): Sequential Part ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_map6_state <= S_IDLE;
            map6_valid_reg     <= 1'b0;
            row_counter6       <= 0;
            col_counter6       <= 0;
        end else begin
            if (!en || ctrl_mode != 3'b101 || mode_changed) begin // Mode for Pooling 1
                current_map6_state <= S_IDLE;
                map6_valid_reg     <= 1'b0;
                row_counter6       <= 0;
                col_counter6       <= 0;
            end else begin
                current_map6_state <= next_map6_state;
                map6_valid_reg     <= next_map6_valid;
                row_counter6       <= next_row_counter6;
                col_counter6       <= next_col_counter6;
            end
        end
    end

    // --- FSM for Map 7 (Pooling 2, 5x5 out): Combinational Part ---
    always @(*) begin
        // 預設下一狀態等於當前狀態
        next_map7_state   = current_map7_state;
        next_map7_valid   = 1'b0; // 預設 valid 為 0
        next_row_counter7 = row_counter7;
        next_col_counter7 = col_counter7;

        case (current_map7_state)
            S_IDLE: begin
                // 只有當 read address 達到起始值時才轉換狀態
                if ((~mode_changed) && ctrl_read_addr_reg >= MAX_POOLING2_START_ADDR) begin
                    next_map7_state = S_FETCH_DATA;
                end else begin
                    next_map7_state = S_IDLE;
                end
            end
            S_FETCH_DATA: begin
                // 2x2 Pooling with stride 2
                // 只有在 2x2 窗口的右下角 (row, col 都是奇數) 才輸出 valid
                if (row_counter7[0] == 1'b1 && col_counter7[0] == 1'b1) begin
                    next_map7_valid = 1'b1;
                end
                
                // 遍歷輸入的 Feature Map (Map 4: 10x10)
                if (col_counter7 < (FEATURE_MAP4_SIZE - 1)) begin
                    next_col_counter7 = col_counter7 + 1;
                end else begin 
                    next_col_counter7 = 0;
                    if (row_counter7 < (FEATURE_MAP4_SIZE - 1)) begin
                        next_row_counter7 = row_counter7 + 1;
                    end else begin 
                        // 遍歷完成，進入 FINISH 狀態
                        next_map7_state = S_FINISH;
                    end
                end
            end
            S_FINISH: begin
                next_map7_valid = 1'b0;
                // 停在 FINISH 狀態，直到模式切換被重置
                next_map7_state = S_FINISH;
            end
            default: begin
                next_map7_state = S_IDLE;
            end
        endcase
    end

    // --- FSM for Map 7 (Pooling 2): Sequential Part ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_map7_state <= S_IDLE;
            map7_valid_reg     <= 1'b0;
            row_counter7       <= 0;
            col_counter7       <= 0;
        end else begin
            if (!en || ctrl_mode != 3'b110 || mode_changed) begin // Mode for Pooling 2
                current_map7_state <= S_IDLE;
                map7_valid_reg     <= 1'b0;
                row_counter7       <= 0;
                col_counter7       <= 0;
            end else begin
                current_map7_state <= next_map7_state;
                map7_valid_reg     <= next_map7_valid;
                row_counter7       <= next_row_counter7;
                col_counter7       <= next_col_counter7;
            end
        end
    end

    // --- FSM for RELU_1 (14*14in, 14*14out): Combinational Part ---
    always @(*) begin
        if(ctrl_mode == 3'b111 && ctrl_read_addr_reg >= RELU_START_ADDR_1 && ctrl_read_addr_reg < RELU_START_ADDR_1 + FEATURE_MAP3_SIZE * FEATURE_MAP3_SIZE) begin
            map8_valid_reg = 1'b1;
        end else begin
            map8_valid_reg = 1'b0;
        end
    end

    // --- 輸出選擇邏輯 (Output MUX) ---
    reg temp_valid;
    reg [2:0] temp_state;
    reg [ADDR_WIDTH-1:0] temp_row_counter;
    reg [ADDR_WIDTH-1:0] temp_col_counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            temp_valid <= 1'b0;
            temp_state <= S_IDLE;
            temp_row_counter <= 0;
            temp_col_counter <= 0;
        end else begin
                case (ctrl_mode)
                    3'b000: begin temp_valid <= map1_valid_reg; temp_state <= current_map1_state; temp_row_counter <= row_counter1; temp_col_counter <= col_counter1; end
                    3'b001: begin temp_valid <= map2_valid_reg; temp_state <= current_map2_state; temp_row_counter <= row_counter2; temp_col_counter <= col_counter2; end
                    3'b010: begin temp_valid <= map3_valid_reg; temp_state <= current_map3_state; temp_row_counter <= row_counter3; temp_col_counter <= col_counter3; end
                    3'b011: begin temp_valid <= map4_valid_reg; temp_state <= current_map4_state; temp_row_counter <= row_counter4; temp_col_counter <= col_counter4; end
                    3'b100: begin temp_valid <= map5_valid_reg; temp_state <= current_map5_state; temp_row_counter <= row_counter5; temp_col_counter <= col_counter5; end
                    3'b101: begin temp_valid <= map6_valid_reg; temp_state <= current_map6_state; temp_row_counter <= row_counter6; temp_col_counter <= col_counter6; end
                    3'b110: begin temp_valid <= map7_valid_reg; temp_state <= current_map7_state; temp_row_counter <= row_counter7; temp_col_counter <= col_counter7; end
                    3'b111: begin temp_valid <= map8_valid_reg; temp_state <= current_map8_state; temp_row_counter <= row_counter8; temp_col_counter <= col_counter8; end
                default: begin temp_valid <= 1'b0; temp_state <= S_IDLE; temp_row_counter <= 0; temp_col_counter <= 0; end
            endcase
        end
    end
    
    assign valid_out = temp_valid;
    assign state_out = temp_state;
    assign row_counter_out = temp_row_counter;
    assign col_counter_out = temp_col_counter;

    // --- 寫入位址產生邏輯 ---
    reg [2:0] ctrl_mode_prev;
    reg [2:0] ctrl_mode_prev2;
    

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_mode_prev  <= 3'b0;
            ctrl_mode_prev2 <= 3'b0;
        end else begin
            ctrl_mode_prev  <= ctrl_mode;
            ctrl_mode_prev2 <= ctrl_mode_prev;
        end
    end

    assign mode_changed = (ctrl_mode_prev != ctrl_mode_prev2);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_write_addr <= 0; 
        end else begin
            if (!en) begin
                ctrl_write_addr <= ctrl_write_addr;
            end else if (mode_changed) begin
                ctrl_write_addr <= 0;
            end else if (valid_out) begin
                ctrl_write_addr <= ctrl_write_addr + 1;
            end
        end
    end

endmodule
