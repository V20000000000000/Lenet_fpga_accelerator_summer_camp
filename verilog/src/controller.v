`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/25 20:17:00
// Design Name: controller
// Module Name: controller
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
//  - Hierarchical FSM controller for CNN datapath.
//  - Durations for each sub-state are configurable.
// 
//////////////////////////////////////////////////////////////////////////////////

module controller #(
    // --- 繼承自 datapath 的參數 ---
    parameter DATA_WIDTH        = 8,
    parameter ADDR_WIDTH        = 11,
    parameter N                 = 5,

    // --- Controller 的配置參數 ---
    parameter WEIGHT_START_ADDR = 10'd800,
    parameter IMG_START_ADDR    = 10'd0,

    // --- 【修改】為每個狀態的「副狀態」定義持續時間 ---
    // CONV Layer
    parameter CONV_PRELOAD_DURATION   = 10'd40,  // Preload 階段的總時脈週期數
    parameter CONV_INFERENCE_DURATION = 10'd800, // Inference 階段的總時脈週期數
    parameter CONV_FINISH_DURATION    = 10'd30,  // Finish 階段 (pipeline flush) 的總時脈週期數

    // MAXPOOL Layer (Placeholder)
    parameter MAXPOOL_READ_DURATION     = 10'd100,
    parameter MAXPOOL_COMPUTE_DURATION  = 10'd50,

    // RELU Layer (Placeholder)
    parameter RELU_EXECUTE_DURATION   = 10'd50,

    // FC Layer (Placeholder)
    parameter FC_READ_DURATION      = 10'd150,
    parameter FC_COMPUTE_DURATION   = 10'd50
)(
    // --- Global Clock and Reset ---
    input wire                      clk,
    input wire                      rst_n,

    // --- Control Interface from Top Level ---
    input wire                      start,
    output wire                     busy,
    output wire                     done,

    // --- Control Signals to Datapath ---
    output reg                      ctrl_ram_en,
    output reg                      ctrl_WorI,
    output reg [2:0]                ctrl_mode,
    output reg [ADDR_WIDTH-1:0]     ctrl_read_addr,
    output reg [$clog2(N*N)-1:0]    ctrl_weight_location
);

    // --- 主狀態機定義 ---
    localparam [2:0] S_IDLE    = 3'b000, S_CONV = 3'b001, S_MAXPOOL = 3'b010, S_RELU = 3'b011, S_FC = 3'b100;
    reg [2:0] current_state, next_state;

    // --- 【新增】為每個主狀態定義各自的副狀態機 ---
    // CONV Sub-FSM
    localparam [1:0] CONV_IDLE = 2'b00, CONV_PRELOAD = 2'b01, CONV_INFERENCE = 2'b10, CONV_FINISH = 2'b11;
    reg [1:0] conv_state, next_conv_state;
    
    // MAXPOOL Sub-FSM
    localparam [1:0] MAXPOOL_IDLE = 2'b00, MAXPOOL_READ = 2'b01, MAXPOOL_COMPUTE = 2'b10;
    reg [1:0] maxpool_state, next_maxpool_state;

    // RELU Sub-FSM
    localparam [0:0] RELU_IDLE = 1'b0, RELU_EXECUTE = 1'b1;
    reg [0:0] relu_state, next_relu_state;

    // FC Sub-FSM
    localparam [1:0] FC_IDLE = 2'b00, FC_READ = 2'b01, FC_COMPUTE = 2'b10;
    reg [1:0] fc_state, next_fc_state;

    // --- 計數器 ---
    reg [$clog2(N*N)-1:0]       weight_counter; // For weight preloading address
    reg [ADDR_WIDTH-1:0]        addr_counter;   // For inference data reading address
    
    // --- 【新增】副狀態持續時間計數器 ---
    reg [ADDR_WIDTH-1:0]        sub_state_duration_counter;

    localparam TOTAL_WEIGHTS = N * N;

    // --- 狀態機暫存器 ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= S_IDLE;
            conv_state    <= CONV_IDLE;
            maxpool_state <= MAXPOOL_IDLE;
            relu_state    <= RELU_IDLE;
            fc_state      <= FC_IDLE;
        end else begin
            current_state <= next_state;
            conv_state    <= next_conv_state;
            maxpool_state <= next_maxpool_state;
            relu_state    <= next_relu_state;
            fc_state      <= next_fc_state;
        end
    end

    // --- 計數器邏輯 ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_counter <= 0;
            addr_counter <= 0;
            sub_state_duration_counter <= 0;
        end else begin
            // --- 【修改】副狀態計數器重置邏輯 ---
            // 當主狀態或對應的副狀態改變時，計數器歸零
            if (next_state != current_state ||
                (current_state == S_CONV    && next_conv_state != conv_state) ||
                (current_state == S_MAXPOOL && next_maxpool_state != maxpool_state) ||
                (current_state == S_RELU    && next_relu_state != relu_state) ||
                (current_state == S_FC      && next_fc_state != fc_state)) 
            begin
                sub_state_duration_counter <= 0;
            end 
            else if (current_state != S_IDLE) begin
                sub_state_duration_counter <= sub_state_duration_counter + 1;
            end

            // Weight and Addr counters now only care about incrementing within their specific sub-state
            // Weight counter logic
            if (current_state == S_CONV && conv_state == CONV_PRELOAD) begin
                // 即使 Preload DURATION > TOTAL_WEIGHTS，地址計數器也應在讀完後停止
                if (sub_state_duration_counter > 5 && sub_state_duration_counter < TOTAL_WEIGHTS + 5) begin
                    weight_counter <= weight_counter + 1;
                end
            end else begin
                weight_counter <= 0;
            end

            // Address counter for inference
            if (current_state == S_CONV && conv_state == CONV_INFERENCE) begin
                if (sub_state_duration_counter > 5 && sub_state_duration_counter < 784 + 5) begin
                    addr_counter <= addr_counter + 1;
                end
            end else begin
                addr_counter <= 0;
            end
        end
    end

    // --- 狀態轉移邏輯 ---
    always @(*) begin
        // Default assignments
        next_state         = current_state;
        next_conv_state    = conv_state;
        next_maxpool_state = maxpool_state;
        next_relu_state    = relu_state;
        next_fc_state      = fc_state;

        case (current_state)
            S_IDLE: begin
                if (start) begin
                    next_state = S_CONV;
                    next_conv_state = CONV_PRELOAD;
                end
            end
            S_CONV: begin
                case (conv_state)
                    CONV_PRELOAD: begin
                        if (sub_state_duration_counter == CONV_PRELOAD_DURATION - 1) begin
                            next_conv_state = CONV_INFERENCE;
                        end
                    end
                    CONV_INFERENCE: begin
                        if (sub_state_duration_counter == CONV_INFERENCE_DURATION - 1) begin
                            next_conv_state = CONV_FINISH;
                        end
                    end
                    CONV_FINISH: begin
                        if (sub_state_duration_counter == CONV_FINISH_DURATION - 1) begin
                            next_conv_state = CONV_IDLE;
                            next_state = S_MAXPOOL; // 進入下一個主狀態
                            next_maxpool_state = MAXPOOL_READ; // 設定下一個主狀態的初始副狀態
                        end
                    end
                    default: next_conv_state = CONV_IDLE;
                endcase
            end
            S_MAXPOOL: begin
                case(maxpool_state)
                    MAXPOOL_READ: begin
                        if (sub_state_duration_counter == MAXPOOL_READ_DURATION - 1) begin
                            next_maxpool_state = MAXPOOL_COMPUTE;
                        end
                    end
                    MAXPOOL_COMPUTE: begin
                        if (sub_state_duration_counter == MAXPOOL_COMPUTE_DURATION - 1) begin
                            next_maxpool_state = MAXPOOL_IDLE;
                            next_state = S_IDLE; // 假設 MAXPOOL 後結束
                        end
                    end
                    default: next_maxpool_state = MAXPOOL_IDLE;
                endcase
            end
            S_RELU: begin
                // ... similar logic for RELU sub-states
                next_state = S_IDLE;
            end
            S_FC: begin
                // ... similar logic for FC sub-states
                next_state = S_IDLE;
            end
        endcase
    end

    // --- 輸出邏輯 ---
    always @(*) begin
        // Default values
        ctrl_ram_en = 1'b0;
        ctrl_WorI = 1'b0;
        ctrl_mode = 3'b000;
        ctrl_read_addr = 0;
        ctrl_weight_location = 0;

        case (current_state)
            S_CONV: begin
                case (conv_state)
                    CONV_PRELOAD: begin
                        ctrl_ram_en = 1'b1;
                        ctrl_WorI = 1'b1;
                        ctrl_read_addr = WEIGHT_START_ADDR + weight_counter;
                        ctrl_weight_location = weight_counter;
                    end
                    CONV_INFERENCE: begin
                        ctrl_ram_en = 1'b1;
                        ctrl_WorI = 1'b0;
                        ctrl_read_addr = IMG_START_ADDR + addr_counter;
                    end
                    CONV_FINISH: begin
                        ctrl_ram_en = 1'b0;
                    end
                endcase
            end
            S_MAXPOOL: begin
                case(maxpool_state)
                    MAXPOOL_READ:    ctrl_ram_en = 1'b1; // 假設讀取時需要致能RAM
                    MAXPOOL_COMPUTE: ctrl_ram_en = 1'b0;
                endcase
            end
            // ... output logic for RELU, FC etc.
        endcase
    end

    // --- 狀態信號 ---
    assign busy = (current_state != S_IDLE);
    // 'done' is now asserted when the very last state of the whole process is finished.
    // For this example, we assume it's after MAXPOOL.
    assign done = (current_state == S_MAXPOOL && maxpool_state == MAXPOOL_COMPUTE && sub_state_duration_counter == MAXPOOL_COMPUTE_DURATION - 1);

endmodule