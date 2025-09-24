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
    parameter N                 = 5
)(
    // --- Global Clock and Reset ---
    input wire                      clk,
    input wire                      rst_n,

    // --- Control Interface from Top Level ---
    input wire                      start,
    output wire                     busy,
    output wire                     done,

    // --- Control Signals to Datapath ---
    output reg                      ctrl_write_en,
    output reg [1:0]                ctrl_mux_sel,
    output reg                      ctrl_WorI,
    output reg                      ctrl_ram_en,
    output reg [$clog2(N*N)-1:0]    ctrl_weight_location,
    output reg                      ctrl_addr_ctrl_en,
    output reg [2:0]                ctrl_mode,
    output reg [ADDR_WIDTH-1:0]     ctrl_read_addr
);
    // --- FSM States ---
    localparam IDLE         = 3'd0; 
    localparam PRELOAD      = 3'd1; 
    localparam INFERENCE    = 3'd2; 
    localparam MAXPOOL      = 3'd3; 
    localparam RELU         = 3'd4; 
    localparam S_DONE       = 3'd5; 

    // --- Cycle Counts for each state ---
    localparam WEIGHT_START_ADDR    = 11'd1200;
    localparam INFERENCE_CYCLES     = 12'd1200;
    localparam MAXPOOL_CYCLES       = 12'd1200;
    localparam RELU_CYCLES          = 12'd300;
    localparam WEIGHT_COUNT         = N*N;
    localparam WEIGHT_PRELOAD_CYCLES_START = 3; // Preload starts after 3 cycles in PRELOAD state
    localparam MEM_READ_DELAY     = 2; // Memory read delay cycles
    localparam START_DELAY        = 5; // Start signal delay cycles
    reg [2:0]  state, next_state;
    reg [ADDR_WIDTH + 2:0] counter; // 14-bit counter

    // =================================================================
    // FSM Next-State Logic (Combinational)
    // This logic correctly implements the required PRELOAD -> ... -> RELU flow.
    // =================================================================
    always @(*) begin
        next_state = state; // Default to current state
        case(state)
            IDLE: begin
                if (start) begin
                    next_state = PRELOAD;
                end
            end
            PRELOAD: begin
                if (counter == WEIGHT_PRELOAD_CYCLES_START + WEIGHT_COUNT - 1) begin
                    next_state = INFERENCE;
                end
            end
            INFERENCE: begin
                if (counter == INFERENCE_CYCLES - 1) begin
                    next_state = MAXPOOL;
                end
            end
            MAXPOOL: begin
                if (counter == MAXPOOL_CYCLES - 1) begin
                    next_state = RELU;
                end
            end
            RELU: begin
                if (counter == RELU_CYCLES - 1) begin
                    next_state = S_DONE;
                end
            end
            S_DONE: begin
                next_state = IDLE;
            end
            default: begin
                next_state = IDLE;
            end
        endcase
    end

    // =================================================================
    // FSM State & Counter Update Logic (Sequential)
    // =================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            counter <= 0;
        end else begin
            state <= next_state;
            if (next_state != state) begin 
                counter <= 0;
            end else if (state != IDLE && state != S_DONE) begin 
                counter <= counter + 1;
            end else begin
                counter <= 0; 
            end
        end
    end

    // =================================================================
    // FSM Output Logic for Control Signals (Combinational)
    // REVISED to prevent latches by setting default values.
    // =================================================================
    always @(*) begin
        // --- Default output values (safe/idle state) ---
        ctrl_write_en        = 1'b0;
        ctrl_mux_sel         = 2'b00;
        ctrl_WorI            = 1'b0; // Default to Inference mode
        ctrl_ram_en          = 1'b0;
        ctrl_weight_location = 'd0;
        ctrl_addr_ctrl_en    = 1'b0;
        ctrl_mode            = 3'b000;
        ctrl_read_addr       = 'd0;

        // --- Assign outputs based on the current state ---
        case(state)
            PRELOAD: begin 
                if(counter >= WEIGHT_PRELOAD_CYCLES_START && counter < WEIGHT_PRELOAD_CYCLES_START + WEIGHT_COUNT) begin
                    ctrl_WorI = 1'b1; // Weight preload mode
                end else begin
                    ctrl_WorI = 1'b0;
                end
                ctrl_ram_en          = 1'b1; 
                ctrl_read_addr       = WEIGHT_START_ADDR + counter;
                ctrl_weight_location = counter;
            end
            INFERENCE: begin 
                if(counter >= START_DELAY && counter < START_DELAY + INFERENCE_CYCLES) begin
                    ctrl_addr_ctrl_en    = 1'b1; 
                end else begin
                    ctrl_addr_ctrl_en    = 1'b0;
                end
                ctrl_read_addr = counter;
                ctrl_ram_en          = 1'b1;
                ctrl_mode            = 3'b000;
                ctrl_mux_sel         = 2'b00;
            end
            MAXPOOL: begin
                ctrl_read_addr = counter;
                ctrl_ram_en          = 1'b1;
                ctrl_addr_ctrl_en    = 1'b1; 
                ctrl_mode            = 3'b101;
                ctrl_mux_sel         = 2'b01;
            end
            RELU: begin 
                ctrl_ram_en          = 1'b1;
                // ctrl_addr_ctrl_en is 0 by default
                ctrl_mode            = 3'b111;
                ctrl_mux_sel         = 2'b10;
                ctrl_read_addr       = counter;
            end
            // IDLE and S_DONE states use the default values
            default: begin
                // All control signals remain at their default safe values
                ctrl_write_en        = 1'b0;
            end
        endcase
    end

    assign busy = (state != IDLE) && (state != S_DONE);
    assign done = (state == S_DONE);

endmodule