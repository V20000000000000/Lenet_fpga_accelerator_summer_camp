`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/09/23 02:34:21
// Design Name: 
// Module Name: top
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
//  - Top-level module for the CNN accelerator.
//  - Instantiates the controller and datapath modules.
// 
// Dependencies: 
//  - controller.v
//  - datapath.v
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module top #(
    // --- Parameters for easy configuration ---
    parameter DATA_WIDTH        = 8,
    parameter ADDR_WIDTH        = 11,
    parameter N                 = 5,
    parameter MAX_WIDTH         = 32,
    parameter PE_DATA_WIDTH     = 22,
    parameter PE_PORT_WIDTH     = 8
)(
    // --- Global Interface ---
    input wire                      clk,
    input wire                      rst_n,
    input wire                      start,          // Start the entire process
    
    // --- Interface for writing initial data/images to internal RAM ---
    // Note: These ports are for the user to write data into the datapath's RAM.
    // The datapath module provided is missing ram_write_data and ram_write_addr inputs,
    // so they are not connected here but are included for a complete top-level design.

    // --- Status and Result Interface ---
    output wire                     busy,           // Accelerator is busy
    output wire                     done,           // Accelerator has finished the whole sequence
    output wire [DATA_WIDTH-1:0]    result          // Final result from the datapath
);

    // =================================================================
    // Internal Wires to connect Controller and Datapath
    // =================================================================
    wire                            ctrl_write_en;
    wire [1:0]                      ctrl_mux_sel;
    wire                            ctrl_WorI;
    wire                            ctrl_ram_en;
    wire [$clog2(N*N)-1:0]          ctrl_weight_location;
    wire                            ctrl_addr_ctrl_en;
    wire [2:0]                      ctrl_mode;
    wire [ADDR_WIDTH-1:0]           ctrl_read_addr;


    // =================================================================
    // Instantiate the Controller
    // =================================================================
    controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N(N)
    ) ctrl_inst (
        // Global signals
        .clk(clk),
        .rst_n(rst_n),
        // Top-level interface
        .start(start),
        .busy(busy),
        .done(done),
        // Control signals (Outputs to Datapath)
        .ctrl_write_en(ctrl_write_en),
        .ctrl_mux_sel(ctrl_mux_sel),
        .ctrl_WorI(ctrl_WorI),
        .ctrl_ram_en(ctrl_ram_en),
        .ctrl_weight_location(ctrl_weight_location),
        .ctrl_addr_ctrl_en(ctrl_addr_ctrl_en),
        .ctrl_mode(ctrl_mode),
        .ctrl_read_addr(ctrl_read_addr)
    );

    // =================================================================
    // Instantiate the Datapath
    // =================================================================
    datapath #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .MAX_WIDTH(MAX_WIDTH),
        .PE_DATA_WIDTH(PE_DATA_WIDTH),
        .PE_PORT_WIDTH(PE_PORT_WIDTH),
        .N(N)
    ) dp_inst (
        // Global signals
        .clk(clk),
        .rst_n(rst_n),
        // Data inputs from top-level
        // .ram_write_addr(ram_write_addr), // Missing in provided datapath module
        // .ram_write_data(ram_write_data), // Missing in provided datapath module
        // Control signals (Inputs from Controller)
        .ctrl_ram_en(ctrl_ram_en),
        .ctrl_addr_ctrl_en(ctrl_addr_ctrl_en),
        .ctrl_WorI(ctrl_WorI),
        .ctrl_mode(ctrl_mode),
        .ctrl_read_addr(ctrl_read_addr),
        .ctrl_weight_location(ctrl_weight_location),
        .ctrl_mux_sel(ctrl_mux_sel),
        // Result output
        .result(result)
    );

endmodule
