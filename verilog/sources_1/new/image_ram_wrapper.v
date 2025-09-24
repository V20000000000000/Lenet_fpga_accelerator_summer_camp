`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/18 01:02:21
// Design Name: 
// Module Name: image_ram_wrapper
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


module image_ram_wrapper #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 11,
    // 不同模式下的特定特徵圖尺寸
    parameter FEATURE_MAP1_SIZE   = 32,
    parameter FEATURE_MAP2_SIZE   = 28,
    parameter FEATURE_MAP3_SIZE   = 14,
    parameter FEATURE_MAP4_SIZE   = 10,
    parameter FEATURE_MAP5_SIZE   = 5,
    parameter WAVEFRONT_DELAY = 4
)(
    // Port A (Read/Write Port)
    input wire                           clka,
    input wire                           ena,
    input wire                           wea, // wea[0:0] 實際上是 1-bit
    input wire      [ADDR_WIDTH-1:0]     addra,
    input wire      [DATA_WIDTH-1:0]     dina,

    // Port B (Read-Only Port)
    input wire                           clkb,
    input wire                           enb,
    input wire      [ADDR_WIDTH-1:0]     addrb,
    output wire     [DATA_WIDTH-1:0]     doutb
);

    // 在這裡例化 (instantiate) 您在 Vivado 中產生的 BRAM IP 核
    // *** 請將 "blk_mem_gen_0" 替換成您自己的 IP 實例名稱 ***
    blk_mem_gen_0 bram_ip_inst (
        .clka(clka),
        .ena(ena),
        .wea(wea), 
        .addra(addra),
        .dina(dina),
        .douta(), 
        
        .clkb(clkb),
        .enb(enb),
        .addrb(addrb),
        .doutb(doutb),
        .dinb(8'b0), 
        .web(1'b0) 
    );

endmodule
