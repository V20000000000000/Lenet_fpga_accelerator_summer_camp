# ===================================================================
# 1. Main Clock Definition (50MHz)
# ===================================================================
# 註：週期 20ns 對應的頻率是 50MHz。
# 必須與你的設計目標和 Testbench 模擬時脈匹配。
create_clock -period 20.000 -name sys_clk [get_ports clk]

# ===================================================================
# 2. Asynchronous Path Constraint
# ===================================================================
# 假設 'rst_n' 是一個異步的重置信號 (例如來自按鈕)，與時脈無關。
# 使用 set_false_path 告訴工具不要對其進行時序分析，避免不必要的時序違規。
set_false_path -from [get_ports rst_n]

# ===================================================================
# 3. Input Delay Constraints
# ===================================================================
# 告訴工具，外部元件的數據在 "clk" 的上升沿之前多久就準備好了。
# 公式: Input Delay = T_ext_clk_to_q + T_ext_pcb_delay

# --- 控制信號 'start' ---
# 假設 'start' 信號在外部時脈觸發後，最多需要 2.0ns 的時間送達 FPGA。
set_input_delay -clock [get_clocks sys_clk] -max 2.0 [get_ports start]
# 假設最小延遲為 0.0ns
set_input_delay -clock [get_clocks sys_clk] -min 0.0 [get_ports start]


# ===================================================================
# 4. Output Delay Constraints
# ===================================================================
# 告訴工具，FPGA 的輸出信號在 "clk" 的上升沿之後，需要預留多少時間給外部元件去接收。
# 公式: Output Delay = T_ext_setup + T_ext_pcb_delay

# --- 狀態信號 'busy' 和 'done' ---
# 假設外部的狀態監控晶片需要 1.8ns 的 setup time + PCB delay。
set_output_delay -clock [get_clocks sys_clk] 1.8 [get_ports {busy done}]

# --- 結果數據 'result' ---
# 假設外部的接收晶片 (例如 RAM 或另一顆 FPGA) 需要 2.0ns 的 setup time + PCB delay。
# 數據匯流排的約束通常比較嚴格。
set_output_delay -clock [get_clocks sys_clk] 2.0 [get_ports result]