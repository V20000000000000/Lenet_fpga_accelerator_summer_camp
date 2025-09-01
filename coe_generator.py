# BRAM 參數
MEM_DEPTH = 2048
DATA_WIDTH = 8
PATTERN_BASE = 32 # 規律的基礎週期
ZERO_FILL_START_ADDR = PATTERN_BASE * PATTERN_BASE # 從這個位址開始補 0 (1024)

# 新增權重位址範圍
WEIGHT_START_ADDR = 1200
WEIGHT_END_ADDR = 1224 # 包含 1224

OUTPUT_FILE = "image.coe"

data_vector = []

# 產生資料
for i in range(MEM_DEPTH):
    
    ## --- 核心邏輯修改 --- ##
    # 優先判斷是否在權重範圍內
    if i >= WEIGHT_START_ADDR and i <= WEIGHT_END_ADDR:
        value = 1
    # 接著判斷是否在圖樣範圍內
    elif i < ZERO_FILL_START_ADDR:
        # 在 576 之前的位址，使用原本的測試規律
        value = (i % PATTERN_BASE) + (i // PATTERN_BASE)
    # 最後，其餘的位址全部補 0
    else:
        value = 0
    
    # 將值格式化為 2 位數的十六進位字串
    hex_val = format(value, '02X')
    data_vector.append(hex_val)

# 寫入 COE 檔案
try:
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"; Test pattern for an {DATA_WIDTH}-bit x {MEM_DEPTH} memory\n")
        f.write(f"; Addr 0-{ZERO_FILL_START_ADDR-1}: Pattern, Addr {WEIGHT_START_ADDR}-{WEIGHT_END_ADDR}: 1, Others: Zero-filled\n")
        f.write("MEMORY_INITIALIZATION_RADIX = 16;\n")
        f.write("MEMORY_INITIALIZATION_VECTOR =\n")
        
        # 為了可讀性，每 24 個值換一行
        for i in range(0, len(data_vector), PATTERN_BASE):
            line_data = data_vector[i:i+PATTERN_BASE]
            f.write(", ".join(line_data))
            # 判斷是否為最後一行，如果不是就加上逗號和換行
            if i + PATTERN_BASE < len(data_vector):
                f.write(",\n")
        
        f.write(";\n")
        
    print(f"檔案 '{OUTPUT_FILE}' 已成功產生。")
except Exception as e:
    print(f"產生檔案時發生錯誤: {e}")