import numpy as np

def read_matrices_from_file(filepath):
    """
    從指定的 txt 檔案中讀取 5x5 的輸入矩陣和 5x5 的濾波器矩陣。

    檔案格式應為：
    - 前 5 行為輸入矩陣
    - 後 5 行為濾波器矩陣
    - 數字間由空格分隔
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # 過濾掉空白行
        lines = [line.strip() for line in lines if line.strip()]

        if len(lines) != 10:
            raise ValueError(f"錯誤：檔案需要包含 10 行數字，但實際找到 {len(lines)} 行。")

        # 讀取輸入矩陣
        input_data = [list(map(float, line.split())) for line in lines[:5]]
        # 讀取濾波器矩陣
        filter_data = [list(map(float, line.split())) for line in lines[5:]]

        input_matrix = np.array(input_data)
        filter_matrix = np.array(filter_data)

        if input_matrix.shape != (5, 5) or filter_matrix.shape != (5, 5):
            raise ValueError(f"錯誤：輸入與濾波器矩陣的維度都必須是 5x5。")

        return input_matrix, filter_matrix

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{filepath}'。請確認檔案路徑是否正確。")
        return None, None
    except Exception as e:
        print(f"讀取檔案時發生錯誤：{e}")
        return None, None

def perform_convolution(input_matrix, kernel):
    """
    執行 2D 卷積操作。
    
    注意：在機器學習領域，這個操作通常稱為「互相關」(cross-correlation)，
    因為濾波器(kernel)沒有經過翻轉。真正的數學卷積會先將濾波器翻轉180度。
    此處我們實作的是機器學習中常見的版本。

    Args:
        input_matrix (np.array): 輸入的 2D 矩陣。
        kernel (np.array): 濾波器/卷積核。

    Returns:
        np.array: 卷積後的結果矩陣。
    """
    # 取得輸入和濾波器的維度
    input_h, input_w = input_matrix.shape
    kernel_h, kernel_w = kernel.shape

    # --- 計算不同類型的卷積 ---

    # 1. 'valid' 卷積: 不使用填充 (padding)
    # 輸出大小會是 (input_h - kernel_h + 1) x (input_w - kernel_w + 1)
    # 對於 5x5 輸入和 5x5 濾波器，輸出是 1x1
    output_h_valid = input_h - kernel_h + 1
    output_w_valid = input_w - kernel_w + 1
    output_valid = np.zeros((output_h_valid, output_w_valid))

    for y in range(output_h_valid):
        for x in range(output_w_valid):
            # 選取輸入矩陣中要進行運算的區域
            region = input_matrix[y : y + kernel_h, x : x + kernel_w]
            # 進行元素級乘積並求和
            output_valid[y, x] = np.sum(region * kernel)

    # 2. 'same' 卷積: 使用填充，使得輸出大小與輸入相同
    # 需要的 padding 大小
    pad_h = (kernel_h - 1) // 2
    pad_w = (kernel_w - 1) // 2
    
    if pad_h > 0 or pad_w > 0:
        padded_input = np.pad(input_matrix, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    else:
        padded_input = input_matrix
        
    output_same = np.zeros_like(input_matrix, dtype=float)

    for y in range(input_h):
        for x in range(input_w):
            region = padded_input[y : y + kernel_h, x : x + kernel_w]
            output_same[y, x] = np.sum(region * kernel)

    return output_valid, output_same


if __name__ == "__main__":
    filepath = 'convtest_input.txt'
    input_matrix, filter_matrix = read_matrices_from_file(filepath)

    if input_matrix is not None and filter_matrix is not None:
        print("--- 讀取到的輸入矩陣 (5x5) ---")
        print(input_matrix)
        print("\n--- 讀取到的濾波器矩陣 (5x5) ---")
        print(filter_matrix)

        # 執行卷積計算
        valid_result, same_result = perform_convolution(input_matrix, filter_matrix)

        print("\n" + "="*40)
        print("           卷積計算結果")
        print("="*40)
        
        print("\n--- 結果 1: 'Valid' 卷積 (無填充) ---")
        print("由於輸入和濾波器大小相同，濾波器只能放置在一個位置，因此結果為一個純量值。")
        print(valid_result)

        print("\n--- 結果 2: 'Same' 卷積 (有填充) ---")
        print("在輸入矩陣周圍補零，使得輸出結果的維度與原始輸入相同 (5x5)。")
        print(same_result)