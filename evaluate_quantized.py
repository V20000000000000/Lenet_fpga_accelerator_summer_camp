import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Modified_LeNet5 as MLeNet5

def evaluate_quantized_model():
    """
    載入客製化的量化權重，並在測試集上評估其精準度。
    """
    # --- Configuration (必須與量化腳本中的設定完全相同) ---
    QUANTIZED_MODEL_PATH = 'quantized_lenet_custom_8bit.pt'
    SCALE = 2**-7  # 縮放因子 (0.0078125)
    BITS = 8       # 量化位元數

    print("--- 評估客製化 8-bit 量化模型 ---")
    print(f"從 '{QUANTIZED_MODEL_PATH}' 載入權重")
    print(f"使用固定 Scale: {SCALE}\n")

    # --- 1. 準備測試資料 ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    # --- 2. 建立模型並載入量化後的權重 ---
    device = torch.device("cpu") # 評估可以在 CPU 上完成
    model = MLeNet5().to(device)
    
    try:
        # 載入 8-bit 整數權重
        quantized_state_dict = torch.load(QUANTIZED_MODEL_PATH, map_location=device)
    except FileNotFoundError:
        print(f"❌ 錯誤: 找不到量化模型檔案 '{QUANTIZED_MODEL_PATH}'。")
        print("請先執行 quantize_custom.py 來產生權重檔案。")
        return

    # 建立一個新的 state_dict 來存放反量化後的浮點權重
    dequantized_state_dict = {}
    
    print("正在將 INT8 權重反量化回 FP32 以進行評估...")
    for name, param in quantized_state_dict.items():
        # 我們只需要處理權重和偏置 (它們在量化時被轉為整數)
        if param.dtype == torch.int32:
            # 反量化: 將整數權重乘以 scale，還原回浮點數
            dequantized_param = param.float() * SCALE
            dequantized_state_dict[name] = dequantized_param
        else:
            dequantized_state_dict[name] = param
            
    # 將反量化後的權重載入到模型中
    model.load_state_dict(dequantized_state_dict)
    print("權重載入完成。\n")

    # --- 3. 在測試集上評估模型 ---
    model.eval() # 設定為評估模式
    correct = 0
    total = 0

    with torch.no_grad(): # 在評估時不需要計算梯度
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ 量化後模型的精準度: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_quantized_model()