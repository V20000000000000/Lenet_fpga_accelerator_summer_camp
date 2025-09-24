# manual_fixed_point_quant.py (FINAL VERSION)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- 1. 定義量化器和定點數格式 ---
N_BITS = 7
SCALE = 2.0**(-N_BITS)

def symmetric_quantize(x):
    x_quant = torch.round(x / SCALE)
    x_quant = torch.clamp(x_quant, -128, 127)
    return x_quant

def symmetric_dequantize(x_quant):
    return x_quant.float() * SCALE

# --- 2. 建立自定義的量化神經層 ---
class QuantizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.stride = stride
        self.padding = padding
        # 初始化權重 (可選，但好習慣)
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x_quant = symmetric_quantize(x)
        w_quant = symmetric_quantize(self.weight)
        
        # 執行整數乘加
        out = F.conv2d(x_quant.float(), w_quant.float(), stride=self.stride, padding=self.padding)
        
        # 添加偏置 (bias)，偏置通常是 INT32，這裡簡化模擬
        if self.bias is not None:
            # 正確的偏置量化比較複雜，這裡我們直接在浮點域添加
            out_dequant = symmetric_dequantize(out)
            bias_fp = self.bias.view(1, -1, 1, 1).expand_as(out_dequant)
            out_dequant = out_dequant + bias_fp
            out = symmetric_quantize(out_dequant) # 重新量化
        
        # Requantization
        out = torch.round(out / (2.0**N_BITS))
        out = torch.clamp(out, -128, 127)
        
        # 反量化
        out_dequant = symmetric_dequantize(out)
        return out_dequant

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x_quant = symmetric_quantize(x)
        w_quant = symmetric_quantize(self.weight)

        out = F.linear(x_quant.float(), w_quant.float())
        
        if self.bias is not None:
            out_dequant = symmetric_dequantize(out)
            out_dequant = out_dequant + self.bias
            out = symmetric_quantize(out_dequant)
            
        out = torch.round(out / (2.0**N_BITS))
        out = torch.clamp(out, -128, 127)
        
        out_dequant = symmetric_dequantize(out)
        return out_dequant

# --- 3. 建立使用量化層的模型 ---
class Quantized_Slide_LeNet_Named(nn.Module):
    def __init__(self):
        super().__init__()
        # 單獨定義每一層，給它們固定的名字
        self.conv1 = QuantizedConv2d(in_channels=1, out_channels=1, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = QuantizedConv2d(in_channels=1, out_channels=1, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = QuantizedLinear(in_features=25, out_features=10)

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2), "constant", 0)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# --- 4. 評估函數 (不變) ---
def evaluate(model, test_loader, model_type="Model", device='cpu'):
    # ... (程式碼同上，未更改)
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"[{model_type}] Accuracy on test set: {acc:.4f}")
    return acc

# --- 5. 主執行流程 ---
if __name__ == '__main__':
    DEVICE = 'cpu'
    
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform_pipeline)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    fp32_model_path = "lenet.pt"
    if not os.path.exists(fp32_model_path):
        raise FileNotFoundError(f"'{fp32_model_path}' not found. Please run train.py first.")

    # 建立我們的量化模型
    quant_model = Quantized_Slide_LeNet_Named().to(DEVICE)
    
    # 讀取訓練好的浮點權重
    fp32_state_dict = torch.load(fp32_model_path, map_location=DEVICE)
    
    # 建立一個新的 state_dict 來匹配我們的新模型命名
    new_state_dict = {}
    new_state_dict['conv1.weight'] = fp32_state_dict['features.0.weight']
    new_state_dict['conv1.bias'] = fp32_state_dict['features.0.bias']
    new_state_dict['conv2.weight'] = fp32_state_dict['features.3.weight']
    new_state_dict['conv2.bias'] = fp32_state_dict['features.3.bias']
    new_state_dict['fc1.weight'] = fp32_state_dict['classifier.1.weight']
    new_state_dict['fc1.bias'] = fp32_state_dict['classifier.1.bias']
    
    # 加載重新命名的權重
    quant_model.load_state_dict(new_state_dict)
    
    print("Successfully loaded and renamed weights into the quantized model.")
    
    print("\nEvaluating model with simulated fixed-point arithmetic...")
    evaluate(quant_model, test_loader, model_type="Simulated Fixed-Point", device=DEVICE)