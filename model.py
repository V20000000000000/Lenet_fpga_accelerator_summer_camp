# model.py
import torch
import torch.nn as nn
import torch.quantization

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 為了量化，我們加入 QuantStub 和 DeQuantStub
        self.quant = torch.quantization.QuantStub()
        
        self.features = nn.Sequential(
            # Input: 1x28x28 (MNIST image size)
            # The original LeNet-5 used 32x32. Padding=2 adapts it for 28x28.
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   
            nn.ReLU(),
            # Output: 6x28x28
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Output: 6x14x14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            # Output: 16x10x10
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Output: 16x5x5
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10)
        )
        
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # 在量化模式下，stubs 會轉換資料類型
        # 在一般模式下，stubs 不做任何事
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x
    
class Modified_LeNet5(nn.Module):
    def __init__(self):
        super(Modified_LeNet5, self).__init__()
        # 為了量化，我們加入 QuantStub 和 DeQuantStub
        self.quant = torch.quantization.QuantStub()
        
        self.features = nn.Sequential( 
            # Input: 1x28x28 (MNIST image size)
            # The original LeNet-5 used 32x32. 
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=0),   
            nn.ReLU(),
            # Output: 1x24x24
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: 1x12x12
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1),
            nn.ReLU(),
            # Output: 1x8x8
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: 1x4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1*4*4, out_features=10)
        )
        
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # 在量化模式下，stubs 會轉換資料類型
        # 在一般模式下，stubs 不做任何事
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x