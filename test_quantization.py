# test_quantization.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.quantization
from torch.quantization import QConfig, MinMaxObserver
import os
import numpy as np

# 從 model.py 導入 LeNet5 class
from model import Modified_LeNet5 as Lenet5

# --- Model Helper Functions ---

def print_model_parameters(model, model_type="Model"):
    """Calculates and prints the total number of parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{model_type}] Total number of parameters: {total_params:,}")
    return total_params

def fuse_model(model):
    """Fuses convolution/BN/relu modules for quantization."""
    print("Fusing model modules...")
    # 注意: 我們需要根據 model.py 中的結構來指定融合的層
    torch.quantization.fuse_modules(model.features, [['0', '1'], ['3', '4']], inplace=True)
    # torch.quantization.fuse_modules(model.classifier, [['1', '2']], inplace=True)
    print("Fusing complete.")

def quantize_static(model, data_loader, backend='fbgemm', device='cpu'):
    """Performs static quantization on the model."""
    print("\n--- Starting Static Quantization ---")
    model.eval()
    model.to(device)
    fuse_model(model)
    model.qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8),
        weight=torch.quantization.default_weight_observer.with_args(dtype=torch.qint8)
    )
    print("Preparing model for quantization...")
    torch.quantization.prepare(model, inplace=True)
    print("Calibrating model with data...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= 100:
                break
            model(inputs.to(device))
    print("Calibration complete.")
    print("Converting model to quantized version...")
    torch.quantization.convert(model, inplace=True)
    print("--- Static Quantization Finished ---")
    return model

def evaluate(model, test_loader, model_type="Quantized", device='cpu'):
    """Evaluates the model's accuracy on the test dataset."""
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
    print(f"[{model_type} Model] Accuracy on test set: {acc:.4f}")
    return acc

# --- Weight & Parameter Saving Functions ---

def print_quantization_details(model: nn.Module):
    print("🔍 Detailed Quantized Layers Info:")
    for name, module in model.named_modules():
        # 檢查是否具有 scale 和 zero_point 屬性
        if hasattr(module, 'scale') and hasattr(module, 'zero_point'):
            scale = module.scale
            zero_point = module.zero_point

            # 如果是 Tensor，就呼叫 .item()；否則保持原狀
            if isinstance(scale, torch.Tensor):
                scale = scale.item()
            if isinstance(zero_point, torch.Tensor):
                zero_point = zero_point.item()

            print(f"  - {name}: scale={scale:.6f}, zero_point={zero_point}")

def print_weight_quantization_details(model: nn.Module):
    """Prints the scale and zero-point for the weights of quantized layers."""
    print("\n⚖️  Detailed Weight Quantization Info:")
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.quantized.Linear, torch.nn.quantized.Conv2d)):
            # 取得量化後的權重 tensor
            q_weight = module.weight()
            
            # 檢查量化方案 (per-tensor or per-channel)
            if q_weight.qscheme() == torch.per_tensor_affine:
                scale = q_weight.q_scale()
                zero_point = q_weight.q_zero_point()
                print(f"  - {name} (per-tensor): scale={scale:.8f}, zero_point={zero_point}")
            
            elif q_weight.qscheme() == torch.per_channel_affine:
                scales = q_weight.q_per_channel_scales()
                zero_points = q_weight.q_per_channel_zero_points()
                print(f"  - {name} (per-channel):")
                print(f"    - scales shape: {scales.shape}")
                print(f"    - scales tensor: {scales}")
                print(f"    - zero_points tensor: {zero_points}")

def save_quantized_int_weights(model, folder='param_data'):
    """Saves the INT8 weights of quantized layers to text files."""
    print(f"\n💾 Saving INT8 weights to '{folder}' directory...")
    os.makedirs(folder, exist_ok=True)
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.quantized.Linear, torch.nn.quantized.Conv2d)):
            q_weight = module.weight()
            weight_int = q_weight.int_repr().cpu().numpy()
            flat_weights = weight_int.reshape(weight_int.shape[0], -1)
            out_file = os.path.join(folder, f"{name.replace('.', '_')}.txt")
            np.savetxt(out_file, flat_weights, fmt="%d")
            print(f"  - [Saved] {out_file} with shape {flat_weights.shape}")

# --- Inference Function ---

def infer_from_txt(txt_path, model, transform, device='cpu'):
    """
    Loads an image from a text file, applies transformations, and runs inference.
    """
    print(f"\nRunning inference on {txt_path}...")
    try:
        arr = np.loadtxt(txt_path, dtype=np.float32).reshape(28, 28)
    except Exception as e:
        print(f"Error loading or reshaping text file: {e}")
        return None

    img_tensor = torch.from_numpy(arr)
    img_tensor = img_tensor.unsqueeze(0)
    if transform and transform.transforms:
        for t in transform.transforms:
            if isinstance(t, transforms.Normalize):
                img_tensor = t(img_tensor)

    img_tensor = img_tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(img_tensor)
    
    pred = out.argmax(dim=1).item()
    print(f"Inference complete. Predicted class: {pred}")
    return pred


if __name__ == '__main__':
    # --- 1. Setup & Data Loading ---
    DEVICE = 'cpu'
    
    # data transformation pipeline(preprocessing)
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
    ])
    
    if not os.path.exists("lenet.pt"):
        print("Pre-trained model 'lenet.pt' not found. Please run 'train.py' first.")
        # Creating a dummy model to avoid crashing, but it won't be accurate
        dummy_model = Lenet5()
        torch.save(dummy_model.state_dict(), "lenet.pt")

    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform_pipeline)    
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform_pipeline)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # --- 2. Load Model and Quantize ---
    quant_model = Lenet5()
    print("Loading pre-trained model...")
    print(quant_model)
    quant_model.load_state_dict(torch.load("lenet.pt", map_location=torch.device(DEVICE)))

    print_model_parameters(quant_model, model_type="Floating Point")

    quant_model = quantize_static(quant_model, train_loader, device=DEVICE)
    print("Quantized model structure:")
    print(quant_model)
    # --- 3. Analyze and Save Quantized Model ---
    print_quantization_details(quant_model)
    print_weight_quantization_details(quant_model)
    save_quantized_int_weights(quant_model, folder='param_data')
    evaluate(quant_model, test_loader, model_type="Quantized", device=DEVICE)
    torch.save(quant_model.state_dict(), 'quant_model_statedict.pt')
    print("\n✅ Saved quantized model state_dict to 'quant_model_statedict.pt'")

    # --- 4. Test Inference from Text File ---
    txt_file_path = "valid_data/picture/0.txt"
    if not os.path.exists(txt_file_path):
        os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
        print(f"\nCreating a dummy inference file at '{txt_file_path}'...")
        sample_img_tensor, label = test_ds[0]
        sample_img_unnormalized = sample_img_tensor * 0.5 + 0.5
        sample_img_int = (sample_img_unnormalized * 255).byte().numpy().squeeze()
        np.savetxt(txt_file_path, sample_img_int, fmt="%d")
        print(f"Dummy file created with image of a '{label}'.")

    infer_from_txt(txt_file_path, quant_model, transform=transform_pipeline, device=DEVICE)
