import torch
import torch.nn as nn
import os
from model import Modified_LeNet5 as MLeNet5

def print_tensor_stats(tensor, name):
    """Prints statistics for a given tensor."""
    print(f"Stats for '{name}':")
    print(f"  Shape: {tensor.shape}")
    print(f"  Min value: {tensor.min().item():.6f}")
    print(f"  Max value: {tensor.max().item():.6f}")
    print(f"  Mean value: {tensor.mean().item():.6f}")
    print("-" * 30)

def custom_symmetric_quantize(tensor, scale, bits):
    """
    Performs custom symmetric quantization on a tensor.
    This function mimics the paper's approach with a fixed scale and zero-point=0.

    Args:
        tensor (torch.Tensor): The input floating-point tensor.
        scale (float): The fixed quantization scale (e.g., 2^-7).
        bits (int): The number of bits for the quantized integer.

    Returns:
        torch.Tensor: The quantized integer tensor.
    """
    # Calculate the min and max integer values based on the number of bits
    # For a signed integer (symmetric), the range is [-(2^(bits-1)), 2^(bits-1) - 1]
    q_min = -(2**(bits - 1))
    q_max = 2**(bits - 1) - 1

    # 1. Scale the tensor
    scaled_tensor = tensor / scale
    
    # 2. Round to the nearest integer
    rounded_tensor = torch.round(scaled_tensor)
    
    # 3. Clamp the values to the allowed integer range
    quantized_tensor = torch.clamp(rounded_tensor, q_min, q_max).to(torch.int32) # Use int32 for safety
    
    return quantized_tensor

def save_quantized_params_to_txt(quantized_state_dict, output_dir='quantized_params_8bit'):
    """Saves quantized parameters to text files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"💾 Saving INT{BITS} weights to '{output_dir}' directory...")
    for name, tensor in quantized_state_dict.items():
        # Flatten the tensor to save as a 1D list of numbers
        flat_tensor = tensor.flatten()
        
        # Create a clean filename
        filename = name.replace('.', '_') + '.txt'
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            for val in flat_tensor:
                f.write(f"{val.item()}\n")
        print(f"  - [Saved] {filepath} with original shape {list(tensor.shape)}")


# --- Main Conversion Logic ---
if __name__ == "__main__":
    # --- Configuration ---
    MODEL_PATH = 'lenet.pt'
    # For 8-bit signed numbers, we can use a Q1.7 format to maximize fractional precision.
    # This means we still have 7 fractional bits, so the scale remains 2^-7.
    SCALE = 2**-7  # This is 0.0078125
    BITS = 8       # MODIFIED: Total bits for the signed integer representation

    print("--- Custom Symmetric Quantization (8-bit) ---")
    print(f"Using fixed scale (2^-7): {SCALE}")
    print(f"Using signed integer bits: {BITS}\n")

    # 1. Load the pre-trained floating-point model
    device = torch.device("cpu") # Quantization is typically done on the CPU
    fp32_model = MLeNet5().to(device)
    
    try:
        fp32_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✅ Successfully loaded floating-point model from '{MODEL_PATH}'")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at '{MODEL_PATH}'.")
        print("Please make sure you have trained the model and saved 'lenet.pt'.")
        exit()
        
    fp32_model.eval()
    
    # 2. Create a new state dictionary for the quantized weights
    quantized_state_dict = {}
    fp32_state_dict = fp32_model.state_dict()

    print("\n--- Starting Quantization Process ---")
    for name, param in fp32_state_dict.items():
        if param.dtype == torch.float32: # We only quantize float tensors (weights and biases)
            print(f"Quantizing layer: {name}")
            # print_tensor_stats(param, f"Original FP32 - {name}")
            
            # Apply our custom quantization function
            quantized_param = custom_symmetric_quantize(param, scale=SCALE, bits=BITS)
            quantized_state_dict[name] = quantized_param
            
            # print_tensor_stats(quantized_param.float(), f"Quantized INT{BITS} - {name}")
        else:
            # Copy over non-float parameters if any (e.g., batchnorm tracking stats)
            quantized_state_dict[name] = param
    print("--- Quantization Finished ---\n")

    # 3. Save the quantized integer parameters to text files for hardware use
    save_quantized_params_to_txt(quantized_state_dict)

    # 4. Save the complete quantized state dictionary as a PyTorch file for simulation
    quantized_model_path = 'quantized_lenet_custom_8bit.pt'
    torch.save(quantized_state_dict, quantized_model_path)
    print(f"\n✅ Saved custom quantized model state_dict to '{quantized_model_path}'")