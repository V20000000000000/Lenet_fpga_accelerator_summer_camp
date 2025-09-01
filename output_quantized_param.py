import torch
import os
import numpy as np
from torchvision import transforms, datasets
from cnn import CNN
from torch.quantization import fuse_modules, prepare, convert, get_default_qconfig

# 1. 重建 & load quantized 模型（同之前）
def build_qmodel(path):
    m = CNN().cpu().eval()
    fuse_modules(m, [['conv1','relu1'], ['conv2','relu2']], inplace=True)
    m.qconfig = get_default_qconfig('fbgemm')
    prepare(m, inplace=True)
    convert(m, inplace=True)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    return m

# 2. 读 txt 到 float 张量
def load_txt(path):
    arr = np.loadtxt(path, dtype=np.int32)
    # 这里我们直接把 0-255 当 uint8，再给 QuantStub
    # 所以转 float [0,1]
    f = torch.from_numpy(arr.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0)
    return f

def save_int8(name, qtensor):
    arr = qtensor.int_repr().squeeze().numpy()
    np.savetxt(f"interconnect_node/{name}.txt", arr, fmt="%4d")

def print_quantization_config():
    # 檢查輸出資料夾
    os.makedirs("param_data", exist_ok=True)

    # 重建量化模型
    qmodel = build_qmodel('quant_model.pt')

    # 取得所有 scale & zero_point
    labels = [
        "QuantStub",
        "Conv1 W","Conv1 Act", 
        "Conv2 W","Conv2 Act",
        "FC W", "FC Act"
    ]
    scales = [
        float(qmodel.quant.scale),
        float(qmodel.conv1.weight().q_scale()), float(qmodel.conv1.scale),
        float(qmodel.conv2.weight().q_scale()), float(qmodel.conv2.scale),
        float(qmodel.fc.weight().q_scale()),    float(qmodel.fc.scale)
    ]
    zero_points = [
        int(qmodel.quant.zero_point),
        int(qmodel.conv1.weight().q_zero_point()), int(qmodel.conv1.zero_point),
        int(qmodel.conv2.weight().q_zero_point()), int(qmodel.conv2.zero_point),
        int(qmodel.fc.weight().q_zero_point()),    int(qmodel.fc.zero_point)
    ]

    # 保存數值到 CSV
    with open("param_data/quant_params.txt", "w") as f:
        for name, sc, zp in zip(labels, scales, zero_points):
            f.write(f"{sc:.40f} {zp}\n")

    print("已將量化參數與圖檔儲存到 param_data/")

if __name__=='__main__':

    qmodel = build_qmodel('lenet.pt') 
    
    print(qmodel)

    # —— 新增：打印所有需要的 scale / zero_point —— 
    print("\n=== Quantization Parameters ===")
    # QuantStub
    sc_q    = float(qmodel.quant.scale)
    zp_q    = int(qmodel.quant.zero_point)
    print(f"Input QuantStub:      scale={sc_q:.6f}, zero_point={zp_q}")

    # Conv1 activation
    sc_c1   = float(qmodel.conv1.scale)
    zp_c1   = int(qmodel.conv1.zero_point)
    print(f"Conv1  activation:     scale={sc_c1:.6f}, zero_point={zp_c1}")
    # Conv1 weight
    w1      = qmodel.conv1.weight()
    sc_w1   = float(w1.q_scale())
    zp_w1   = int(w1.q_zero_point())
    print(f"Conv1  weight:         scale={sc_w1:.6f}, zero_point={zp_w1}")

    # Conv2 activation
    sc_c2   = float(qmodel.conv2.scale)
    zp_c2   = int(qmodel.conv2.zero_point)
    print(f"Conv2  activation:     scale={sc_c2:.6f}, zero_point={zp_c2}")
    # Conv2 weight
    w2      = qmodel.conv2.weight()
    sc_w2   = float(w2.q_scale())
    zp_w2   = int(w2.q_zero_point())
    print(f"Conv2  weight:         scale={sc_w2:.6f}, zero_point={zp_w2}")

    # FC activation
    sc_fc   = float(qmodel.fc.scale)
    zp_fc   = int(qmodel.fc.zero_point)
    print(f"FC     activation:     scale={sc_fc:.6f}, zero_point={zp_fc}")
    # FC weight
    wf      = qmodel.fc.weight()
    sc_wf   = float(wf.q_scale())
    zp_wf   = int(wf.q_zero_point())
    print(f"FC     weight:         scale={sc_wf:.6f}, zero_point={zp_wf}")

    x_float = load_txt('valid_data/picture/0.txt')
    # 3. 先量化输入 → 得到一个 torch.quint8
    x_q = qmodel.quant(x_float)
    print("input int8:\n", x_q.int_repr().squeeze().numpy())

    print_quantization_config()

    # 4. 手动走各层，不调用 dequant
    c1_q = qmodel.conv1(x_q)      ;  print("conv1 out int8:\n", c1_q.int_repr().squeeze().numpy())
    p1_q = qmodel.pool1(c1_q)     ;  print("pool1 out int8:\n", p1_q.int_repr().squeeze().numpy())
    c2_q = qmodel.conv2(p1_q)     ;  print("conv2 out int8:\n", c2_q.int_repr().squeeze().numpy())
    p2_q = qmodel.pool2(c2_q)     ;  print("pool2 out int8:\n", p2_q.int_repr().squeeze().numpy())

    # flatten + fc
    flat_q = p2_q.reshape(1, -1); print("flat_q out int8:\n", flat_q.int_repr().squeeze().numpy())
    fc_q   = qmodel.fc(flat_q)
    out_int = fc_q.int_repr().squeeze().numpy()
    print("fc out int8:\n", out_int)

    
    save_int8("input",  x_q)
    save_int8("conv1",  c1_q)
    save_int8("pool1",  p1_q)
    save_int8("conv2",  c2_q)
    save_int8("pool2",  p2_q)
    save_int8("fc",     fc_q)

    # —— 新增：在 int8 logits 上做 argmax —— 
    pred_label = int(np.argmax(out_int))
    print("Predicted label (from int8 logits):", pred_label)