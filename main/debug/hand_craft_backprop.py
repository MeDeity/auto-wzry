import torch
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # s'(x) = s(x) * (1 - s(x))
    s = sigmoid(x)
    return s * (1 - s)

def hand_craft_backprop():
    print("==================================================")
    print("       ✍️ 手搓反向传播 (Hand-Craft Backprop)       ")
    print("==================================================")
    print("我们将用 NumPy 手动计算梯度，并与 PyTorch 的自动求导结果进行 PK。\n")

    # ------------------------------------------------
    # 1. 初始化数据和权重
    # ------------------------------------------------
    # 简单的 3 层网络: 输入层(2) -> 隐藏层(2) -> 输出层(1)
    
    # 输入数据 x (2个特征)
    x_data = np.array([0.5, 0.1])
    # 目标真值 target (1个数值)
    y_target = np.array([0.8])

    print(f"1. [准备数据]")
    print(f"   输入 x: {x_data}")
    print(f"   目标 y: {y_target}")
    print("-" * 30)

    # 初始化权重 (为了方便计算，我们固定具体的值)
    # W1: 输入层到隐藏层的权重 (2x2)
    w1_val = np.array([[0.1, 0.2], 
                       [0.3, 0.4]])
    # W2: 隐藏层到输出层的权重 (2x1)
    w2_val = np.array([[0.5], 
                       [0.6]])

    print(f"2. [初始化权重]")
    print(f"   W1 (Input->Hidden):\n{w1_val}")
    print(f"   W2 (Hidden->Output):\n{w2_val}")
    print("-" * 30)

    # ------------------------------------------------
    # PyTorch 自动求导版 (作为标准答案)
    # ------------------------------------------------
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_target, dtype=torch.float32)
    
    w1_tensor = torch.tensor(w1_val, dtype=torch.float32, requires_grad=True)
    w2_tensor = torch.tensor(w2_val, dtype=torch.float32, requires_grad=True)

    # 前向传播
    # h_raw = x @ W1
    h_raw_tensor = torch.matmul(x_tensor, w1_tensor) 
    # h = sigmoid(h_raw)
    h_tensor = torch.sigmoid(h_raw_tensor)
    # y_pred = h @ W2
    y_pred_tensor = torch.matmul(h_tensor, w2_tensor)
    
    # 计算 Loss (MSE)
    loss_tensor = 0.5 * (y_pred_tensor - y_tensor) ** 2

    # 反向传播
    loss_tensor.backward()

    print("3. [PyTorch 自动求导结果]")
    print(f"   Loss: {loss_tensor.item():.6f}")
    print(f"   W2.grad:\n{w2_tensor.grad.numpy()}")
    print(f"   W1.grad:\n{w1_tensor.grad.numpy()}")
    print("-" * 30)

    # ------------------------------------------------
    # NumPy 手搓版 (Manual Calculation)
    # ------------------------------------------------
    print("4. [NumPy 手搓推导过程]")

    # --- A. 前向传播 (Forward Pass) ---
    # h_raw = x @ W1
    # [0.5, 0.1] @ [[0.1, 0.2], = [0.5*0.1 + 0.1*0.3, 0.5*0.2 + 0.1*0.4]
    #               [0.3, 0.4]] = [0.05 + 0.03,       0.10 + 0.04]
    #                           = [0.08,              0.14]
    h_raw = np.dot(x_data, w1_val)
    
    # h = sigmoid(h_raw)
    h = sigmoid(h_raw) # [sigmoid(0.08), sigmoid(0.14)] ≈ [0.519989, 0.534943]
    
    # y_pred = h @ W2
    # [0.5199, 0.5349] @ [[0.5], = 0.5199*0.5 + 0.5349*0.6
    #                     [0.6]]
    y_pred = np.dot(h, w2_val) # scalar
    
    # Loss
    loss = 0.5 * (y_pred - y_target) ** 2

    print(f"   [Forward] h_raw: {h_raw}")
    print(f"   [Forward] h (activated): {h}")
    print(f"   [Forward] y_pred: {y_pred}")
    print(f"   [Forward] Loss: {loss}")

    # --- B. 反向传播 (Backward Pass) ---
    # 这里的核心是链式法则 (Chain Rule)
    # Loss = 0.5 * (y_pred - y_target)^2
    
    # 1. 计算 Loss 对 y_pred 的导数
    # d(Loss)/d(y_pred) = y_pred - y_target
    d_loss_d_ypred = y_pred - y_target
    
    # 2. 计算 Loss 对 W2 的导数
    # y_pred = h @ W2
    # d(y_pred)/d(W2) = h.T (转置)
    # 链式: d(Loss)/d(W2) = d(Loss)/d(y_pred) * d(y_pred)/d(W2)
    #                     = (y_pred - y_target) * h
    d_loss_d_w2 = d_loss_d_ypred * h.reshape(-1, 1) # reshape为了匹配形状

    # 3. 计算 Loss 对 h 的导数 (为了继续传给 W1)
    # d(y_pred)/d(h) = W2.T
    # d(Loss)/d(h) = d(Loss)/d(y_pred) * d(y_pred)/d(h)
    d_loss_d_h = d_loss_d_ypred * w2_val.T # [1, 2]

    # 4. 计算 Loss 对 h_raw 的导数 (穿过 Sigmoid 激活函数)
    # h = sigmoid(h_raw)
    # d(h)/d(h_raw) = h * (1 - h)
    # d(Loss)/d(h_raw) = d(Loss)/d(h) * d(h)/d(h_raw)
    d_loss_d_hraw = d_loss_d_h * (h * (1 - h)) # [1, 2]

    # 5. 计算 Loss 对 W1 的导数
    # h_raw = x @ W1
    # d(h_raw)/d(W1) = x.T
    # d(Loss)/d(W1) = x.T @ d(Loss)/d(h_raw)
    d_loss_d_w1 = np.outer(x_data, d_loss_d_hraw) # [2, 2]

    print("\n   [Backward] 梯度计算结果:")
    print(f"   W2 Gradient (Hand):\n{d_loss_d_w2}")
    print(f"   W1 Gradient (Hand):\n{d_loss_d_w1}")

    # ------------------------------------------------
    # 验证
    # ------------------------------------------------
    print("-" * 30)
    print("5. [验证结论]")
    
    diff_w2 = np.abs(d_loss_d_w2 - w2_tensor.grad.numpy()).sum()
    diff_w1 = np.abs(d_loss_d_w1 - w1_tensor.grad.numpy()).sum()
    
    if diff_w2 < 1e-5 and diff_w1 < 1e-5:
        print("✅ 成功！手搓梯度与 PyTorch 结果完全一致！")
        print("这证明了：所谓 AI 的学习，本质上就是这些基础的微积分运算。")
    else:
        print("❌ 失败！结果有偏差，请检查推导过程。")
        print(f"Diff W2: {diff_w2}")
        print(f"Diff W1: {diff_w1}")

if __name__ == "__main__":
    hand_craft_backprop()
