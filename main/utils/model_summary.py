import torch
import sys
import os

# 添加项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from main.model import ActorCritic

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # 假设输入是 224x224 RGB
    state_dim = (3, 224, 224)
    action_dim = 4
    
    model = ActorCritic(state_dim, action_dim)
    
    print(f"Model Structure:\n{model}")
    
    total_params = count_parameters(model)
    print(f"\nTotal Trainable Parameters: {total_params:,}")
    
    # 估算模型大小 (float32 = 4 bytes)
    model_size_mb = total_params * 4 / 1024 / 1024
    print(f"Estimated Model Size: {model_size_mb:.2f} MB")
    
    # 测试前向传播
    dummy_input = torch.randn(1, 3, 224, 224)
    try:
        mean, std, value = model(dummy_input)
        print(f"\nForward pass check: Success")
        print(f"Output shapes - Mean: {mean.shape}, Std: {std.shape}, Value: {value.shape}")
    except Exception as e:
        print(f"\nForward pass check: Failed - {e}")

if __name__ == "__main__":
    main()