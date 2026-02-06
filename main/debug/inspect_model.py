import torch
import torch.nn as nn
import sys
import os

# 获取当前脚本所在目录: d:\Project\auto-wzry\main\debug
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录: d:\Project\auto-wzry
project_root = os.path.dirname(os.path.dirname(current_dir))

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from main.model import ActorCritic

def inspect_model_flow():
    print("==================================================")
    print("       🔍 AI 模型数据流显微镜 (Model Inspector)       ")
    print("==================================================")
    print("正在初始化模型...\n")

    # 1. 初始化模型
    # 假设动作维度为 4 (x, y, is_touch, reserved)
    model = ActorCritic(state_dim=(3, 224, 224), action_dim=4)
    model.eval() # 切换到评估模式 (不启用 Dropout/BatchNorm 更新)

    # 2. 创建一个伪造的输入数据 (模拟一张图片)
    # 形状: [Batch=1, Channels=3, Height=224, Width=224]
    fake_image = torch.randn(1, 3, 224, 224)
    
    print(f"1. [输入层] 模拟游戏画面")
    print(f"   - 形状: {fake_image.shape}")
    print(f"   - 说明: 这是一个 224x224 的 RGB 图像矩阵")
    print("-" * 50)

    # 3. 逐步执行 forward 中的逻辑
    
    # --- Step 1: Encoder ---
    print(f"2. [视觉中枢] CNN Encoder 特征提取")
    features = model.encoder(fake_image)
    print(f"   - 操作: features = model.encoder(image)")
    print(f"   - 输出形状: {features.shape}")
    print(f"   - 前10个特征值: {features[0][:10].detach().numpy()}")
    print(f"   - 说明: 图片被压缩成了 512 个抽象数字")
    print("-" * 50)

    # --- Step 2: Actor Head ---
    print(f"3. [决策部] Actor Head 生成动作")
    # 注意：ActorCritic 的 forward 会经过 sigmoid，这里我们为了演示拆开来看
    actor_output = model.actor_mean(features)
    # 经过 Sigmoid 归一化到 0-1
    action_mean = torch.sigmoid(actor_output)
    
    print(f"   - 操作: mean = sigmoid(model.actor_mean(features))")
    print(f"   - 输出形状: {action_mean.shape}")
    print(f"   - 具体数值: {action_mean[0].detach().numpy()}")
    print(f"   - 解读: [x坐标, y坐标, 按下概率, 预留]")
    print("-" * 50)

    # --- Step 3: Critic Head ---
    print(f"4. [评估部] Critic Head 局势打分")
    value = model.critic(features)
    
    print(f"   - 操作: value = model.critic(features)")
    print(f"   - 输出形状: {value.shape}")
    print(f"   - 局势评分: {value.item():.4f}")
    print(f"   - 解读: 分数越高，代表当前局势越有利")
    print("-" * 50)
    
    print("\n✅ 演示结束！这就是数据在 AI 大脑中流动的全过程。")

if __name__ == "__main__":
    inspect_model_flow()
