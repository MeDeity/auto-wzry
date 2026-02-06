import torch
import torch.nn as nn
import sys
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from main.model import ActorCritic
from main.algo.ppo import PPO

def inspect_ppo_update():
    print("==================================================")
    print("       🧬 PPO 进化实验室 (Evolution Lab)       ")
    print("==================================================")
    
    # 1. 准备实验环境
    # 模拟一个小规模的模型和数据
    action_dim = 4
    model = ActorCritic(state_dim=(3, 224, 224), action_dim=action_dim)
    
    # 创建 PPO 算法实例
    ppo = PPO(model, lr=0.01, ppo_epochs=1, batch_size=2) # 使用较大的 lr 以便观察变化
    
    print("1. [初始状态] 模型已就位")
    print("   - 我们假设模型在某次团战中，胡乱放了一个技能。")
    print("   - 此时 Critic 给出的评分 (Value) 也许很低。")
    print("-" * 50)
    
    # 2. 伪造一段“经验” (Rollout Data)
    # 假设 Batch Size = 2
    fake_states = torch.randn(2, 3, 224, 224)
    fake_actions = torch.randn(2, 4)
    fake_log_probs = torch.tensor([-1.0, -1.0]) # 假设旧的概率
    fake_returns = torch.tensor([1.0, 0.5])     # 实际回报 (Reward): 一个好(1.0)，一个一般(0.5)
    fake_advantages = torch.tensor([0.5, -0.2]) # 优势值: 第一个动作比平均好，第二个比平均差
    fake_values = torch.tensor([0.5, 0.7])      # Critic 之前的预测
    
    rollouts = {
        'states': fake_states,
        'actions': fake_actions,
        'log_probs': fake_log_probs,
        'returns': fake_returns,
        'advantages': fake_advantages,
        'values': fake_values
    }
    
    print(f"2. [收集经验] AI 回忆刚才的操作")
    print(f"   - 动作1: 优势值 = {fake_advantages[0]:.1f} (做得好！应鼓励)")
    print(f"   - 动作2: 优势值 = {fake_advantages[1]:.1f} (做得差！应惩罚)")
    print("-" * 50)
    
    # 3. 记录更新前的权重 (用于对比)
    # 我们只看 Actor 某一个权重的变化
    before_weight = model.actor_mean.weight.data[0][0].item()
    print(f"3. [进化前] 观察某一个神经元突触")
    print(f"   - 权重值: {before_weight:.6f}")
    
    # 4. 执行 PPO 更新
    print("\n⚡ 正在进行 PPO 核心更新 (反向传播)...")
    ppo.update(rollouts)
    
    # 5. 记录更新后的权重
    after_weight = model.actor_mean.weight.data[0][0].item()
    
    print(f"\n4. [进化后] 神经元突触发生了改变")
    print(f"   - 权重值: {after_weight:.6f}")
    print(f"   - 变化量: {after_weight - before_weight:.6f}")
    print("-" * 50)
    
    print("\n✅ 演示结束！")
    print("这就是“强化学习”的本质：")
    print("AI 根据优势值 (Advantage)，微调每一个神经元的连接权重，")
    print("让“好动作”在未来出现的概率变大，让“坏动作”出现的概率变小。")
    print("-" * 50)
    print("👉 想了解背后的数学原理（梯度、反向传播）？")
    print("   请阅读文档: docs/07_backpropagation_essence.md")

if __name__ == "__main__":
    inspect_ppo_update()
