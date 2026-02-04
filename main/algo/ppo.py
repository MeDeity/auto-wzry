import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class PPO:
    """
    PPO (Proximal Policy Optimization) 算法实现
    
    PPO 是一种"在线"强化学习算法，通过"裁剪"（Clip）操作限制策略更新的幅度，
    保证训练的稳定性。它是目前最流行的强化学习算法之一。
    """
    def __init__(self, 
                 actor_critic, 
                 lr=3e-4, 
                 gamma=0.99, 
                 gae_lambda=0.95, 
                 clip_param=0.2, 
                 value_loss_coef=0.5, 
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 ppo_epochs=10,
                 batch_size=64,
                 device='cpu'):
        """
        初始化 PPO 算法
        
        参数详解:
        :param actor_critic: 我们的"大脑"模型（包含 Actor 和 Critic 网络）
        :param lr: 学习率 (Learning Rate)，决定每次自我修正的步长大小。
                   太大会导致学过头（震荡），太小会导致学得慢。
        :param gamma: 折扣因子 (Discount Factor)，范围 [0, 1]。
                      gamma 越大，AI 越重视未来的长远收益（高瞻远瞩）；
                      gamma 越小，AI 越只看重眼前的即时收益（目光短浅）。
        :param gae_lambda: GAE (Generalized Advantage Estimation) 的平滑因子，范围 [0, 1]。
                           用于平衡方差和偏差，通常设为 0.95。
        :param clip_param: PPO 的核心参数，裁剪范围（通常为 0.1 或 0.2）。
                           它限制了新旧策略之间的差异，防止一次更新改动太大导致模型"崩溃"。
                           例如 0.2 表示策略更新幅度限制在 [0.8, 1.2] 倍之间。
        :param value_loss_coef: 价值损失（Critic Loss）的权重系数。
                                用来平衡 Policy Loss 和 Value Loss 的大小。
        :param entropy_coef: 熵系数（Entropy Coefficient）。
                             "熵"代表随机性。这个系数鼓励 AI 保持一定的探索性，
                             防止它过早"固步自封"（陷入局部最优）。
        :param max_grad_norm: 梯度裁剪阈值，防止梯度爆炸（更新过猛）。
        :param ppo_epochs: 每次收集完数据后，利用这些数据训练多少轮。
        :param batch_size: 每次训练喂给模型的数据量大小。
        :param device: 运行设备 ('cpu' 或 'cuda')。
        """
        
        self.policy = actor_critic.to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device

    def update(self, rollouts):
        """
        PPO 的核心更新逻辑：这也是 AI "反思"和"进化"的过程
        
        参数:
        :param rollouts: 一个字典，包含了一段时间内收集的所有数据（经验），都在 device 上
            - states: 看到的状态
            - actions: 做出的动作
            - log_probs: 当时做这个动作的概率（对数）
            - returns: 实际获得的回报（目标价值）
            - advantages: 优势值（这个动作比平均水平好多少）
            - values: Critic 预测的价值
        """
        states = rollouts['states']
        actions = rollouts['actions']
        old_log_probs = rollouts['log_probs']
        returns = rollouts['returns']
        advantages = rollouts['advantages']
        
        # 1. 优势值归一化 (Advantage Normalization)
        # 这是一个常用的 Trick，把优势值变成均值为0，方差为1。
        # 作用：让训练更加稳定，收敛更快。
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 构建数据集，准备分批训练
        dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 2. PPO 更新循环
        # 我们会拿着这一批经验反复学习几次 (ppo_epochs)
        for _ in range(self.ppo_epochs):
            for batch in loader:
                b_states, b_actions, b_old_log_probs, b_returns, b_advantages = batch
                
                # --- 评估当前策略 ---
                # 让现在的模型再看一遍当时的状态，看看它现在会给出什么概率和价值预测
                # new_log_probs: 新的动作概率
                # values: 新的价值预测
                # entropy: 策略的随机程度（熵）
                new_log_probs, values, entropy = self.policy.evaluate(b_states, b_actions)
                
                # --- 计算重要性采样比率 (Ratio) ---
                # ratio = P_new(action|state) / P_old(action|state)
                # 如果 ratio > 1，说明新策略更倾向于这个动作；
                # 如果 ratio < 1，说明新策略不想做这个动作。
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                
                # --- 计算 Surrogate Loss (代理损失) ---
                # surr1: 无约束的更新目标
                # surr2: 裁剪后的更新目标 (PPO 的核心魔法)
                # 只有当 ratio 超出 [1-clip, 1+clip] 范围时，我们才停止激励它继续变大/变小。
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * b_advantages
                
                # 最终的 Actor Loss 是取两者的最小值（悲观估计），我们要最小化 Loss，所以加负号
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # --- 计算 Value Loss (价值损失) ---
                # Critic 的任务是预测越准越好，所以是均方误差 (MSE)
                value_loss = (b_returns - values).pow(2).mean()
                
                # --- 计算 Entropy Loss (熵损失) ---
                # 我们希望熵越大越好（保持探索），所以 Loss = -entropy
                entropy_loss = -entropy.mean()
                
                # --- 总损失 ---
                # Total Loss = Actor Loss + c1 * Value Loss + c2 * Entropy Loss
                loss = actor_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # --- 反向传播更新 ---
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪：防止因为某个数据异常导致梯度爆炸，毁掉模型
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

class RolloutBuffer:
    """
    经验笔记本 (Rollout Buffer)
    
    作用：
    AI 在玩游戏时，会把它的所见(State)、所做(Action)、所得(Reward)
    都记录在这个本子上。等存满一定数量后，PPO 就会拿这个本子去学习。
    学习完后，这个本子就会被清空，准备记录下一轮的经验。
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, state, action, reward, done, log_prob, value):
        """记录单步数据"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
    def clear(self):
        """清空笔记本"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def compute_returns_and_advantages(self, next_value, gamma, gae_lambda, device):
        """
        计算回报 (Returns) 和 优势 (Advantages) - 使用 GAE 算法
        
        GAE (Generalized Advantage Estimation) 是一种非常强大的技术，
        它可以更准确地评估一个动作到底好不好。
        
        参数:
        :param next_value: 最后一个状态的价值估计（用于处理未结束的轨迹）
        """
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device).unsqueeze(1)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device).unsqueeze(1)
        # 把最后时刻的价值拼接到 values 列表后面，方便统一计算
        values = torch.cat(self.values + [next_value])
        
        returns = []
        gae = 0
        # 从后往前逆序计算
        for i in reversed(range(len(rewards))):
            # delta (TD Error): 真实发生的惊喜 = 即时奖励 + 下一刻价值 - 这一刻预估价值
            # 如果 delta > 0，说明情况比预期的好；delta < 0，说明比预期的差。
            delta = rewards[i] + gamma * values[i+1] * (1 - dones[i]) - values[i]
            
            # GAE 公式：当前优势 = 当前惊喜 + 折扣后的未来优势
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            
            # Return = Advantage + Value
            returns.insert(0, gae + values[i])
            
        returns = torch.cat(returns).detach()
        # Advantage = Return - Value
        advantages = returns - values[:-1].detach()
        
        return {
            'states': torch.cat(self.states).detach(),
            'actions': torch.cat(self.actions).detach(),
            'log_probs': torch.cat(self.log_probs).detach(),
            'returns': returns,
            'advantages': advantages,
            'values': values[:-1].detach()
        }
