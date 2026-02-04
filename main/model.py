import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

class CNNEncoder(nn.Module):
    """
    使用 ResNet18 作为特征提取器
    Input: (B, 3, 224, 224)
    Output: (B, feature_dim)
    """
    def __init__(self, output_dim=512):
        super(CNNEncoder, self).__init__()
        
        # 加载 ResNet18 结构 (不加载预训练权重，因为我们是玩游戏，不是分类 ImageNet)
        # 如果有 ImageNet 预训练权重，收敛会更快，但需要联网下载
        # 这里我们选择 weights=None，完全从头训练
        resnet = models.resnet18(weights=None)
        
        # 去掉最后的全连接层 (fc)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet18 的输出维度是 512
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.features(x) # -> (B, 512, 1, 1)
        x = x.view(x.size(0), -1) # -> (B, 512)
        x = F.relu(self.fc(x))
        return x

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(ActorCritic, self).__init__()
        
        # 共享特征提取器
        self.encoder = CNNEncoder(output_dim=hidden_dim)
        
        # Actor: 输出动作均值 (Mean)
        # 我们使用 Tanh 将输出限制在 [-1, 1]，然后在环境交互时映射到 [0, 1]
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        
        # Actor: 动作标准差 (Log Std)，作为一个可学习参数
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic: 输出状态价值 (Value)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.encoder(state)
        
        # Actor
        mean = torch.tanh(self.actor_mean(features))
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        
        # Critic
        value = self.critic(features)
        
        return mean, std, value

    def get_action(self, state):
        """
        采样动作
        """
        mean, std, value = self(state)
        dist = torch.distributions.Normal(mean, std)
        
        # 采样动作 (带有梯度)
        action = dist.sample()
        
        # 计算 log_prob
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value

    def evaluate(self, state, action):
        """
        评估动作 (用于 PPO 更新)
        """
        mean, std, value = self(state)
        dist = torch.distributions.Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy
