import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class PPO:
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
        PPO 更新逻辑
        rollouts: dict containing tensors on device
            - states
            - actions
            - log_probs
            - returns
            - advantages
            - values
        """
        states = rollouts['states']
        actions = rollouts['actions']
        old_log_probs = rollouts['log_probs']
        returns = rollouts['returns']
        advantages = rollouts['advantages']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.ppo_epochs):
            for batch in loader:
                b_states, b_actions, b_old_log_probs, b_returns, b_advantages = batch
                
                # Evaluate current policy
                new_log_probs, values, entropy = self.policy.evaluate(b_states, b_actions)
                
                # Ratio
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                
                # Surrogate Loss
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = (b_returns - values).pow(2).mean()
                
                # Entropy Loss
                entropy_loss = -entropy.mean()
                
                # Total Loss
                loss = actor_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def compute_returns_and_advantages(self, next_value, gamma, gae_lambda, device):
        """
        计算 GAE 和 Returns
        """
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device).unsqueeze(1)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device).unsqueeze(1)
        values = torch.cat(self.values + [next_value])
        
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            returns.insert(0, gae + values[i])
            
        returns = torch.cat(returns).detach()
        advantages = returns - values[:-1].detach()
        
        return {
            'states': torch.cat(self.states).detach(),
            'actions': torch.cat(self.actions).detach(),
            'log_probs': torch.cat(self.log_probs).detach(),
            'returns': returns,
            'advantages': advantages,
            'values': values[:-1].detach()
        }
