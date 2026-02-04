import os
import sys

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
import torch
import numpy as np
from main.env.wzry_env import WZRYEnv
from main.model import ActorCritic
from main.algo.ppo import PPO, RolloutBuffer

def train():
    # 1. 配置
    log_dir = "logs"
    model_dir = "models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("Train")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. 初始化环境
    logger.info("Initializing environment...")
    env = WZRYEnv(render_mode="rgb_array", resolution=(224, 224))
    
    # 3. 初始化模型
    state_dim = env.observation_space.shape # (3, 224, 224)
    action_dim = env.action_space.shape[0]  # 4
    
    policy = ActorCritic(state_dim, action_dim).to(device)
    
    # 4. 初始化算法
    ppo_agent = PPO(
        actor_critic=policy,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        ppo_epochs=10,
        batch_size=64,
        device=device
    )
    
    buffer = RolloutBuffer()
    
    # 训练参数
    max_timesteps = 100000
    steps_per_epoch = 2048 # 每次更新前收集的步数
    
    state, _ = env.reset()
    # numpy (C,H,W) -> tensor (1,C,H,W)
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
    
    total_steps = 0
    epoch = 0
    
    logger.info("Starting training...")
    
    try:
        while total_steps < max_timesteps:
            # === Collection Phase ===
            epoch_rewards = []
            
            for t in range(steps_per_epoch):
                # 采样动作
                with torch.no_grad():
                    action, log_prob, value = policy.get_action(state_tensor)
                
                # Tensor -> Numpy
                action_np = action.cpu().numpy()[0]
                # Clip action to [0, 1] for environment
                action_env = np.clip(action_np, 0.0, 1.0)
                
                # 环境交互
                next_state, reward, terminated, truncated, info = env.step(action_env)
                done = terminated or truncated
                
                # 存储经验
                buffer.add(state_tensor, action, reward, done, log_prob, value)
                
                state = next_state
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                
                epoch_rewards.append(reward)
                total_steps += 1
                
                if (t+1) % 100 == 0:
                    logger.info(f"Step {t+1}/{steps_per_epoch} (Total: {total_steps}) | Last Reward: {reward:.4f}")
                
                if done:
                    state, _ = env.reset()
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            
            # === Update Phase ===
            logger.info(f"Epoch {epoch+1} finished. Avg Reward: {np.mean(epoch_rewards):.4f}")
            
            # 计算下一个状态的 Value (用于 GAE)
            with torch.no_grad():
                _, _, next_value = policy(state_tensor)
            
            # 计算 Returns & Advantages
            rollouts = buffer.compute_returns_and_advantages(next_value, ppo_agent.gamma, ppo_agent.gae_lambda, device)
            
            # PPO 更新
            logger.info("Updating policy...")
            ppo_agent.update(rollouts)
            
            # 清空 Buffer
            buffer.clear()
            
            # 保存模型
            if (epoch + 1) % 1 == 0: # 每个 Epoch 保存一次
                save_path = os.path.join(model_dir, f"ppo_wzry_epoch_{epoch+1}.pth")
                torch.save(policy.state_dict(), save_path)
                logger.info(f"Model saved to {save_path}")
            
            epoch += 1

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
    finally:
        env.close()
        logger.info("Environment closed.")

if __name__ == "__main__":
    train()