import os
import sys

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
import argparse
import torch
import numpy as np
from main.env.wzry_env import WZRYEnv
from main.model import ActorCritic
from main.algo.ppo import PPO, RolloutBuffer

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO Agent for WZRY")
    parser.add_argument("--pretrained-path", type=str, default=None, help="Path to pretrained BC model (e.g., models/bc_model_epoch_50.pth)")
    parser.add_argument("--resume-path", type=str, default=None, help="Path to existing PPO model to resume training (e.g., models/ppo_wzry_epoch_10.pth)")
    parser.add_argument("--max-timesteps", type=int, default=100000, help="Maximum training timesteps")
    return parser.parse_args()

def train():
    args = parse_args()
    
    # 1. 配置
    log_dir = "logs"
    model_dir = "models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # ... (logging config) ...
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
    
    start_epoch = 0

    # 优先加载断点续训模型
    if args.resume_path and os.path.exists(args.resume_path):
        logger.info(f"Resuming training from {args.resume_path}...")
        policy.load_state_dict(torch.load(args.resume_path, map_location=device))
        
        # 尝试从文件名解析 epoch (例如 ppo_wzry_epoch_10.pth -> 10)
        try:
            filename = os.path.basename(args.resume_path)
            if "epoch_" in filename:
                start_epoch = int(filename.split("epoch_")[1].split(".")[0])
                logger.info(f"Resumed from epoch {start_epoch}")
        except:
            logger.warning("Could not parse epoch from filename, starting from epoch 0")
            
    # 如果没有断点，则加载 BC 预训练权重 (如果提供)
    elif args.pretrained_path:
        if os.path.exists(args.pretrained_path):
            logger.info(f"Loading pretrained BC model from {args.pretrained_path}...")
            checkpoint = torch.load(args.pretrained_path, map_location=device)
            
            # 适配 BCModel -> ActorCritic 的权重名称
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('encoder.'):
                    new_state_dict[k] = v
                elif k.startswith('fc.'):
                    new_key = k.replace('fc.', 'actor_mean.')
                    new_state_dict[new_key] = v
            
            msg = policy.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Loaded pretrained BC weights. Missing keys: {msg.missing_keys}")
        else:
            logger.warning(f"Pretrained path {args.pretrained_path} does not exist. Starting from scratch.")
    
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
    epoch = start_epoch
    
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