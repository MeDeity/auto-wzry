import os
import sys

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
import argparse
import torch
import cv2
import numpy as np
from main.env.wzry_env import WZRYEnv
from main.model import ActorCritic
from main.algo.ppo import PPO, RolloutBuffer

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO Agent for WZRY")
    parser.add_argument("--pretrained-path", type=str, default=None, help="Path to pretrained BC model (e.g., models/bc_model_epoch_50.pth)")
    parser.add_argument("--resume-path", type=str, default=None, help="Path to existing PPO model to resume training (e.g., models/ppo_wzry_epoch_10.pth)")
    parser.add_argument("--max-timesteps", type=int, default=100000, help="Maximum training timesteps")
    parser.add_argument("--render", action="store_true", default=True, help="Render the environment during training")
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
                    
                    # 检查形状是否匹配
                    # PPO Actor mean: [3, 512]
                    
                    if v.shape[0] == 4 and action_dim == 3:
                        logger.warning(f"Detected 4D weights (Swipe) for 3D environment (Continuous). "
                                       f"This BC model is incompatible. Slicing anyway but performance will be poor.")
                        new_state_dict[new_key] = v[:3] # 强行切片，但这会导致 press_prob (Dim 2) 被映射为 x2
                    elif v.shape[0] == 3 and action_dim == 3:
                        # 完美匹配
                        new_state_dict[new_key] = v
                    else:
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
    
    # Visualization State
    last_info_text = ""
    last_viz_pos = None # 上一帧的坐标 (x, y) 用于绘制轨迹

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
                
                # Visualization
                if args.render:
                    try:
                        raw_frame = env.render()
                        if raw_frame is not None:
                            # Convert to BGR for OpenCV
                            display_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                            
                            # Draw ROI (Debug)
                            if hasattr(env, 'reward_calc'):
                                display_frame = env.reward_calc.debug_show_roi(display_frame)

                            h, w = display_frame.shape[:2]
                            
                            # Action: [x, y, press_prob]
                            pred_x, pred_y, pred_prob = action_env
                            
                            # Draw Crosshair
                            center_x = int(pred_x * w)
                            center_y = int(pred_y * h)
                            
                            is_pressed = pred_prob > 0.5
                            color = (0, 0, 255) if is_pressed else (0, 255, 0) # Red if press, Green if release
                            
                            # Determine Status Text (DOWN / UP / MOVE)
                            status_text = "UP"
                            if is_pressed:
                                if last_viz_pos is not None:
                                    # 如果上一帧也是按下，且距离超过一定阈值，则视为 MOVE
                                    lx, ly = last_viz_pos
                                    dist = ((center_x - lx)**2 + (center_y - ly)**2)**0.5
                                    if dist > 5:
                                        status_text = "MOVE"
                                        # 画轨迹线
                                        cv2.line(display_frame, (lx, ly), (center_x, center_y), (0, 255, 255), 2)
                                    else:
                                        status_text = "DOWN"
                                else:
                                    status_text = "DOWN"
                                # 更新上一帧位置 (仅在按下时记录)
                                last_viz_pos = (center_x, center_y)
                            else:
                                last_viz_pos = None # 抬起后重置轨迹
                                
                            status_text += f" ({pred_prob:.2f})"
                            
                            cv2.circle(display_frame, (center_x, center_y), 20, color, 2)
                            cv2.line(display_frame, (center_x - 30, center_y), (center_x + 30, center_y), color, 2)
                            cv2.line(display_frame, (center_x, center_y - 30), (center_x, center_y + 30), color, 2)
                            
                            # Draw Status Text
                            cv2.putText(display_frame, f"Action: {status_text}", (10, 80), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            
                            # Draw Reward
                            cv2.putText(display_frame, f"Reward: {reward:.4f}", (10, 120),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                            
                            # Update Info Text if new info available
                            if info:
                                new_text_parts = []
                                if 'gold' in info:
                                    new_text_parts.append(f"Gold: {info['gold']}")
                                if 'kda' in info:
                                    new_text_parts.append(f"KDA: {info['kda']}")
                                if new_text_parts:
                                    last_info_text = " | ".join(new_text_parts)
                                    
                            # Draw Info Text
                            if last_info_text:
                                cv2.putText(display_frame, last_info_text, (10, 160), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                            cv2.imshow("PPO Training Monitor", display_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                logger.info("Visualization stopped by user.")
                                args.render = False
                                cv2.destroyAllWindows()
                    except Exception as e:
                        logger.warning(f"Visualization error: {e}")
                
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
        if args.render:
            cv2.destroyAllWindows()
        logger.info("Environment closed.")

if __name__ == "__main__":
    train()