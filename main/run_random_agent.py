import time
import logging
import numpy as np
import sys
import os

# 将项目根目录添加到 python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main.env.wzry_env import WZRYEnv

def run_random_agent(max_steps=100):
    """
    运行随机 Agent 以测试环境稳定性
    """
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("RandomAgent")
    
    logger.info("Initializing Environment...")
    env = None
    try:
        # 初始化环境
        env = WZRYEnv(render_mode="rgb_array", resolution=(224, 224))
        
        # Reset 环境
        obs, info = env.reset()
        logger.info(f"Environment reset complete. Obs shape: {obs.shape}")
        
        # 性能统计变量
        total_time = 0
        step_times = []
        start_time = time.time()
        
        logger.info(f"Starting loop for {max_steps} steps...")
        
        for step in range(max_steps):
            step_start = time.time()
            
            # 1. 随机采样动作
            # WZRYEnv Action: [x1, y1, x2, y2] (0.0 - 1.0)
            action = env.action_space.sample()
            
            # 2. 执行动作并获取新的观察
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 3. 统计时间
            step_end = time.time()
            duration = step_end - step_start
            step_times.append(duration)
            total_time += duration
            
            # 每 10 步打印一次日志
            if (step + 1) % 10 == 0:
                avg_step_time = np.mean(step_times[-10:])
                fps = 1.0 / avg_step_time
                logger.info(f"Step {step+1}/{max_steps} | Action: {action[:2]}... | Latency: {duration*1000:.1f}ms | FPS: {fps:.1f}")
                
            if terminated or truncated:
                logger.info("Episode finished.")
                obs, info = env.reset()
                
        # 总结合
        avg_latency = np.mean(step_times) * 1000
        avg_fps = 1.0 / np.mean(step_times)
        logger.info("=== Test Finished ===")
        logger.info(f"Total Steps: {max_steps}")
        logger.info(f"Average Latency: {avg_latency:.2f} ms")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info("Environment stability test passed.")

    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
    finally:
        if env:
            logger.info("Closing environment...")
            env.close()

if __name__ == "__main__":
    # 可以通过命令行参数控制步数，这里默认 100 步
    run_random_agent(max_steps=100)
