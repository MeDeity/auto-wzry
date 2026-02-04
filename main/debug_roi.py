import sys
import os

# 将项目根目录添加到 sys.path
# 假设脚本位于 d:\Project\auto-wzry\main\debug_roi.py
# 我们需要添加 d:\Project\auto-wzry
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import logging
from main.env.wzry_env import WZRYEnv
import time

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DebugROI")
    
    logger.info("Initializing environment...")
    env = WZRYEnv(render_mode="rgb_array")
    
    try:
        env.reset()
        logger.info("Taking screenshot...")
        
        # 等待几秒让画面稳定
        time.sleep(2)
        
        # 获取原始屏幕截图 (不是 resize 后的 obs)
        screen = env._get_screen()
        
        if screen is None:
            logger.error("Failed to capture screen.")
            return

        # 在图上画出 ROI
        logger.info("Drawing ROI boxes...")
        debug_img = env.reward_calc.debug_show_roi(screen)
        
        # 保存图片
        output_path = "debug_roi.jpg"
        cv2.imwrite(output_path, debug_img)
        logger.info(f"Debug image saved to {output_path}")
        logger.info("Please open this image to verify if the red box correctly covers the KDA numbers.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    main()