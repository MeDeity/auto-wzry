import logging
import time
import sys
import os

# 将项目根目录添加到 python path，以便能导入 main 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main.env.adb_wrapper import AdbWrapper
from main.env.scrcpy_wrapper import ScrcpyWrapper
from main.env.window_capture import WindowCapture
from main.env.image_utils import ImageProcessor
from main.env.types import EnvState, ActionType, EnvAction
# from main.env.minitouch_wrapper import MinitouchWrapper # 暂时注释，避免没有设备时报错
try:
    from main.env.wzry_env import WZRYEnv
    import torch
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("Gymnasium or Torch not installed, skipping Gym env test.")

def test_gym_environment():
    if not GYM_AVAILABLE:
        return
        
    logger = logging.getLogger("TestGymEnv")
    logger.info("=== Testing WZRY Gym Environment ===")
    
    try:
        # 初始化环境
        env = WZRYEnv(render_mode="rgb_array", resolution=(224, 224))
        
        # Reset
        logger.info("Resetting environment...")
        obs, info = env.reset()
        logger.info(f"Observation shape: {obs.shape}, range: [{obs.min():.2f}, {obs.max():.2f}]")
        
        # ToTensor 演示
        logger.info("Converting to Tensor...")
        # obs 已经是 CHW 格式，所以设置 input_chw=True
        obs_tensor = ImageProcessor.to_tensor(obs, input_chw=True) 
        logger.info(f"Tensor shape: {obs_tensor.shape}, Device: {obs_tensor.device}")
    except Exception as e:
        logger.error(f"Gym Env Test Failed: {e}")
    finally:
        if 'env' in locals():
            env.close()


def test_environment():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("TestEnv")

    logger.info("=== Starting Environment Test ===")

    # 1. Test ADB
    logger.info("--- Testing ADB Wrapper ---")
    adb = AdbWrapper()
    try:
        devices = adb.get_devices()
        logger.info(f"Connected devices: {devices}")
        
        if not devices:
            logger.warning("No devices connected. Skipping Scrcpy and Minitouch tests.")
            return
        
        device_id = devices[0]
        logger.info(f"Target Device: {device_id}")
    except Exception as e:
        logger.error(f"ADB Test Failed: {e}")
        return

    # 2. Test Scrcpy
    logger.info("--- Testing Scrcpy Wrapper ---")
    window_title = f"Auto-WZRY-Test-{int(time.time())}"
    scrcpy = ScrcpyWrapper(device_id=device_id, max_size=800)
    
    try:
        scrcpy.start(window_title=window_title)
        logger.info("Waiting for Scrcpy window to appear...")
        time.sleep(3) # Give it some time to launch
    except Exception as e:
        logger.error(f"Scrcpy Test Failed: {e}")
        return

    # 3. Test Window Capture
    logger.info("--- Testing Window Capture ---")
    latest_frame = None
    try:
        cap = WindowCapture(window_title)
        
        # Try to capture a few frames
        for i in range(5):
            frame = cap.capture()
            if frame is not None:
                logger.info(f"Frame {i} captured successfully. Shape: {frame.shape}")
                latest_frame = frame
            else:
                logger.warning(f"Frame {i} capture failed (Window not found yet?)")
            time.sleep(1)

        # 4. Test Image Processing & Types
        if latest_frame is not None:
            logger.info("--- Testing Image Processing & Types ---")
            processor = ImageProcessor()
            processed = processor.preprocess(latest_frame)
            logger.info(f"Processed frame shape: {processed.shape}, Type: {processed.dtype}, Range: [{processed.min():.2f}, {processed.max():.2f}]")
            
            state = EnvState(raw_screen=latest_frame, processed_screen=processed)
            logger.info(f"EnvState created successfully.")
            
            action = EnvAction(action_type=ActionType.TAP, start_pos=(0.5, 0.5))
            logger.info(f"EnvAction sample created: {action}")
            
    except Exception as e:
        logger.error(f"Window Capture/Processing Test Failed: {e}")
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        scrcpy.stop()
        logger.info("Test Finished.")

if __name__ == "__main__":
    test_environment()
    # test_gym_environment() # Uncomment to test gym env
