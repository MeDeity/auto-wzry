import sys
import os
import time
import logging

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main.env.minitouch_wrapper import MinitouchWrapper
from main.env.adb_wrapper import AdbWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_minitouch():
    adb = AdbWrapper()
    devices = adb.get_devices()
    if not devices:
        logger.error("No devices found")
        return

    device_id = devices[0]
    logger.info(f"Testing on device: {device_id}")

    mt = MinitouchWrapper(device_id)
    
    try:
        mt.start()
        time.sleep(2)
        
        if not mt._is_connected or not mt.device:
            logger.error("Failed to connect")
            return

        # 尝试获取 bounds
        # pyminitouch 的 MNTDevice 对象通常会解析 header
        # 我们可以查看它的属性
        logger.info(f"Minitouch Connection: {mt.device.connection}")
        # 尝试访问 max_x, max_y (这取决于 pyminitouch 的具体实现，我们先盲猜一下或者打印 dir)
        logger.info(f"Device attributes: {dir(mt.device)}")
        
        # 发送一个点击测试 (屏幕中心)
        # 假设我们不知道分辨率，先发个小的
        logger.info("Sending touch_down at 500, 500")
        mt.touch_down(500, 500)
        time.sleep(0.1)
        mt.touch_up()
        
        logger.info("Test complete")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        mt.stop()

if __name__ == "__main__":
    test_minitouch()
