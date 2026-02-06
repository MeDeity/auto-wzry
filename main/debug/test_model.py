
import os
import sys
import time
import cv2
import torch
import numpy as np
import argparse

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main.env.wzry_env import WZRYEnv
from main.train_bc import BCModel
from main.env.image_utils import ImageProcessor

def test_model():
    parser = argparse.ArgumentParser(description="Test BC Model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model .pth file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found.")
        return

    # 1. 加载模型
    print(f"Loading model from {args.model_path} on {args.device}...")
    model = BCModel().to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    # 2. 初始化环境和处理器
    env = WZRYEnv(render_mode="rgb_array", resolution=(224, 224))
    processor = ImageProcessor()
    
    window_name = f"Model Test: {os.path.basename(args.model_path)}"
    cv2.namedWindow(window_name)
    
    print("\nStarting Model Test...")
    print("--------------------------------------------------")
    print("Press 'q' to quit.")
    print("Press 'a' to toggle Auto-Pilot (Real Control). DEFAULT: OFF")
    print("--------------------------------------------------")
    
    auto_pilot = False
    is_touching = False # 跟踪触摸状态，防止重复发送 down/up
    
    try:
        with torch.no_grad():
            while True:
                # 1. 获取画面
                raw_frame = env._get_screen() # 原始分辨率
                
                # 2. 预处理
                # 注意：这里需要和训练时一致的缩放和归一化
                # env.resolution 是 (224, 224)
                # train_bc.py 里是 cv2.imread -> processor.preprocess -> processor.to_tensor
                # processor.preprocess 做了 resize(224, 224) 和 normalize
                
                processed_img = processor.preprocess(raw_frame)
                img_tensor = processor.to_tensor(processed_img).unsqueeze(0).to(args.device)
                
                # 3. 推理
                # output: [x, y, is_active, reserved]
                preds = model(img_tensor).cpu().numpy()[0]
                
                pred_x, pred_y = preds[0], preds[1]
                pred_active = preds[2]
                
                # 4. 可视化
                display_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                h, w = display_frame.shape[:2]
                
                # 还原坐标
                center_x = int(pred_x * w)
                center_y = int(pred_y * h)
                
                # 绘制预测点
                # 红色 = 按下, 绿色 = 抬起
                color = (0, 0, 255) if pred_active > 0.5 else (0, 255, 0)
                status_text = "DOWN" if pred_active > 0.5 else "UP"
                
                # 画十字准星
                cv2.circle(display_frame, (center_x, center_y), 20, color, 2)
                cv2.line(display_frame, (center_x - 30, center_y), (center_x + 30, center_y), color, 2)
                cv2.line(display_frame, (center_x, center_y - 30), (center_x, center_y + 30), color, 2)
                
                # 绘制状态文本
                cv2.putText(display_frame, f"Action: {status_text} ({pred_active:.2f})", (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # 绘制 Auto-Pilot 状态
                mode_color = (0, 255, 255) if auto_pilot else (200, 200, 200)
                mode_text = "AUTO-PILOT: ON" if auto_pilot else "AUTO-PILOT: OFF (Observation Only)"
                cv2.putText(display_frame, mode_text, (30, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

                cv2.imshow(window_name, display_frame)
                
                # 5. 执行动作 (如果开启)
                if auto_pilot and env.minitouch:
                    # 映射回真实分辨率 (使用 transform_touch 处理横竖屏转换)
                    real_x, real_y = env.transform_touch(pred_x, pred_y)
                    
                    if pred_active > 0.5:
                        if not is_touching:
                            env.minitouch.touch_down(real_x, real_y)
                            is_touching = True
                        else:
                            env.minitouch.touch_move(real_x, real_y)
                    else:
                        if is_touching:
                            env.minitouch.touch_up()
                            is_touching = False
                elif not auto_pilot and env.minitouch:
                    # 如果关闭了 Auto-Pilot 且之前还在触摸，则强制抬起
                    if is_touching:
                        env.minitouch.touch_up()
                        is_touching = False

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    auto_pilot = not auto_pilot
                    print(f"Auto-Pilot toggled: {auto_pilot}")
                    
    finally:
        cv2.destroyAllWindows()
        env.close()

if __name__ == "__main__":
    test_model()
