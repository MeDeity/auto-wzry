import os
import sys
import time
import logging
import cv2
import numpy as np
import threading
from pynput import mouse, keyboard
import json
from datetime import datetime

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.env.wzry_env import WZRYEnv

class ExpertRecorder:
    def __init__(self, output_dir="data/expert_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.env = WZRYEnv(render_mode="rgb_array", resolution=(224, 224))
        self.is_recording = False
        self.actions = [] # (timestamp, x, y, type)
        self.frames = []  # (timestamp, image_path)
        
        # 屏幕映射参数
        # 假设我们在电脑上通过 Scrcpy 窗口操作
        # 我们需要知道 Scrcpy 窗口在屏幕上的位置和大小
        # 这里简化处理：假设用户手动输入或通过截图校准
        # 目前先全屏截图或者假设固定位置
        # 为了更准确，我们记录鼠标在 Scrcpy 窗口内的相对坐标
        # 但 pynput 只能获取全局坐标
        # 解决方案：使用 OpenCV 显示窗口并捕获该窗口内的鼠标事件
        
        self.window_name = "Expert Recorder - Press 'R' to Start/Stop, 'Q' to Quit"
        
        # 状态缓存
        self.current_frame = None
        self.last_action_time = 0
        
    def _get_next_episode_idx(self):
        """
        自动探测 data/expert_data 下最大的 episode_idx
        防止覆盖已有数据
        """
        json_files = [f for f in os.listdir(self.output_dir) if f.startswith("episode_") and f.endswith(".json")]
        if not json_files:
            return 0
            
        indices = []
        for f in json_files:
            try:
                # episode_12.json -> 12
                idx = int(f.split("_")[1].split(".")[0])
                indices.append(idx)
            except ValueError:
                continue
                
        if not indices:
            return 0
            
        return max(indices) + 1
        
    def run(self):
        print("Initializing environment...")
        self.env.reset()
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        print("Recorder started.")
        print("1. Click on the window to control (simulated).")
        print("2. Press 'r' to start/stop recording.")
        print("3. Press 'q' to quit.")
        
        # 自动探测下一个 episode_idx
        episode_idx = self._get_next_episode_idx()
        print(f"Next episode index: {episode_idx}")
        
        try:
            while True:
                # 获取当前帧
                frame = self.env._get_screen()
                self.current_frame = frame
                
                # 显示
                display_frame = frame.copy()
                
                if self.is_recording:
                    # 录制状态指示
                    cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
                    
                    # 保存数据
                    timestamp = time.time()
                    
                    # 保存帧 (降频保存，比如 10fps)
                    # 这里简化为每帧都存，实际应用需要控制频率
                    img_filename = f"ep{episode_idx}_{int(timestamp*1000)}.jpg"
                    img_path = os.path.join(self.output_dir, img_filename)
                    
                    # 优化：缩小图片以节省内存 (1920x1080 -> 224x224)
                    # 避免长时间录制导致 MemoryError
                    resized_frame = cv2.resize(frame, self.env.resolution)

                    self.frames.append({
                        "timestamp": timestamp,
                        "img_path": img_filename,
                        "img_data": resized_frame
                    })
                
                cv2.imshow(self.window_name, cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.is_recording = not self.is_recording
                    if self.is_recording:
                        print("Recording started...")
                        self.frames = []
                        self.actions = []
                        episode_idx += 1
                    else:
                        print("Recording stopped. Saving data...")
                        self.save_data(episode_idx)
                        
        finally:
            self.env.close()
            cv2.destroyAllWindows()

    def on_mouse(self, event, x, y, flags, param):
        if not self.is_recording:
            return
            
        timestamp = time.time()
        h, w = self.current_frame.shape[:2]
        
        # 归一化坐标
        norm_x = x / w
        norm_y = y / h
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 点击/按下
            self.actions.append({
                "timestamp": timestamp,
                "type": "down",
                "x": norm_x,
                "y": norm_y
            })
            # 同时发送给手机执行 (可选，为了边玩边录)
            # self.env.minitouch.tap(int(norm_x * self.env.real_width), int(norm_y * self.env.real_height))
            
        elif event == cv2.EVENT_LBUTTONUP:
            # 抬起
            self.actions.append({
                "timestamp": timestamp,
                "type": "up",
                "x": norm_x,
                "y": norm_y
            })
            
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            # 拖拽
            self.actions.append({
                "timestamp": timestamp,
                "type": "move",
                "x": norm_x,
                "y": norm_y
            })

    def save_data(self, episode_idx):
        if not self.frames:
            print("No data recorded.")
            return
            
        print(f"Saving {len(self.frames)} frames and {len(self.actions)} actions...")
        
        try:
            # 保存图片
            for frame_data in self.frames:
                path = os.path.join(self.output_dir, frame_data["img_path"])
                cv2.imwrite(path, frame_data["img_data"])
                
            # 匹配标签 (Frame -> Action)
            # 这是一个关键步骤：每一帧对应什么动作？
            # 简单策略：找到该帧时间戳之后最近的一个动作
            
            # 保存原始数据索引
            index_file = os.path.join(self.output_dir, f"episode_{episode_idx}.json")
            with open(index_file, "w") as f:
                json.dump({
                    "frames": [{"ts": f["timestamp"], "path": f["img_path"]} for f in self.frames],
                    "actions": self.actions
                }, f)
                
            print(f"Data successfully saved to {index_file}")
            
        except Exception as e:
            print(f"Error saving data: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    recorder = ExpertRecorder()
    recorder.run()