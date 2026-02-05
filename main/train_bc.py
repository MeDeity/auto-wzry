import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import numpy as np

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.model import CNNEncoder
from main.env.image_utils import ImageProcessor

class ExpertDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.processor = ImageProcessor()
        
        # 加载所有 episode 数据
        json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        if not json_files:
            print(f"Warning: No .json files found in {data_dir}. Did you run record_expert.py and save successfully?")
            
        for filename in json_files:
            with open(os.path.join(data_dir, filename), "r") as f:
                data = json.load(f)
                self._process_episode(data)
    
    def _process_episode(self, data):
        frames = sorted(data["frames"], key=lambda x: x["ts"])
        actions = sorted(data["actions"], key=lambda x: x["timestamp"])
        
        # 状态机处理：按时间顺序遍历，维护当前的鼠标/触摸状态
        action_idx = 0
        
        # 初始状态 (假设未按下，坐标在中心)
        is_pressed = False
        last_x, last_y = 0.5, 0.5 
        
        for frame in frames:
            ts = frame["ts"]
            
            # 更新状态：处理所有发生在该帧之前(或刚好同时)的动作
            while action_idx < len(actions) and actions[action_idx]["timestamp"] <= ts:
                act = actions[action_idx]
                if act["type"] == "down" or act["type"] == "move":
                    is_pressed = True
                    last_x = act["x"]
                    last_y = act["y"]
                elif act["type"] == "up":
                    is_pressed = False
                    last_x = act["x"]
                    last_y = act["y"]
                action_idx += 1
            
            # 根据当前状态构造 Label
            # 格式: [x, y, is_active, reserved]
            if is_pressed:
                target_action = np.array([last_x, last_y, 1.0, 0.0], dtype=np.float32)
            else:
                # 未按下时，目标设为中心点，且 active=0
                target_action = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
            
            img_path = os.path.join(self.data_dir, frame["path"])
            self.samples.append((img_path, target_action))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, action = self.samples[idx]
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            # 容错
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            
        # 预处理
        processed = self.processor.preprocess(img)
        img_tensor = self.processor.to_tensor(processed)
        
        action_tensor = torch.from_numpy(action)
        
        return img_tensor, action_tensor

class BCModel(nn.Module):
    """
    行为克隆模型 (Behavior Cloning)
    结构比 RL 模型简单，只需要 Actor 部分
    """
    def __init__(self, action_dim=4):
        super(BCModel, self).__init__()
        self.encoder = CNNEncoder(output_dim=512)
        self.fc = nn.Linear(512, action_dim)
        
    def forward(self, x):
        feat = self.encoder(x)
        return torch.sigmoid(self.fc(feat)) # 输出 0-1 之间的坐标

def train_bc():
    data_dir = "data/expert_data"
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 数据集
    dataset = ExpertDataset(data_dir)
    if len(dataset) == 0:
        print("No samples found.")
        return
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. 模型
    model = BCModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss() # 预测坐标的均方误差
    
    # 3. 训练循环
    epochs = 50
    print(f"Start training for {epochs} epochs...", flush=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 使用 tqdm 显示进度条，强制输出到 stdout
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", file=sys.stdout)
        
        for imgs, actions in pbar:
            imgs = imgs.to(device)
            actions = actions.to(device)
            
            optimizer.zero_grad()
            pred_actions = model(imgs)
            
            loss = criterion(pred_actions, actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 实时更新进度条后缀显示当前 Loss
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} Average Loss: {avg_loss:.6f}", flush=True)
        
        os.makedirs("models", exist_ok=True)
        
        # 保存
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"models/bc_model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_bc()