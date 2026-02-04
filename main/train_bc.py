import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                with open(os.path.join(data_dir, filename), "r") as f:
                    data = json.load(f)
                    self._process_episode(data)
    
    def _process_episode(self, data):
        frames = data["frames"]
        actions = data["actions"]
        
        # 简单对齐策略：为每一帧找到最近的动作
        # 更好的策略：插值或使用滑动窗口
        
        for frame in frames:
            img_path = os.path.join(self.data_dir, frame["path"])
            ts = frame["ts"]
            
            # 找到 ts 之后最近的动作 (简单预测下一步操作)
            # 这里简化处理：如果没有动作，则设为 [0, 0, 0, 0]
            # 实际需要根据项目需求定义 Expert Action 格式
            
            # 示例：寻找最近的 'down' 或 'move'
            target_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
            # ... (数据清洗逻辑)
            
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
    for epoch in range(epochs):
        total_loss = 0
        for imgs, actions in dataloader:
            imgs = imgs.to(device)
            actions = actions.to(device)
            
            optimizer.zero_grad()
            pred_actions = model(imgs)
            
            loss = criterion(pred_actions, actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # 保存
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"models/bc_model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_bc()