import torch
import torch.nn as nn
from torch.utils.data import DataLoader , random_split
from torchvision import models , transforms
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os 

print("=" *20 + "Day12:寻找最佳模型(chechpoint)" + "="*20)

# ================0. 准备环境（复用之前的逻辑）==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================ 1. 极简 Dataset===========
# 为了方便，我们这里直接创建一个临时的Dataset类
class MockDataset(Dataset):
    def __init__(self,length = 100):
        self.length = length
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        # 模拟数据
        img = torch.randn(3,224,224) # 假装已经transform过了
        score = torch.tensor(np.random.uniform(1,5),dtype = torch.float32)
        return img ,score
    
# ===============2. 数据切分(Trian/Val Split)=============
full_dataset = MockDataset(length=100)

#80%用于训练，20%用于验证
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# random_split 是PyTorch提供的切分工具
train_dataset , val_dataset = random_split(full_dataset,[train_size,val_size])

train_loader  = DataLoader(train_dataset,batch_size = 16,shuffle = True)
val_loader = DataLoader(val_dataset,batch_size = 16,shuffle = False)  # 验证集不需要shuffle

print(f"训练集数量：{len(train_dataset)},验证集数量：{len(val_dataset)}")

# ===============3. 模型准备=============
model = models.resnet18(weights = 'DEFAULT')
model.fc = nn.Linear(512,1)  # 修改Head
model = model.to(device)

criterion =  nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)

# ===============4. 核心，训练于验证循环 ==============
# 初始化“历史最高分”（如果是loss，初始化为无穷大；如果是SRCC，初始化为-1）
best_val_loss = float('inf')

for epoch in range(10):  #跑十轮
    print(f"\n--- Epoch{epoch +1}/10---")
    
    # ===A.训练阶段（Train) === 
    model.train()  # 开启训练模式（启用Dropout/BatchNorm)
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs , labels = imgs.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs,labels.view(-1,1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss= train_loss / len(train_loader)

    # ==== B. 验证阶段（Eval)=======
    model.eval()  # 开启评估模式（冻结Dropout/BatchNorm)
    val_loss = 0.0

    # 验证不需要梯度，必须关掉
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs,labels.view(-1,1))
            val_loss += loss.item()

        avg_val_loss = val_loss/len(val_loader)

        print(f"Train loss:{avg_train_loss:.4f},| Val loss:{avg_val_loss:.4f}")

    # =========c.擂台赛：是不是我们最好的模型========
    # 我们希望Loss 越小越好
    if avg_val_loss < best_val_loss:
        print(f"发现新纪录!(Val loss:{best_val_loss:.4f} -> {avg_val_loss:.4f})")
        best_val_loss = avg_val_loss

        # 保存这个最好的模型
        torch.save(model.state_dict(),"best_model.pth")
        print("已保存 best_model.pth")
    else:
        print(f"表现一般，没有打破记录({best_val_loss:.4f})")

print("\n 训练结束。请使用 best_model.pth进行推理。")