import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data  import Dataset,DataLoader
import os
import random 

# ============ 1. 准备数据（复用Day4） =========
fake_file = "fake_labels.txt"
if not os.path.exists(fake_file):
    with open(fake_file,"w") as f:
        for i in range(20):  # 这次我们生成20个数据，多一些
            line = f"image_{i}.jpg,{random.uniform(1,5):.2f}\n"
            f.write(line)

class IQADataset(Dataset):
    def __init__(self,txt_file):
        self.data_info = []     # 用来存[文件名，分数]的列表
        with open(txt_file,"r") as f:
            for line in f:
                filename,score = line.strip().split(",")
                self.data_info.append((filename,float(score)))

    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self,idx):
        filename,score = self.data_info[idx]
        # 模拟读取照片，3通道，64 *64大小，跑的快一点
        fake_image = torch.randn(3,64,64)
        label = torch.tensor(score,dtype = torch.float32)
        return fake_image , label
    
# ========== 2.定义模型（day5新增） ===========
#所有模型都要继承 nn.Module
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #定义第一层卷积：输入3通道，输出16通道(也叫做卷积核数量)，卷积核为3x3
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3)    #16个独立的卷积核组，每个卷积核有三个通道
        self.relu = nn.ReLU()  #激活函数
        self.pool = nn.AdaptiveAvgPool2d((1,1))  #把无论多大的图，都变成1x1
        #全连接层：把16个通道的特征，变成一个分数，IQA是一个打分
        self.fc = nn.Linear(16, 1)

    def forward(self,x):      #self是类的实例对象
        # x 的形状：[Batch,3,64,64]
        x = self.conv1(x)   #->[batch,16,62,62]
        x = self.relu(x)    # 激活
        x = self.pool(x)    #全局平均池化
        x = x.view(x.size(0), -1)  # 拉平 -> [batch16]
        x = self.fc(x)      # ->[batch,1] (预测分数)
        return x
    
#===========3.准备训练环境==========
# A.设置设备（day3知识点）
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print(f"Trianing on : {device}")

# B.实例化组件
dataset = IQADataset(fake_file)
dataloader = DataLoader(dataset,batch_size = 4,shuffle = True)
model = SimpleCNN().to(device)  # 把模型搬到GPU

# C.定义裁判
# MSELoss :计算（预测值-真实值）^ 2
criterion = nn.MSELoss()
# SGD: 学习率 lr = 0.01
optimizer = optim.SGD(model.parameters(),lr=0.01)

print("开始训练")
print("-" *50)

#=============4. 训练循环（Training Loop)==========
# 训练5轮
for epoch in range(5):
    total_loss = 0

    # 每一轮里，遍历所有数据
    for batch_idx,(images,label) in enumerate(dataloader):
        #1. 搬运数据到GPU
        images = images.to(device)
        labels = label.to(device)

        #2. 梯度清零（必须做，否则梯度会累加）
        optimizer.zero_grad()

        #3. 前向传播（Model 预测）
        outputs = model(images)  #outputs 形状是[4,1]

        # 形状对齐：labels是[4],outputs是[4,1],把labels 变成[4,1]才能和outputs计算Loss
        loss = criterion(outputs,labels.view(-1,1))

        #4. 反向传播(算梯度)
        loss.backward()

        #5. 更新参数（根据梯度修正模型）
        optimizer.step()

        total_loss +=loss.item()

    print(f"Epoch {epoch+1},Average Loss{total_loss/len(dataloader):.4f}")

print("-" * 50 )
print("训练完成")

# 保存模型参数到当前目录下的 iqa_model.pth
torch.save(model.state_dict(), "iqa_model.pth")
print("模型已保存为 iqa_model.pth")