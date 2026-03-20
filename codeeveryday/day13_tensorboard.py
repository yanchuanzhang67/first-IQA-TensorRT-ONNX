import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torch.utils.tensorboard import SummaryWriter   # 今天主要学的内容
import shutil   # 用来删除旧日志
import os

print("=" * 20 + "Day13:可视化神器TensorBoard" + "=")

# ======== 0. 清除旧日志（可选）=========
log_dir = "run/iqa_experimnet"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)  #删除旧的，防止曲线重叠

# ==========1. 初始化Writer==========
# 这会在当前目录下生成一个‘runs'文件夹
writer = SummaryWriter(log_dir = log_dir)

# ....（中间数据集的定义，模型定义省略，和DAY12一样）
#为了省代码，直接采用假数据集模拟训练过程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights = 'DEFAULT')
model.fc = nn.Linear(512,1)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

# ========== 2.训练循环中埋点=============
print("开始训练...(注意runs文件夹生成)")

for epoch in range(10):
    #模拟训练用loss(假设它在慢慢下降)
    #这里用随机数模拟，方便演示
    train_loss = 10.0 / (epoch +1 ) + torch.randn(1).item() *0.5
    val_loss = 10.0 /(epoch +1) + torch.randn(1).item() *0.8

    print(f"Epoch {epoch+1}:Train Loss {train_loss:.2f} | Val Loss {val_loss:.2f}")

    # 核心，写入日志
    # add_scalar('图表标题’，y轴数值。x轴数值) 
    writer.add_scalar('Loss/Train',train_loss,epoch)
    writer.add_scalar('Loss/Val',val_loss,epoch)

    #记录学习率
    writer.add_scalar('Learning Rate',optimizer.param_groups[0]['lr'],epoch)

# ===========3.关闭writer=========
writer.close()
print("-" * 50)
print("日志已经写入")
print("请在终端运行以下命令启动命令面板：")
print(f"tensorboard --logdir={log_dir}")