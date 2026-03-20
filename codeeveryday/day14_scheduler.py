import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import shutil
import os 

print("=" * 20 +"Day14:学习率调度器(Scheduler)" +"+" * 20)

# 0. 准备日志
log_dir =  "run/iqa_experimnet"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir) 
writer = SummaryWriter(log_dir = log_dir)

# 1.准备模型和环境
model = models.resnet18(weights='DEFAULT' )
model.fc = nn.Linear(512,1)
#初始学习率调高一点：0.1
optimizer = optim.SGD(model.parameters(),lr=0.1)

# 2.定义调度器（Scheduler)
# 策略：每隔10轮（step_size),把学习率×0.1（gamma）
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 10,gamma=0.1)

print("开始训练...(模拟30个Epoch)")

for epoch in range(30):
    #模拟Loss
    fake_loss = 1.0 /(epoch+1)
    
    # 3. 获取当前的学习率(用来画图)
    #优化器中有一个param_groups列表，存着当前的lr
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch}: LR = {current_lr:.6f}")

    # 记录到tensorboard
    writer.add_scalar('Learning Rate',current_lr)

    # 4.更新学习率
    #注意：schedular.step()通常在每个epoch结束时调用，而optimizer.step()是在每个Batch结束时调用
    scheduler.step()

writer.close()
print("-" *50)
print(" 训练结束。打开tensorboard查看学习率")
