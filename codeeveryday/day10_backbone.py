import torch
import torch.nn as nn 
from torchvision import models  #从torch库中下载好已有的模型

print("="* 20 + " Day10: 迁移学习ResNet" +"=" * 20)

# ================1. 加载预训练的模型===================
# Weight = 'DEFAULT' 表示下载并加载在ImageNet 上训练好的参数
# 第一次运行会自动下载模型，存到服务器缓存里面，大概45MB
print ("正在下载/加载 ResNet18 预训练权重...")
resnet = models.resnet18(weight = 'DEFAULT')

print("模型加载成功")

# ================2. 观察模型结构（这是基本功，能够看懂模型长什么样子）================
# 这一步会打印出网络的所有层
# 会发现最后一层叫 "fc" : Linear（in_features= 512 ，out_features = 1000,bias = True）
print(resnet)

# ============== 3. 手术改造（修改Head） =================
# 我们的目标：把最后一层输出1000改成输出为IQA评估分数的1
# 1.获取最后一层全连接层的输入通道数（ResNet18 是512）
num_ftrs = resnet.fc.in_features

print(f"原始全连接层输入特征数:{num_ftrs}")
print(f"原始全连接层输出特征数：{resnet.fc.out_features}(对应ImageNet1000分类)")

# 2.暴力替换最后一层
# nn.Linear(512,1) -> 输入512，输出1（MOS分数）
resnet.fc = nn.Linear(num_ftrs,1)

print(f"改造后的全连接层：{resnet.fc}")

# ==================4. 封装自己的IQA模型================
# 在实际工程当中，通常这样写
class ResNetIQA(nn.Module):
    def __init__(self):
        super().__init__()
        #加载backbone
        self.backbone = models.resnet18(weight = 'DEFAULT')
        #修改最后一层
        self.backbone.fc = nn.Linear(512,1)

    def forward(self,x):
        return self.backbone(x)
    
# ===============5. 测试跑通=================
device = torch.device("cuda" if torch.cuda.is_available else "CPU")
model = ResNetIQA().to(device)

# 模拟一个Batch 的图片（有关day9的知识）
#[Batch = 4,Channel = 3, Height = 224, Width = 224)
dummy_input = torch.randn(4,3,224,224).to(device)

print("\n正在进行前向推理")
output = model(dummy_input)

print(f"输入形状：{dummy_input.shape}")
print(f"输出形状：{output.shape}")
# 预期输出：[4,1] -> 完美符合IQA要求 

print("\n 认识了这个ImageNet世界级额特征提取器")