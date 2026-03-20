import torch
import torch.nn as nn

# 1. 必须先把模型结构重新定义一遍（通常会把模型定义单独放在一个model.py文件里 再import进来）
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(16,1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    
# 2. 准备环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Inference on: {device}")

# 3. 实例化模型
model = SimpleCNN().to(device)

# 4. 加载权重（Load Weights)
#map_location 是为了防止在GPU上训练，结果在只有CPU的电脑上跑报错
model.load_state_dict(torch.load("iqa_model.pth",map_location=device, weights_only=True))
print("成功加载权重")

# 5. 切换到评估模式（Evaluation Mode） 重要的一步！！！！
#这里将会告诉BatchNorm 和 Dropout 层，不要再改变了
model.eval()

# 6. 模拟一张新图片（推理Inference)
#假设这张图也是（1，3，64，64） -> Batch Size = 1
fake_image = torch.randn(1,3,64,64).to(device)

# 7. 预测
print("正在为新图片打分...")
# with torch.no_grad() 告诉 PyTorch:这一段代码不需要计算梯度，省显存
with torch.no_grad():
    prediction = model(fake_image)

print(f"预测模型分数：{prediction.item():.2f}")
 