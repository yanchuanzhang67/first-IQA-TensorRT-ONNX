import torch
import torch.nn as nn
from torchvision import models

print("=" * 20 + "Day11: 模型微调" + "=")

# 1.加载预训练模型
model = models.resnet18(weights = 'DEFAULT')

# 2. 核心步骤：冻结所有参数
# 遍历模型中的每一个参数，告诉PyTorch:这个参数不需要算梯度
for para in model.parameters():
    para.requires_grad = False

print("已冻结ResNet所有层")

# 3. 修改最后一层
# 注意：新创建的层，默认requires_grad = Ture（是要求算梯度的）
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,1)

print("已替换fc层(新层默认可训练)")

# 4. 验证一下：到底那些层会被训练？
print("\n 检查将会被更新的参数：")
for name,para in model.named_parameters():
    if para.requires_grad:
        print(f" 正在训练：{name}")
    # 这里的else分支就是那些被冻结的层（conv1,layer1...)

# 5. 定义优化器
# 关键点：优化器只给它传“需要更新的参数”
# 如果传 model.parameters() 也可以，但效率低，而且容易报错
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=1e-3)

print("\n 准备完毕!现在的模型只会更新FC层,前面的特征提取器纹丝不动")

