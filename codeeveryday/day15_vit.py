import torch
import torch.nn as nn
from torchvision import models

print("=" * 20 + "Day15: Vision Transformer(Vit)" +"=" *20)

# ============= 1. 加载ViT模型===============
print("正在下载权重/加载ViT-B-16...")
vit = models.vit_b_16(weight = 'DEFAULT')
# 下的是16x16的patch,大概是300MB
print("ViT 加载成功")

# =============2.观察ViT结构（解剖）=======
print(vit)

#核心考点，图片是怎么变成序列的，224x224 -> 16x16 再加一个class token(代表整张图的分类特征，放在最前面)
# 196个patch,序列长度为197，
print("\n Vit 核心参数揭秘：")
print(f"Patch size: {vit.patch_size}")
print(f"Hidden Dim:{vit.hidden_dim}(每个Patch被映射为一个768维的变量)")

# ========= 3.修改Head（迁移学习）=======
# Resnet最后一层叫‘fc', ViT的最后一层是‘heads'
print(f"\n 原始Head:{vit.heads}")

# 将最后一层修改为IQA任务，将输出变成一个分数，获取输入特征维数
in_features= vit.heads.head_in_features
vit.heads.head = nn.Linear(in_features,1)

print(f"修改后的Head:{vit.heads}")

# ======== 4.跑通推理============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit = vit.to(device)

# 模拟一个Batch
dummy_img = torch.randn(4,3,224,224).to(device)

print("\n正在进行VIT 推理...")
output = vit(dummy_img)
print(f"输入形状：{dummy_img.shape}")
print(f"输出形状：{output.shape}")   # 预期[4,1]
