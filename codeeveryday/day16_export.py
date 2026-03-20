import torch
import torch.nn as nn
from torchvision import models
import onnx

print("=" * 20 + "Day16 :模型导出(ONNX)" + "=" * 20)

# ========== 1. 准备模型==========（还是使用resnet18模型）
model = models.resnet18(weights = 'DEFAULT')
model.fc = nn.Linear(512,1)  # 改成IQA的head
model.eval() # 切换到eval模式，

# ========2. 准备假输入========
# 导出ONNX 需要让模型跑一遍，记录数据流动的路径
# 所以需要喂给它一个假数据
# 形状： [Batch = 1,channel = 3, Height = 224,Width = 224]
dummy_input = torch.randn(1,3,224,224)

# ======== 3.核心：导出ONNX======
output_file = "resnet_iqa.onnx"

print(f"正在导出模型到{output_file}...")

# torch.onnx.export是今天的核心函数
torch.onnx.export(
    model,              # 要导入的模型
    dummy_input,        # 假输入
    output_file,        # 保存的文件名
    export_params=True, # 是否把权重（weights）也存进去，当然要
    opset_version =11   # ONNX版本
    do_constant_folding = True  #优化模型
    input_names = ['input']   # 给输入节点起个名字
    output_names = ['output']

    # 关键点，动态轴 dynamic axes
    # 如果不写这行，模型只能处理 Batch =1 的图片
    # 有了这行，C++部署时 ，Batch Size 可以时任意数（1，4，8）
    dynami_axes ={
        'input':{0:'batch_size'}, # 第0维时动态的
        'output':{0:'batch_size'}
    }
)


print(f"导出文件成功，文件大小：{os.path.getsize(output_file)/1024/1024:.2f} MB")

# ======== 4.验证 ONNX模型=======
# 这一步时为了确保导出的模型结构没有坏掉 import onnx

onnx_model = onnx.load(output_file)
try:
    onnx.checker.check_model(onnx_model)
    print("ONNX模型结构检查通过!他是合法的")
except onnx.checker.ValidationError as e:
    print(f" 模型有错：{e}")

print("\n 下一步,我们将用c++来加载这个.onnx文件")