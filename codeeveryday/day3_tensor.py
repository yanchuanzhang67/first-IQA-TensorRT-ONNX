import torch
import numpy as np

print("=" *20 + " 步骤1: Numpy进化到tensor " +  "=" *20)
#1. 创建一个Numpy数组（模拟一张全黑的图片：3通道，224*242）
#关键点，使用dtype=np.float32,表示使用32位双精度的字符，深度学习默认用float32，不用这个会报错或者显存爆炸
np_data = np.zeros((3,224,224),dtype = np.float32)


#2.将Numpy数组转换为PyTorch Tensor张量，
# torch.from_numpy() 是零拷贝转换，非常快，他和原来的numpy数组共享内存
tensor_data = torch.from_numpy(np_data)

print(f"Numpy形状:{np_data.shape},类型：{np_data.dtype}")
print(f"Tensor形状:{tensor_data.shape},类型：{tensor_data.dtype}")
print ("-" * 50)


print ("=" *20 + " 步骤2:点亮GPU(搬运工)" + "=" *20)
#1.自动检测显卡
#这行代码是工业界的标准写法：如果显卡（cuda)可用，就用（cuda);否则只能委屈用cpu
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print (f"当前使用的计算设备：{device}")

#2.把Tensor搬运到显卡上
#注意：这一步是有返回值的，必须赋值给一个新的变量，或者覆盖原变量
tensor_gpu = tensor_data.to(device)

print(f"原tensor设备:{tensor_data.device}")
print(f"搬运后的Tensor:{tensor_gpu.device}")
#如果成功，应该能成功看到device = 'cuda:0'
print ("-" *60)


print("=" *20 +"步骤3:体验自动求导(Autograd)" +"="*20)
#1.创建一个需要求导的变量X = 2.0
#require_grad = True 是告诉PyTorch :“请帮我记录这个变量的所有计算过程”
x = torch.tensor([2.0],requires_grad = True)

#2.定义函数：
y = x ** 2 + 5

print(f"x的值:{x.item()}")
print(f"y的值(计算结果):{y.item()}")

#3.反向传播（back Propagation)
#这一句是深度学习的核心，它会自动计算链式法则，自动计算dy/dx
y.backward()

#4.查看梯度
print(f"x的梯度:{x.grad}")
print(f"验证结果：{x.grad.item() == 4.0}")