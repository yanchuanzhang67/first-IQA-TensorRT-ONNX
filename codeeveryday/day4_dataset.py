import torch 
from torch.utils.data import Dataset,DataLoader
import os
import random

# =========准备工作：生成假数据======
# 我们在当前目录下创建一个fake_labels.txt 模拟IQA的标签文件
# 格式：文件名，分数

fake_file = "fake_labels.txt"
if not os.path.exists(fake_file):
    print(f"正在生成假标签文件：{fake_file} ...")
    with open(fake_file,"w") as f:
        for i in range(10): # 假装我们有10张图
            #随机生成一个文件名和分数（1-5分）
            line = f"image_{i}.jpg,{random.uniform(1,5):.2f}\n"
            f.write(line)
    print("生成完毕！")
else:
    print(f"{fake_file} 已存在，跳过生成")

print("-" * 50)


# ========核心：定义自己的Dataset==========
class IQADataset(Dataset):
    def __init__(self,txt_file):
        """
        初始化函数：在这里我们只读取清单，不读取真正的图片
        这样可以避免内存爆炸。
        """
        self.data_info = []    # 用来存[文件名，分数]的列表
        # 打开txt 文件，一行一行读
        with open(txt_file,"r") as f:
            for line in f:
                #line 的样子是"image_0.jpg,4.50\n"
                #strip() 去掉换行符，split(',') 按逗号切割
                filename,score  = line.strip().split(",")
                self.data_info.append((filename,float(score)))
    
    def __len__(self):
        """
        告诉 DataLoader 我们一共有多少数据
        """
        return len(self.data_info)
    
    def __getitem__(self,idx):
        """
        PyTorch 想要第idx 个数据是，会调用这个函数。
        这里是真正干活（读取图片、预处理）的地方。
        """
        # 1. 根据索引idx,从清单里找到文件名和标签
        filename,score = self.data_info[idx]

        # 2. 读取图片（用随机tensor模拟） 真实场景中：image= cv2.imread(filename) , 
        fake_image = torch.randn(3,224,224)

        # 3. 处理标签 ，必须转成Tensor，且IQA的分数一般是float32
        label = torch.tensor(score,dtype=torch.float32)

        # 4. 返回一个字典或元组（Image，Label)
        return fake_image , label
    
# ======测试管道========
print("1.实例化Datase..")
my_dataset = IQADataset(fake_file)
print(f"数据集中共有{len(my_dataset)}个样本。")

# 测试  __getitem__ 
img , label = my_dataset[0]  # 拿第0个元素试试
print(f"第0个样本 -  图像形状：{img.shape}, 分数{label}")
print( "-" * 50)

print("2. 实例化 DataLoader(打包工)...")
#batch_size = 4 :每次吐出4个样本
train_loader = DataLoader(dataset= my_dataset,batch_size = 4,shuffle= True)

print("3. 模拟训练循环(epoch)..")
#这里模拟了一次训练过程
for i, (images,label) in enumerate(train_loader):
    # images 的形状应该是 （4，3，224，224） -> (Batch,Chanel,H,W)
    # labels 的形状应该是 （4）
    print(f"Batch{i}:")
    print(f" 图像Batch形状:{images.shape}")
    print(f" 标签Batch 内容：{label}")