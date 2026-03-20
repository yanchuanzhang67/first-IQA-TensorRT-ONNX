import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
from torchvision import transforms    #学习重点
from PIl import Image   #Python处理图像的标准库

print("=" * 20 + "Day9:数据增强于标准化" + "="* 20)

# ====================1. 定义数据预处理(Transforms) ==========
#这一串操作就像一条流水线，图片进来会被依次处理
train_transform = transforms.Compose([
    #A.调整大小：不管原图多大，强行缩放到256x256
    transforms.Resize((256,256)),
    
    #B.随机裁剪，从256x256里随机切一块224x224出来
    transforms.RandomCrop((224,224)),

    #c.随机水平翻转，50%的概率让图片左右镜像
    transforms.RandomHorizontalFlip(p = 0.5),

    #D.必须要做，转为Tensor,并归一化到[0,1]
    transforms.ToTensor(),

    #E.标准化：（像素-mean)/std
    #这三个数字是ImageNet 数据集的统计均值和方差，业界通用标准
    transforms.Normalize(mean = [0.485,0.456,0.406], std =[0.229,0.224,0.225])

])


#===============2.升级版，Dataset(结合Pandas)================
class ProfessionalDataset(Dataset):
    def __init__(self,csv_file,transform=None):
        """
        :param csv_file : Day8 生成的csv路径
        :param transform: 上面定义的预处理流水线
        """

        # 1. 直接用pandas 读取csv, 一步到位
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        # 1. 获取第idx行数据
        # iloc是pandas 按索引取值的函数
        row = self.df.iloc[idx]

        filename = row['image_name']
        score = row['MOS']

        # 2. 模拟读取真实图片
        #假设原图尺寸是不固定的，比如我们根据csv里的height/width 来生成
        h,w = row['height'],['width']

        # 用numpy 生成随机噪点图（模拟unit8格式的JPG图片）
        fake_image_array = np.random.randint(0,225,(h,w,3),dtype = np.uint8)

        # 转成PIL Image对象（transforms 库要求输入必须是PIL Image）
        img = Image.fromarray(fake_image_array)

        #3. 关键步骤：应用预处理
        if self.transform:
            img_tensor = self.transform(img)
        else:
            #如果没有定义 transform,至少要转成Tensor
            img_tensor = transforms.ToTensor()(img)

        # 4. 处理标签
        label = torch.tensor(score,dtype = torch.float32)

        return img_tensor,label
    
# ================3. 测试管道===============
csv_path = 'koniq10k_mock.csv'  #确保生成的这个csv文件还在

#实例化数据集，传入我们的transform
dataset = ProfessionalDataset(csv_path,transform=train_transform)
dataloader = DataLoader(dataset,batch_size = 4,shuffle = True)

print(f"数据集长度：{len(dataset)}")

#获取一个Batch看看
#iter() 和next() 是Python 获取迭代器下一个元素的方法
images,labels = next(iter(dataloader))

print("\n 检查Batch数据:")
print(f"Input Images Shape:{images.shape}")
#预期输出：[4,3,224,224] -> 即使原图大小不一，现在也变成整齐了

print(f"Lables:{labels}")

# 验证一下数值范围（因为做了 Normaliza，所以会有负数）
print(f"Min Pixel: {images.min():.2f}, Max Pixel:{images.max():.2f}")
