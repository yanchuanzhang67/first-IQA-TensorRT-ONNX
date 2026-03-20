import pandan as pd  # 业界行规，pandas简写成pd
import numpy as np
import os

print("=" *20 + "Day8: 数据分析实战" + "=" *20) 

# ============= 1. 造一个真实的CSV 表格 ============
csv_file = "koniq10k_mock.csv"

# 模拟100行数据

data = {
    "image_name" :[f"{1000+i}.jpg" for i in range(100)], #文件名：1000.jpg...
    "MOS": np.random.uniform(1.0,5.0,100).round(2),      #分数：1.00-5.00
    "height":np.random.choice([512,768,1080],100),        #随机分辨率高
    "width": np.random.choice([512,1024,1920],100)       #随机分辨率宽
}

#创建 DataFrame（Pandas的核心对象，相当于一张excel表）
df = pd.DataFrame(data)

#保存为CSV文件
df.to_csv(csv_file,index = False)
print(f"已生成模拟数据集表格：{csv_file}")
print("-" * 30)

# ============2.像科学家一样读取分析=============
# 读取csv
df_read = pd.read_csv(csv_file)

# A .查看前五行（看看数据长什么样）
print("数据预览(Head):")
print(df_read.head())
print("-" * 30)

# B. 统计信息（平均分、最大值、最小值）——写论文"Dataset Description"章节是比用
print("数据集统计：")
print(f"图片总数：{len(df_read)}")
print(f"平均MOS 分数：{df_read['MOS'].mean():.2f}")
print(f"最高分：{df_read['MOS'].max()}")
print(f"最低分：{df_read['MOS'].min()}")
print("-" *30)

# ==========3.数据清洗（Filtering）=========
#场景：把高分图片挑出来单独研究
#语法：df[ 条件 ]
high_quality_df = df_read[df_read['MOS'] >4.5]

print(f"高质量图片(分数>4.5)共有：{len(high_quality_df)}张")
print("部分展示：")
print(high_quality_df.head())

#保存清洗后的结果
high_quality_df.to_csv("high_quality_images.csv",index = False)
print("已保存筛选结果到high_quality_images.csv")