import numpy as np   # 这是行规，大家都把numpy简写成np

# 步骤2 ：造一张“假图片” （Create）

print("步骤2 : 创建黑色图片")
# 创建一个全为0 的数组，形状是（224，224，3）
image = np.zeros((224,224,3))
print (f"原始图片形状：{image.shape}")  #设置图片形状为（224，224，3）
print (f"数据类型：{image.dtype}")
print ("-" * 50 )

# 步骤3 ：调整视角（Transpose） -- 重点
print("步骤 3 :转置图片(HWC → CHW)")
#把（224，224，3）变换成（3，224，224）
#原索引：0(H),1(W),2(C) → 新索引2(C),0(H),1(W)
image_transpose = image.transpose(2,0,1)
print(f"转置后形状：{image_transpose.shape}")  #图片形状变成（3,224,224,)
print("-" * 50 )

# 步骤 4 ：增加“批次” （Expend dims)维度
print("步骤4: 增加批次维度")
#把（3，224，224）变成（1，3，224，224）
#在第0个位置插入批次维度N
image_batched = np.expand_dims(image_transpose,axis = 0)
print (f"增加批次后形状：{image_batched.shape}")

# 步骤 5 ：展平它（Flatter）
image_flatten = image_batched.flatten()
print(f"使用 flatten() 后长度：{len(image_flatten)}")  

# 计算理论长度
expected_length = 1 * 3 * 224 *224
print(f"理论计算长度：{expected_length}")

# 验证计算是否正确
print(f"验证结果：{len(image_flatten) == expected_length}")
print("-" * 50)