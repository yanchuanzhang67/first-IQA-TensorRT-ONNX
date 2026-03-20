import torch
import numpy as np
from scipy import stats  # 这是写论文必备的统计库

print("=" * 20 + "IQA评价指标实践" + "=" *20)

# 1. 模拟数据
#假设我们有5张图
#target:真实的人类打分
#pred_good: 一个训练的很好的模型预测的分数
#pred_bad : 一个瞎猜的模型预测的分数
target = np.array([1.0,2.0,3.0,4.0,5.0])

pred_good = np.array([1.1,2.1,3.1,4.1,5.1])  # 趋势完全一致
pred_bad = np.array([5.0,4.0,3.0,2.0,1.0])  #完全反了

# 2. 定义计算函数（在之后的论文代码里可以直接赋值这个函数）
def compute_metrics(pred,target):
    srcc,_ = stats.spearmanr(pred,target)  #SRCC：spearmanr等级相关系数（关注排名）
    plcc,_ = stats.pearsonr(pred,target)   #PLCC:personr线性相关系数（关注线性拟合度)
     
    return srcc ,plcc

# 3. 评测好模型
srcc ,plcc = compute_metrics(pred_good,target)
print(f"好模型性能：")
print(f" SRCC:{srcc:.4f}(越接近于1越好)")
print(f" PLCC:{plcc:.4f}(越接近于1越好)")

print("*" *50)

# 4. 评测坏模型
srcc , plcc = compute_metrics(pred_bad,target)
print(f"坏模型性能")
print(f" SRCC:{srcc:.4f}")
print(f" PLCC{plcc:.4f}")

# 5. 思考题代码验证
#如果预测分都偏高10分，会影响SRCC吗？
pred_shift = target + 10.0
srcc, plcc = compute_metrics(pred_shift,target)
print("-" * 30)
print(f"偏移模型性能(整体加10分)")
print(f"  SRCC:{srcc:.4f}")
print(f"  PLCC:{plcc:.4f}")