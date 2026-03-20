class IQA_student:
    #这是初始化函数，造人时自动调用
    def __init__(self,name,university,paper_count):
        self.name = name
        self.university = university
        self.paper_count = paper_count

    # 定义学习方法
    def study(self):
        print("Reading a paper ...")
        # 让“我自己”的论文数 +1
        self.paper_count += 1

    # 定义自我介绍方法
    def info(self):
        # 这里的 f"..." 是python 的格式化字符串，大括号里可以直接填变量
        print(f"我是{self.name},来自{self.university},读了{self.paper_count}篇论文。")

# TJU_student 继承了 IQA_student 的所有东西
class TJU_student(IQA_student):

    # 只重写 info 方法，其他的不变
    def info(self):
        # 先打印这一行特殊的
        print("=== 天津大学优秀毕业生 ===")
        # 然后调用父类（super）的 info 方法打印基础信息
        # 这一句相当于把上面的 IQA_student 的 info 代码拿过来跑一遍
        super().info()

# 1.实例化：造一个叫 me 的对象
#这里会自动调用 __init__ , 把括号里的三个值分别赋值给 self.name , self.university ...
me = TJU_student("EKio","Tianjin University",40)

# 2.调用方法
print("--- 开始学习 ---")
me.study() # 第一次学习，论文数加1
me.study() # 第二次学习

# 3.打印最终信息
print("--- 最终状态 ---")
me.info()