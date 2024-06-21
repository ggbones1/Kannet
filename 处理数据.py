import pandas as pd
import numpy as np
from scipy.integrate import simps
df1 = pd.read_excel(r'c:\\Users\\liquanbo\\Documents\\WeChat Files\\wxid_gpvcwl5ukthi22\\FileStorage\\File\\2024-06\\0.5-中心1-流量数据.xlsx',sheet_name='插值速度x')
df2 = pd.read_excel(r'c:\\Users\\liquanbo\\Documents\\WeChat Files\\wxid_gpvcwl5ukthi22\\FileStorage\\File\\2024-06\\0.5-中心1-流量数据.xlsx',sheet_name='插值速度y')
df3 = pd.read_excel(r'c:\\Users\\liquanbo\\Documents\\WeChat Files\\wxid_gpvcwl5ukthi22\\FileStorage\\File\\2024-06\\0.5-中心1-流量数据.xlsx',sheet_name='插值速度v')
arr1 = df1.to_numpy().flatten()
arr2 = df2.to_numpy().flatten()
arr3 = df3.to_numpy().flatten()
x = arr1
y = arr2
z = arr3
n=10 # 块数
# 初始化积分结果
total_integral = 0.0

# 定义块的大小
block_size = len(x) // n

# 分块计算积分
def integrate_block(x_block, y_block, z_block, data_block):
    return simps(simps(simps(data_block, z_block), y_block), x_block)
# 将数据均匀分成十段
x= np.array_split(arr1, n, axis=0)
y= np.array_split(arr2, n, axis=0)
z= np.array_split(arr3, n, axis=0)
for k in range(0, n, block_size):
            x_block = x[k]
            y_block = y[k]
            z_block = z[k]
            data_block =np.random.rand(block_size, block_size, block_size)
            
            block_integral = integrate_block(x_block, y_block, z_block, data_block)
            total_integral += block_integral

print("三维积分结果:", total_integral)