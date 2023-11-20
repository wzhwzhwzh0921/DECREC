import pickle

# 打开文件并读取序列化的数据
# with open('tmp\movie\dataset.pkl', 'rb') as file:
#     loaded_data = pickle.load(file)
# for key,values in loaded_data:
#     print(key)
#     print(values)
# # 现在loaded_data包含了从文件中加载的对象
# print(loaded_data)
import matplotlib.pyplot as plt

data1 = [0.0240, 0.0205, 0.0418, 0.0196, 0.0067, 0.0117, 0.0331, 0.0536, 0.0378, 0.0448, 0.0264, 0.0271, 0.0278, 0.0287, 0.0000, 0.0000]

data2 = [0.0660, 0.0514, 0.0451, 0.0378, 0.0295, 0.0253, 0.0408, 0.0265, 0.0409, 0.0227, 0.0262, 0.0478, 0.0282, 0.0323, 0, 0]

data3 = [0.0420, 0.0397, 0.0217, 0.0422, 0.0348, 0.0192, 0.0547, 0.0622, 0.0557, 0.0366, 0.0585, 0.0528, 0.0689, 0.0461, 0, 0]



# x轴数据（假设有16个数据点）
x = list(range(16))



# 创建折线图
plt.plot(x, data1, label='UNICORN')
plt.plot(x, data2, label='MCMIPL')
plt.plot(x, data3, label='DECREC')

# 添加标签和标题
plt.xlabel('数据点')
plt.ylabel('数值')
plt.title('三条折线图')

# 添加图例
plt.legend()

# 显示图
plt.show()