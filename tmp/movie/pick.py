import pickle

# 打开文件并读取序列化的数据
with open('kg.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# 现在loaded_data包含了从文件中加载的对象
print(loaded_data)
