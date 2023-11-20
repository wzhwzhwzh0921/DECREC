import pickle
import json
import os


print ([1]*5)
import numpy as np

# 创建两个示例数组
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# 沿着指定轴（axis）拼接数组
# 0 表示垂直拼接，1 表示水平拼接
# 在此示例中，我们进行水平拼接
result = np.concatenate((array1, array2), axis=0)

# 打印结果
print(result)

import numpy as np

# 创建两个示例数组
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# 垂直堆叠数组
vertical_stack = np.vstack((array1, array2))

# 水平堆叠数组
horizontal_stack = np.hstack((array1, array2))

# 打印结果
print("垂直堆叠：")
print(vertical_stack)
print("\n水平堆叠：")
print(horizontal_stack)
dir_path = os.path.dirname(os.getcwd())
from collections import defaultdict
import numpy as np
from scipy.stats import entropy
# 读取用户访问item记录（user_item.pkl）
with open(dir_path + '/UI_interaction_data/review_dict_train.json', 'rb') as f:
    user_item_data = json.load(f)
# print(user_item_data)
# print(user_item_data)
# 读取item的小特征和小特征到大特征的映射（item_fea.pkl和fea_large.pkl）
with open('item_feature.pkl', 'rb') as f:
    item_feature_data = pickle.load(f)
# print(item_feature_data)
with open('small_to_large.pkl', 'rb') as f:
    small_to_large_data = dict(pickle.load(f)) #33 0-32
# print(small_to_large_data)
max_=0
for large_feature in small_to_large_data.values() :
    la = list(large_feature)
    la.append(0)
    max_temp = max(la)
    if max_temp >max_ :
        max_ = max_temp
print(max_)

# 初始化统计字典
user_large_feature_counts = defaultdict(list)   # 用户访问每个大特征的总次数
user_unique_large_feature_counts = defaultdict(list)  # 用户访问每个大特征去重次数
user_large_feature_num = defaultdict(int)  # 用户访问每个大特征去重次数
user_total_interaction_counts = defaultdict(int)  # 用户总访问次数
# 遍历用户访问记录
for user, items in user_item_data.items():
    user_total_interaction_counts[user] = len(items)  # 记录用户总访问次数
    temp = [0] *33
    temp_unique = [0]*33
    for item in items:
        small_features = list(item_feature_data.get(item, []))
        all_large_features =[]
        for small_feature in small_features:
            large_features = list(small_to_large_data.get(small_feature, []))
            all_large_features+= large_features
        for large_feature in all_large_features:
            temp[large_feature] +=1
        all_large_features_unique = list(set(all_large_features))
        for large_features_unique in all_large_features_unique:
            temp_unique[large_features_unique] +=1
    n=0
    for i in temp:
        if i!=0:
            n+=1

    user_large_feature_counts[user] = temp
    user_unique_large_feature_counts[user] = temp_unique
    user_large_feature_num[user] = n
print(user_large_feature_counts.get('0',[]))
print(user_unique_large_feature_counts.get('0',[]))
print(user_large_feature_num.get('0',[]))
print(user_total_interaction_counts.get('0',[]))
entropy_value_li=[]
# 假设访问次数列表
for large_features_list in user_large_feature_counts.values():
    user_item_visits = large_features_list
    prob_distribution = np.array(user_item_visits) / sum(user_item_visits)
    entropy_value = entropy(prob_distribution, base=2)  # 使用2为底的对数计算熵
    entropy_value_li.append(entropy_value)
# 将访问次数列表转换为概率分布

# 计算熵

# print("熵:", entropy_value_li)