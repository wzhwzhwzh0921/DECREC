import pickle
from collections import defaultdict
import numpy as np
from scipy.stats import entropy
# 读取用户访问item记录（user_item.pkl）
with open('user_like.pkl', 'rb') as f:
    user_item_data = pickle.load(f)

# print(user_item_data)
# 读取item的小特征和小特征到大特征的映射（item_fea.pkl和fea_large.pkl）
with open('item_fea.pkl', 'rb') as f:
    item_feature_data = pickle.load(f)
# print(item_feature_data)
with open('fea_large.pkl', 'rb') as f:
    small_to_large_data = dict(pickle.load(f)) #33 0-32
# print(small_to_large_data)
# max_=0
# for large_feature in small_to_large_data.values() :
#     la = list(large_feature)
#     la.append(0)
#     max_temp = max(la)
#     if max_temp >max_ :
#         max_ = max_temp
# print(max_)

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
print(user_large_feature_counts.get(2,[]))
print(user_unique_large_feature_counts.get(2,[]))
print(user_large_feature_num.get(2,[]))
print(user_total_interaction_counts.get(2,[]))
entropy_value_li=[]
# 假设访问次数列表
for large_features_list in user_large_feature_counts.values():
    user_item_visits = large_features_list
    prob_distribution = np.array(user_item_visits) / sum(user_item_visits)
    entropy_value = entropy(prob_distribution, base=2)  # 使用2为底的对数计算熵
    entropy_value_li.append(entropy_value)
# 将访问次数列表转换为概率分布

# 计算熵

print("熵:", entropy_value_li)

# 将统计结果保存到字典文件
