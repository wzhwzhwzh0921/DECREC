import math
import random
import numpy as np
import sys
from tqdm import tqdm
import pickle
import json
import pickle
import ipdb


# 定义 LastFmGraph 类
class LastFmGraph(object):
    def __init__(self):
        # 初始化图结构
        self.G = dict()
        # 调用私有方法加载用户、物品和特征的信息
        self.__get_user__()
        self.__get_item__()
        self.__get_feature__()

    # 私有方法：获取用户信息
    def __get_user__(self):
        # 打开用户关系数据文件 user_friends.pkl
        with open('./data/lastfm_star/Graph_generate_data/user_friends.pkl', 'rb') as f:
            user_friends = pickle.load(f)
        # 打开用户喜欢数据文件 user_like.pkl
        with open('./data/lastfm_star/Graph_generate_data/user_like.pkl', 'rb') as f:
            user_like = pickle.load(f)
        # 打开用户交互数据文件 review_dict_train.json
        with open('./data/lastfm_star/UI_Interaction_data/review_dict_train.json', 'r', encoding='utf-8') as f:
            ui_train = json.load(f)
            # 初始化图中的用户节点
            self.G['user'] = {}
            for user in tqdm(ui_train):
                self.G['user'][int(user)] = {}
                # 用户的互动、好友和喜欢关系
                self.G['user'][int(user)]['interact'] = tuple(ui_train[user])
                self.G['user'][int(user)]['friends'] = tuple(user_friends[int(user)])
                self.G['user'][int(user)]['like'] = tuple(user_like[int(user)])

    # 私有方法：获取物品信息
    def __get_item__(self):
        # 打开物品特征数据文件 item_fea.pkl
        with open('./data/lastfm_star/Graph_generate_data/item_fea.pkl', 'rb') as f:
            item_feature = pickle.load(f)
        # 打开特征映射数据文件 fea_large.pkl
        with open('./data/lastfm_star/Graph_generate_data/fea_large.pkl', 'rb') as f:
            small_to_large = pickle.load(f)

        feature_index = {}  # 特征索引字典
        i = 0
        # 构建特征索引
        for key in small_to_large.keys():
            if key in feature_index:
                continue
            else:
                feature_index[key] = i
                i += 1

        self.G['item'] = {}  # 初始化图中的物品节点
        # 遍历物品特征信息
        for item in item_feature:
            self.G['item'][item] = {}
            fea = []
            # 将小特征映射到大特征的索引
            for feature in item_feature[item]:
                fea.append(feature_index[feature])
            # 物品的属于、互动和大特征关系
            self.G['item'][item]['belong_to'] = tuple(set(fea))
            self.G['item'][item]['interact'] = tuple(())
            self.G['item'][item]['belong_to_large'] = tuple(())

        # 遍历用户节点，为物品节点添加互动关系
        for user in self.G['user']:
            for item in self.G['user'][user]['interact']:
                self.G['item'][item]['interact'] += tuple([user])

    # 私有方法：获取特征信息
    def __get_feature__(self):
        # 打开特征映射数据文件 fea_large.pkl
        with open('./data/lastfm_star/Graph_generate_data/fea_large.pkl', 'rb') as f:
            small_to_large = pickle.load(f)

        feature_index = {}  # 特征索引字典
        i = 0
        # 构建特征索引
        for key in small_to_large.keys():
            if key in feature_index:
                continue
            else:
                feature_index[key] = i
                i += 1

        self.G['feature'] = {}  # 初始化图中的特征节点
        # 遍历小特征映射到大特征的字典
        for key in small_to_large:
            idx = feature_index[key]
            self.G['feature'][idx] = {}
            # 特征的链接到特征、喜欢和属于关系
            self.G['feature'][idx]['link_to_feature'] = tuple(small_to_large[key])
            self.G['feature'][idx]['like'] = tuple(())
            self.G['feature'][idx]['belong_to'] = tuple(())

        # 遍历物品节点，为特征节点添加属于关系
        for item in self.G['item']:
            for feature in self.G['item'][item]['belong_to']:
                self.G['feature'][feature]['belong_to'] += tuple([item])

        # 遍历用户节点，为特征节点添加喜欢关系
        for user in self.G['user']:
            for feature in self.G['user'][user]['like']:
                self.G['feature'][feature]['like'] += tuple([user])

            

            