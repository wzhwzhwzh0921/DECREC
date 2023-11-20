
import json
import numpy as np
import os
import random
from utils import *
from torch import nn
import ipdb
from tkinter import _flatten
from collections import Counter


class MultiChoiceRecommendEnv(object):
    def __init__(self, kg, dataset, data_name, embed, seed=1, max_turn=15, cand_num=15, cand_item_num=15, attr_num=20,
                 mode='train', ask_num=1, entropy_way='weight entropy', fm_epoch=0, choice_num=2, fea_score="entropy"):
        self.data_name = data_name  # 数据集的名称
        self.mode = mode  # 模式：'train'用于训练，'test'用于测试
        self.seed = seed  # 随机种子
        self.max_turn = max_turn  # 对话的最大轮次
        self.attr_state_num = attr_num  # 属性状态数量
        self.kg = kg  # 知识图谱对象
        self.dataset = dataset  # 数据集对象
        self.feature_length = getattr(self.dataset, 'feature').value_len  # 特征的长度
        self.user_length = getattr(self.dataset, 'user').value_len  # 用户的长度
        user = getattr(self.dataset, 'user').id

        self.item_length = getattr(self.dataset, 'item').value_len  # 物品的长度
        self.large_feature_length = 42  # 大特征的长度
        self.other_feature = self.large_feature_length + 1  # 其他特征的索引
        self.small_feature_to_large = {}  # 小特征对应的大特征映射字典
        self.get_feature_dict()  # 初始化特征映射字典
        self.choice_num = choice_num  # 选择的数量
        self.fea_score = fea_score  # 特征得分计算方式，'entropy' 或 'embedding'

        # 动作参数
        self.ask_num = ask_num  # 每轮的询问数量
        self.rec_num = 10  # 推荐的数量
        self.random_sample_feature = False  # 是否随机采样特征
        self.random_sample_item = False  # 是否随机采样物品

        # 根据候选数设置参数
        if cand_num == 0:
            self.cand_num = 15
            self.random_sample_feature = True
        else:
            self.cand_num = cand_num
        if cand_item_num == 0:
            self.cand_item_num = 15
            self.random_sample_item = True
        else:
            self.cand_item_num = cand_item_num

        # 熵的计算方式，'entropy' 或 'weight entropy'
        self.ent_way = entropy_way

        # 用户的个人资料
        self.reachable_feature = []  # 用户可达特征
        self.user_acc_feature = []  # 用户接受的特征，由agent提问得到
        self.user_rej_feature = []  # 用户拒绝的特征，由agent提问得到
        self.cand_items = []  # 候选物品
        self.rej_item = []  # 被用户拒绝的物品
        self.item_feature_pair = {}  # 物品-特征对应关系
        self.cand_item_score = []  # 候选物品的得分

        # 用户ID、物品ID、对话步数、当前节点集等初始化
        self.user_id = None
        self.target_item = None
        self.cur_conver_step = 0  # 当前步数中的对话数
        self.cur_node_set = []  # 当前节点集，可能是单一节点或节点集，通常保存特征节点

        # 状态向量
        self.user_embed = None  # 用户嵌入向量
        self.conver_his = []  # 对话历史
        self.attr_ent = []  # 特征熵

        # 加载数据
        self.ui_dict, self.u_multi = self.__load_rl_data__(data_name, mode=mode)  # 用户-物品字典、用户-多物品字典
        item_freq = {}
        for user_items in self.u_multi.keys():
            for items in self.u_multi[user_items]:
                for item in items:
                    if item in item_freq:
                        item_freq[item] +=1
                    else:
                        item_freq[item] = 1
        yuzhi = 0
        item_long_tail = []
        item_head = []
        for item in item_freq:
            if item_freq[item] < yuzhi:
                item_long_tail.append(item)
            else:
                item_head.append(item)

        self.user_weight_dict = dict()  # 用户权重字典
        self.user_items_dict = dict()  # 用户物品字典

        # 初始化随机种子和用户字典
        set_random_seed(self.seed)  # 设置随机种子
        if mode == 'train':
            self.__user_dict_init__()  # 初始化用户权重和用户物品字典
        elif mode == 'test':
            self.ui_array = None  # 用户-物品数组
            self.__test_tuple_generate__()  # 生成测试元组
            self.test_num = 0
        # 加载嵌入向量
        embeds = load_embed(data_name, embed, epoch=fm_epoch)
        if len(embeds) != 0:
            self.ui_embeds = embeds['ui_emb']  # 用户物品嵌入向量
            self.feature_emb = embeds['feature_emb']  # 特征嵌入向量
        else:
            self.ui_embeds = nn.Embedding(self.user_length + self.item_length, 64).weight.data.numpy()
            self.feature_emb = nn.Embedding(self.feature_length, 64).weight.data.numpy()

        self.action_space = 2  # 动作空间的维度

        # 奖励和历史字典
        self.reward_dict = {
            'ask_suc': 0.01,
            'ask_fail': -0.1,
            'rec_suc': 1,
            'rec_fail': -0.1,
            'until_T': -0.6,  # 达到最大轮次的惩罚
            'cand_none': -0.1  # 候选物品为空的惩罚
        }
        self.history_dict = {
            'ask_suc': 1,
            'ask_fail': -1,
            'rec_suc': 2,
            'rec_fail': -2,
            'until_T': 0  # 达到最大轮次的历史记录标识
        }
        self.attr_count_dict = dict()  # 用于计算特征熵的计数字典

    def __load_rl_data__(self, data_name, mode):
        if mode == 'train':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_valid.json'), encoding='utf-8') as f:
                print('train_data: load RL valid data')
                mydict = json.load(f)
            with open(os.path.join(DATA_DIR[data_name], 'UI_data/train.pkl'), 'rb') as f:
                u_multi = pickle.load(f)
        elif mode == 'test':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_test.json'), encoding='utf-8') as f:
                print('test_data: load RL test data')
                mydict = json.load(f)
            with open(os.path.join(DATA_DIR[data_name], 'UI_data/test.pkl'), 'rb') as f:
                u_multi = pickle.load(f)

        return mydict,u_multi


    def __user_dict_init__(self):   #Calculate the weight of the number of interactions per user
        ui_nums = 0
        for items in self.ui_dict.values():
            ui_nums += len(items)
        for user_str in self.ui_dict.keys():
            user_id = int(user_str)
            self.user_weight_dict[user_id] = len(self.ui_dict[user_str])/ui_nums
        print('user_dict init successfully!')

    def get_feature_dict(self):
        num=0
        # for n in self.kg.G['item']:
        #     enablePrint()
        #     ipdb.set_trace()
        for m in self.kg.G['feature']:
            if len(self.kg.G['feature'][m]['link_to_feature']):
                large=self.kg.G['feature'][m]['link_to_feature'][0]
                self.small_feature_to_large[m]=large
            else:
                self.small_feature_to_large[m]=self.other_feature
                num+=1
            

    def __test_tuple_generate__(self):
        ui_list = []
        for user_str, items in self.u_multi.items():
            user_id = int(user_str)
            for item_id in items:
                ui_list.append([user_id, item_id])
        self.ui_array = np.array(ui_list)
        np.random.shuffle(self.ui_array)

    def get_sameatt_items(self):
        users = list(self.ui_dict.keys())  # 获取所有用户ID

        self.ui_satt_items = {}  # 用户-物品-相同属性物品字典，存储每个用户每个物品的相同属性物品集合

        for user in users:
            user = int(user)  # 转换为整数类型
            all_items = self.ui_dict[str(user)]  # 获取用户的所有物品
            same_att_items = {}  # 物品-相同属性物品字典，存储每个物品的相同属性物品集合

            a2i, i2a = {}, {}  # 属性到物品的映射字典，物品到属性的映射字典
            for item in all_items:
                att = set(self.kg.G['item'][item]['belong_to'])  # 获取物品的属性集合
                i2a[item] = att  # 将物品与属性的映射添加到字典中
                for a in att:
                    if a in a2i:
                        a2i[a].append(item)  # 如果属性已经在字典中，将物品添加到对应的列表中
                    else:
                        a2i[a] = [item]  # 如果属性不在字典中，创建一个新的列表并添加物品
            for item in all_items:
                can_att = i2a[item]  # 获取物品的属性集合
                can_items = []  # 存储相同属性物品的列表
                for a in can_att:
                    tmp_items = a2i[a]  # 获取相同属性的物品列表
                    can_items += tmp_items  # 将相同属性的物品添加到列表中
                same_att_items[item] = can_items  # 将相同属性物品列表与物品关联存储

            self.ui_satt_items[user] = same_att_items  # 将用户的相同属性物品字典与用户关联存储

    def reset(self, embed=None):
        # 如果传入了嵌入向量，则将其分配给用户和物品的嵌入向量
        if embed is not None:
            self.ui_embeds = embed[:self.user_length + self.item_length]
            self.feature_emb = embed[self.user_length + self.item_length:]

        # 初始化当前会话步骤、当前节点集合等
        self.cur_conver_step = 0
        self.cur_node_set = []
        self.rej_item = []

        # 根据模式初始化用户和目标物品
        if self.mode == 'train':
            users = list(self.user_weight_dict.keys())
            self.user_id = np.random.choice(users)
            self.target_item = random.choice(self.u_multi[str(self.user_id)])
        elif self.mode == 'test':
            self.user_id = self.ui_array[self.test_num, 0]
            self.target_item = self.ui_array[self.test_num, 1]
            self.test_num += 1

        # 初始化用户的特征信息
        print('-----------重置状态向量------------')
        print('user_id:{}, target_item:{}'.format(self.user_id, self.target_item))
        feature_groundtrue = []
        for i in self.target_item:
            feature_groundtrue += self.kg.G['item'][i]['belong_to']
        self.feature_groundtrue = list(set(feature_groundtrue)) #获取target_item所对应的属性（后续提问）

        # 初始化其他变量，如可达特征、用户接受的特征、候选物品等
        self.reachable_feature = []
        self.user_acc_feature = []
        self.user_rej_feature = []
        self.cand_items = list(range(self.item_length))

        # 初始化用户嵌入、会话历史等
        self.user_embed = self.ui_embeds[self.user_id].tolist()
        self.conver_his = [0] * self.max_turn
        self.attr_ent = [0] * self.attr_state_num

        # 随机选择一个问题来初始化对话
        attrs = set(self.kg.G['item'][self.target_item[0]]['belong_to'])
        for i in range(1, len(self.target_item)):
            attrs2 = set(self.kg.G['item'][self.target_item[i]]['belong_to'])
            attrs = attrs & attrs2   # 随机选的特征
        attrs = list(attrs)
        user_like_random_fea = random.choice(attrs)
        self.user_acc_feature.append(user_like_random_fea)
        self.cur_node_set.append(user_like_random_fea)
        self._update_cand_items([user_like_random_fea], acc_rej=True) #更新备选项目
        self._updata_reachable_feature() #更新可以询问的特征
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
        self.cur_conver_step += 1

        print('=== 初始化用户偏好特征: {}'.format(self.cur_node_set))
        self._update_feature_entropy()
        print('重置可达特征数目: {}'.format(len(self.reachable_feature)))

        # 根据特征熵对可达特征进行排序
        reach_fea_score = self._feature_score()
        max_ind_list = []
        for k in range(self.cand_num):
            max_score = max(reach_fea_score)
            max_ind = reach_fea_score.index(max_score)
            reach_fea_score[max_ind] = 0
            if max_ind in max_ind_list:
                break
            max_ind_list.append(max_ind)

        max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
        [self.reachable_feature.remove(v) for v in max_fea_id]
        [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]
        can_feature,can_item = self._get_cand()
        return self._get_state(), [can_feature,can_item], self._get_action_space()

    def _get_cand(self):
        if self.random_sample_feature:
            cand_feature = self._map_to_all_id(random.sample(self.reachable_feature, min(len(self.reachable_feature),self.cand_num)),'feature')
        else:
            cand_feature = self._map_to_all_id(self.reachable_feature[:self.cand_num],'feature')
        if self.random_sample_item:
            cand_item =  self._map_to_all_id(random.sample(self.cand_items, min(len(self.cand_items),self.cand_item_num)),'item')
        else:
            cand_item = self._map_to_all_id(self.cand_items[:self.cand_item_num],'item')
        cand = cand_feature + cand_item
        return cand_feature, cand_item
    
    def _get_action_space(self):
        action_space = [self._map_to_all_id(self.reachable_feature,'feature'), self._map_to_all_id(self.cand_items,'item')]
        return action_space

    def _get_state(self):
        """
        获取当前状态的函数

        返回值:
        - 包含当前状态信息的字典
        """

        # 如果数据集为'YELP_STAR'，则只考虑前5000个候选项
        if self.data_name in ['YELP_STAR']:
            self_cand_items = self.cand_items[:5000]
            set_cand_items = set(self_cand_items)
        else:
            self_cand_items = self.cand_items

        # 用户ID
        user = [self.user_id]

        # 当前节点集合，映射到对应的节点索引
        cur_node = [x + self.user_length + self.item_length for x in self.cur_node_set]

        # 候选项集合，映射到对应的节点索引
        cand_items = [x + self.user_length for x in self_cand_items]

        # 可达特征集合，映射到对应的节点索引
        reachable_feature = [x + self.user_length + self.item_length for x in self.reachable_feature]

        # 所有邻居节点的索引
        neighbors = cur_node + user + cand_items + reachable_feature

        # 构建索引映射字典，用于将节点值映射到索引
        idx = dict(enumerate(list(self.cur_node_set) + user + list(self_cand_items) + list(self.reachable_feature)))
        idx = {v: k for k, v in idx.items()}

        i = []  # 存储稀疏矩阵的行索引和列索引
        v = []  # 存储稀疏矩阵的值

        # 构建候选项与其特征之间的连接关系
        for item in self_cand_items:
            for fea in self.item_feature_pair[item]:
                i.append([idx[item], idx[fea]])
                i.append([idx[fea], idx[item]])
                v.append(1)
                v.append(1)

        user_idx = len(cur_node)  # 用户节点的索引
        cand_item_score = self.sigmoid(self.cand_item_score)

        # 构建用户与候选项之间的连接关系，带有评分
        for item, score in zip(self.cand_items, cand_item_score):
            if self.data_name in ['YELP_STAR']:
                if item not in set_cand_items:
                    continue
            i.append([user_idx, idx[item]])
            i.append([idx[item], user_idx])
            v.append(score)
            v.append(score)

        # 创建稀疏矩阵的表示
        i = torch.LongTensor(i)
        v = torch.FloatTensor(v)
        neighbors = torch.LongTensor(neighbors)
        adj = torch.sparse.FloatTensor(i.t(), v, torch.Size([len(neighbors), len(neighbors)]))

        # 构建状态字典，包括当前节点、邻居节点、邻接矩阵、被拒绝的特征和被拒绝的物品
        state = {
            'cur_node': cur_node,
            'neighbors': neighbors,
            'adj': adj,
            'rej_feature': self.user_rej_feature,
            'rej_item': self.rej_item,
            'user': self.user_id
        }

        return state
    def step(self, action, sorted_actions, embed=None):
        if embed is not None:
            self.ui_embeds = embed[:self.user_length + self.item_length]
            self.feature_emb = embed[self.user_length + self.item_length:]

        done = 0  # 是否完成标志，0 表示未完成
        print('---------------step:{}-------------'.format(self.cur_conver_step))
        action_type = 0 # 0 代表推荐属性，1代表推荐物品
        sorted_actions_feature = sorted_actions[0]
        sorted_actions_item = sorted_actions[1]
        recom_items = []
        asked_feature = []
        if self.cur_conver_step == self.max_turn:
            reward = self.reward_dict['until_T']  # 达到最大轮次的惩罚奖励
            self.conver_his[self.cur_conver_step - 1] = self.history_dict['until_T']  # 更新对话历史记录
            print('--> Maximum number of turns reached !')
            done = 1  # 设置完成标志为 1，表示对话已完成
            action_type = 1
        elif action >= self.user_length + self.item_length:  # 询问特征
            action_type = 0
            # 创建一个空字典来存储分数
            score = {}

            # 初始化最大分数及其对应的ID
            max_score_id = None
            max_score = -999

            # 创建一个字典来存储大特征与小特征的映射关系
            large_small = {}

            # 遍历一个范围，范围的长度等于sorted_actions的长度
            for i in range(len(sorted_actions_feature)):
                # 获取当前迭代的动作
                act = sorted_actions_feature[i]

                # 如果动作的值小于用户长度和物品长度之和，跳过当前迭代
                if act < self.user_length + self.item_length:
                    continue

                # 将动作映射为原始特征ID
                small_fea = self._map_to_old_id(act)

                # 获取大特征ID
                large = self.small_feature_to_large[small_fea]

                # 如果大特征ID已经在分数字典中存在
                if large in score:
                    # 增加分数，分数计算方式为1 / (i + 1)
                    score[large] += 1 / (i + 1)

                    # 将小特征ID添加到对应的大特征的列表中
                    large_small[large].append(small_fea)
                else:
                    # 如果大特征ID不在分数字典中，初始化分数为0.0
                    score[large] = 0.0

                    # 创建大特征对应的小特征列表，并将当前小特征ID添加到列表中
                    large_small[large] = []
                    score[large] += 1 / (i + 1)
                    large_small[large].append(small_fea)

                # 如果当前大特征的分数大于最大分数
                if score[large] > max_score:
                    # 更新最大分数和对应的大特征ID
                    max_score = score[large]
                    max_score_id = large

            asked_feature = large_small[max_score_id][:self.choice_num]  # 根据得分获取要询问的特征

            print('-->action: ask features {}, max entropy feature {}'.format(asked_feature,
                                                                              self.reachable_feature[:self.cand_num]))
            reward, done, acc_rej = self._ask_update(asked_feature)  # 更新用户的资料：user_acc_feature 和 user_rej_feature
            self._update_cand_items(asked_feature, acc_rej)  # 更新候选物品列表
        else:  # 推荐物品
            action_type = 1
            # ===================== rec update=========
            recom_items = []
            recom_items_gnn_id = []
            for act in sorted_actions_item:
                if act < self.user_length + self.item_length:
                    recom_items_gnn_id.append(act)
                    recom_items.append(self._map_to_old_id(act))  # 将推荐动作映射为原始物品ID
                    if len(recom_items) == self.rec_num:
                        break
            reward, done = self._recommend_update(recom_items)  # 更新推荐状态
            # ========================================
            if reward > 0:
                print('-->Recommend successfully!')
            else:
                self.rej_item += recom_items_gnn_id
                print('-->Recommend fail !')

        self._updata_reachable_feature()  # 更新用户的可达特征

        print('reachable_feature num: {}'.format(len(self.reachable_feature)))
        print('cand_item num: {}'.format(len(self.cand_items)))

        self._update_feature_entropy()  # 更新特征熵
        if len(self.reachable_feature) != 0:  # 如果可达特征不为空
            reach_fea_score = self._feature_score()  # 计算特征得分

            max_ind_list = []  # 用于存储最大得分的索引列表

            # 遍历获取最大得分的特征索引
            for k in range(self.cand_num):
                max_score = max(reach_fea_score)  # 获取可达特征得分中的最大值
                max_ind = reach_fea_score.index(max_score)  # 获取最大得分对应的特征索引
                reach_fea_score[max_ind] = 0  # 将最大得分置零，以便下一次找下一个最大得分
                if max_ind in max_ind_list:
                    break  # 如果这个最大得分的索引已经在列表中，退出循环
                max_ind_list.append(max_ind)  # 将最大得分的索引添加到列表中

            max_fea_id = [self.reachable_feature[i] for i in max_ind_list]  # 根据最大得分的索引获取最大得分的特征ID列表
            [self.reachable_feature.remove(v) for v in max_fea_id]  # 从可达特征列表中移除最大得分的特征
            [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]  # 将最大得分的特征按逆序插入到可达特征列表的开头

        self.cur_conver_step += 1  # 当前对话步数加一
        can_feature,can_item = self._get_cand()
        return self._get_state(), [can_feature,can_item], self._get_action_space(), reward, done, action_type, recom_items, asked_feature  # 返回状态、候选物品、动作空间、奖励和完成标志

    def _updata_reachable_feature(self):
        # 初始化一个列表，用于存储下一个可达特征
        next_reachable_feature = []

        # 初始化一个字典，用于存储物品和其关联特征的对应关系
        reachable_item_feature_pair = {}

        # 遍历候选物品
        for cand in self.cand_items:
            # 获取候选物品属于的特征列表
            fea_belong_items = list(self.kg.G['item'][cand]['belong_to'])  # 属于候选物品的特征列表
            # 将特征列表中的特征添加到下一个可达特征列表中
            next_reachable_feature.extend(fea_belong_items)
            # 计算物品和特征之间的关系，排除用户拒绝的特征
            reachable_item_feature_pair[cand] = list(set(fea_belong_items) - set(self.user_rej_feature))
            # 将下一个可达特征列表转换为不重复的特征集合
            next_reachable_feature = list(set(next_reachable_feature))

        # 计算最终的可达特征，排除用户已接受和拒绝的特征
        self.reachable_feature = list(
            set(next_reachable_feature) - set(self.user_acc_feature) - set(self.user_rej_feature))

        # 将物品和特征的对应关系保存为成员变量
        self.item_feature_pair = reachable_item_feature_pair

    def _feature_score(self):
        reach_fea_score = []  # 用于存储特征的得分

        if self.fea_score == "entropy":
            # 如果特征得分方式为 "entropy"
            for feature_id in self.reachable_feature:

                # score = self.attr_ent[feature_id]
                # reach_fea_score.append(score)

                feature_embed = self.feature_emb[feature_id]  # 获取特征的嵌入向量
                score = 0  # 初始化得分

                # 计算用户嵌入向量与特征嵌入向量的内积，并加到得分中
                score += np.inner(np.array(self.user_embed), feature_embed)

                # 获取用户接受的特征的嵌入向量
                prefer_embed = self.feature_emb[self.user_acc_feature, :]  # np.array (x*64)

                # 对于每个用户接受的特征，计算其嵌入向量与特征嵌入向量的内积，并加到得分中
                for i in range(len(self.user_acc_feature)):
                    score += np.inner(prefer_embed[i], feature_embed)

                # 如果特征被用户拒绝，使用sigmoid函数进行转换后减去得分（惩罚）
                if feature_id in self.user_rej_feature:
                    score -= self.sigmoid([feature_embed, feature_embed])[0]

                reach_fea_score.append(score)  # 将得分添加到特征得分列表中
        else:
            # 如果特征得分方式不是 "entropy"
            for feature_id in self.reachable_feature:
                fea_embed = self.feature_emb[feature_id]  # 获取特征的嵌入向量
                score = 0  # 初始化得分

                # 计算用户嵌入向量与特征嵌入向量的内积，并加到得分中
                score += np.inner(np.array(self.user_embed), fea_embed)

                # 获取用户接受和拒绝的特征的嵌入向量
                prefer_embed = self.feature_emb[self.user_acc_feature, :]  # np.array (x*64)
                rej_embed = self.feature_emb[self.user_rej_feature, :]  # np.array (x*64)

                # 对于每个用户接受的特征，计算其嵌入向量与特征嵌入向量的内积，并加到得分中
                for i in range(len(self.user_acc_feature)):
                    score += np.inner(prefer_embed[i], fea_embed)

                # 对于每个用户拒绝的特征，使用sigmoid函数进行转换后减去得分（惩罚）
                for i in range(len(self.user_rej_feature)):
                    score -= self.sigmoid([np.inner(rej_embed[i], fea_embed)])

                reach_fea_score.append(score)  # 将得分添加到特征得分列表中

        return reach_fea_score  # 返回特征的得分列表

    def _item_score(self):
        cand_item_score = []  # 用于存储候选物品的得分

        # 遍历每个候选物品
        for item_id in self.cand_items:
            item_embed = self.ui_embeds[self.user_length + item_id]  # 获取物品的嵌入向量
            score = 0  # 初始化得分

            # 计算用户嵌入向量与物品嵌入向量的内积，并加到得分中
            score += np.inner(np.array(self.user_embed), item_embed)

            # 获取用户接受的特征的嵌入向量
            prefer_embed = self.feature_emb[self.user_acc_feature, :]  # np.array (x*64)

            # 获取用户拒绝的特征中与当前物品相关的特征
            unprefer_feature = list(set(self.user_rej_feature) & set(self.kg.G['item'][item_id]['belong_to']))

            # 获取拒绝特征的嵌入向量
            unprefer_embed = self.feature_emb[unprefer_feature, :]  # np.array (x*64)

            # 对于每个用户接受的特征，计算其嵌入向量与物品嵌入向量的内积，并加到得分中
            for i in range(len(self.user_acc_feature)):
                score += np.inner(prefer_embed[i], item_embed)

            # 对于每个用户拒绝的与物品相关的特征，计算其嵌入向量与物品嵌入向量的内积，
            # 并使用sigmoid函数进行转换后减去得分（惩罚）
            for i in range(len(unprefer_feature)):
                score -= self.sigmoid([np.inner(unprefer_embed[i], item_embed)])[0]

            cand_item_score.append(score)  # 将得分添加到候选物品得分列表中

        return cand_item_score  # 返回候选物品的得分列表

    def _ask_update(self, asked_features):
        '''
        :return: 奖励值 (reward)，接受的特征 (acc_feature)，拒绝的特征 (rej_feature)
        '''
        done = 0
        # TODO datafram! groundTruth == target_item features

        # 初始化奖励值
        reward = 0

        # 初始化接受或拒绝特征的标志位
        acc_rej = False

        # 将当前会话步骤的历史失败记录保存到conver_his中
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']

        # 遍历被询问的特征列表
        for asked_feature in asked_features:
            # 如果被询问的特征在真实特征中存在
            if asked_feature in self.feature_groundtrue:
                acc_rej = True

                # 将被接受的特征添加到用户已接受的特征列表中
                self.user_acc_feature.append(asked_feature)

                # 将被接受的特征添加到当前节点集合中
                self.cur_node_set.append(asked_feature)

                # 奖励值增加，使用奖励字典中的'ask_suc'值
                reward += self.reward_dict['ask_suc']

                # 更新conver_his中当前会话步骤的历史成功记录
                self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']

            # 如果被询问的特征不在真实特征中
            else:
                # 将被拒绝的特征添加到用户已拒绝的特征列表中
                self.user_rej_feature.append(asked_feature)

                # 奖励值增加，使用奖励字典中的'ask_fail'值
                reward += self.reward_dict['ask_fail']

        # 如果候选项为空
        if self.cand_items == []:
            done = 1

            # 设置奖励值为候选项为空时的奖励值，使用奖励字典中的'cand_none'值
            reward = self.reward_dict['cand_none']

        # 返回奖励值、done标志位以及接受或拒绝特征的标志位
        return reward, done, acc_rej

    def _update_cand_items(self, asked_feature, acc_rej):
        acc_item = []  # 用于存储根据接受的特征更新后的候选物品
        rej_item = []  # 用于存储根据拒绝的特征更新后的候选物品

        # 对于每个被询问的特征
        for fea in asked_feature:
            if fea in self.feature_groundtrue:  # 如果特征被用户接受
                print('=== 询问接受的特征 {}: 更新候选物品'.format(fea))
                feature_items = self.kg.G['feature'][fea]['belong_to']  # 获取属于这个特征的物品
                cand_items = set(self.cand_items) & set(feature_items)  # 取候选物品和特征相关物品的交集
                acc_item += list(cand_items)  # 将交集中的物品添加到接受物品列表中
            else:  # 如果特征被用户拒绝
                feature_items = self.kg.G['feature'][fea]['belong_to']  # 获取属于这个特征的物品
                cand_items = set(self.cand_items) & set(feature_items)  # 取候选物品和特征相关物品的交集
                rej_item += list(cand_items)  # 将交集中的物品添加到拒绝物品列表中
                print('=== 询问拒绝的特征 {}: 更新候选物品'.format(fea))

        # 根据用户接受和拒绝的特征更新候选物品
        if len(acc_item) == 0:
            # 如果接受物品为空，候选物品为候选物品减去拒绝物品的差集
            cand_items = list(set(self.cand_items) - (set(self.cand_items) & set(rej_item)))
        else:
            # 否则，候选物品为接受物品减去接受物品和拒绝物品的交集
            cand_items = list(set(acc_item) - (set(acc_item) & set(rej_item)))

        if len(cand_items) != 0:
            self.cand_items = cand_items  # 更新候选物品

        # 为候选物品计算得分并排序
        cand_item_score = self._item_score()  # 获取候选物品的得分
        item_score_tuple = list(zip(self.cand_items, cand_item_score))  # 将物品和得分打包成元组
        sort_tuple = sorted(item_score_tuple, key=lambda x: x[1], reverse=True)  # 根据得分降序排列
        self.cand_items, self.cand_item_score = zip(*sort_tuple)  # 更新候选物品和得分

    def _recommend_update(self, recom_items):
        # 打印动作：推荐物品
        print('-->动作：推荐物品')

        # 打印推荐物品中在候选项中不存在的物品
        print(set(recom_items) - set(self.cand_items[: self.rec_num]))

        # 将候选物品和它们的得分列表转换为列表形式
        self.cand_items = list(self.cand_items)
        self.cand_item_score = list(self.cand_item_score)

        # 初始化命中标志为False
        hit = False

        # 检查目标物品是否在推荐物品中
        for i in self.target_item:
            if i in recom_items:
                hit = True
                break

        # 如果有命中
        if hit:
            # 设置奖励为成功推荐时的奖励值
            reward = self.reward_dict['rec_suc']

            # 更新会话历史向量中当前会话步骤的历史记录为成功推荐
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_suc']

            # 创建一个临时分数列表，用于存储推荐物品的得分
            tmp_score = []

            # 遍历推荐物品
            for item in recom_items:
                # 获取物品在候选物品中的索引
                idx = self.cand_items.index(item)

                # 将物品的得分添加到临时分数列表中
                tmp_score.append(self.cand_item_score[idx])

            # 更新候选物品列表和它们的得分列表为推荐物品及其得分
            self.cand_items = recom_items
            self.cand_item_score = tmp_score

            # 设置完成标志为命中的物品在推荐列表中的索引加1
            done = recom_items.index(i) + 1
        else:
            # 设置奖励为推荐失败时的奖励值
            reward = self.reward_dict['rec_fail']

            # 更新会话历史向量中当前会话步骤的历史记录为推荐失败
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_fail']

            # 如果候选物品数量大于推荐数量
            if len(self.cand_items) > self.rec_num:
                # 遍历推荐物品
                for item in recom_items:
                    # 删除物品和其特征对应关系
                    del self.item_feature_pair[item]

                    # 获取物品在候选物品中的索引
                    idx = self.cand_items.index(item)

                    # 从候选物品列表和得分列表中移除该物品
                    self.cand_items.pop(idx)
                    self.cand_item_score.pop(idx)

            # 设置完成标志为0
            done = 0

        # 返回奖励值和完成标志
        return reward, done

    def _update_feature_entropy(self):
        # 如果计算特征熵的方式是'entropy'
        if self.ent_way == 'entropy':
            # 初始化候选物品中包含的特征列表
            cand_items_fea_list = []

            # 遍历候选物品，将它们包含的特征添加到列表中
            for item_id in self.cand_items:
                cand_items_fea_list.append(list(self.kg.G['item'][item_id]['belong_to']))

            # 将嵌套的特征列表扁平化
            cand_items_fea_list = list(_flatten(cand_items_fea_list))

            # 统计特征的出现次数
            self.attr_count_dict = dict(Counter(cand_items_fea_list))

            # 重置特征熵列表
            self.attr_ent = [0] * self.attr_state_num

            # 找到可查询特征中既在候选物品特征中又在特征计数字典中的特征
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))

            # 遍历这些可查询特征
            for fea_id in real_ask_able:
                # 计算特征的两类概率
                p1 = float(self.attr_count_dict[fea_id]) / len(self.cand_items)
                p2 = 1.0 - p1

                # 计算特征的熵值
                if p1 == 1:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent

        # 如果计算特征熵的方式是'weight_entropy'
        elif self.ent_way == 'weight_entropy':
            # 初始化候选物品中包含的特征列表
            cand_items_fea_list = []

            # 初始化特征计数字典
            self.attr_count_dict = {}

            # 使用Sigmoid函数将候选物品的得分进行转换
            cand_item_score_sig = self.sigmoid(self.cand_item_score)

            # 遍历候选物品
            for score_ind, item_id in enumerate(self.cand_items):
                # 获取候选物品包含的特征列表
                cand_items_fea_list = list(self.kg.G['item'][item_id]['belong_to'])

                # 遍历特征列表
                for fea_id in cand_items_fea_list:
                    if self.attr_count_dict.get(fea_id) == None:
                        self.attr_count_dict[fea_id] = 0

                    # 基于物品得分更新特征的权重
                    self.attr_count_dict[fea_id] += cand_item_score_sig[score_ind]

            # 重置特征熵列表
            self.attr_ent = [0] * self.attr_state_num

            # 找到可查询特征中既在候选物品特征中又在特征计数字典中的特征
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))

            # 计算所有候选物品得分的Sigmoid值的总和
            sum_score_sig = sum(cand_item_score_sig)

            # 遍历这些可查询特征
            for fea_id in real_ask_able:
                # 计算特征的两类概率
                p1 = float(self.attr_count_dict[fea_id]) / sum_score_sig
                p2 = 1.0 - p1

                # 计算特征的熵值
                if p1 == 1 or p1 <= 0:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent

    def sigmoid(self, x_list):
        x_np = np.array(x_list)
        s = 1 / (1 + np.exp(-x_np))
        return s.tolist()

    def _map_to_all_id(self, x_list, old_type):
        if old_type == 'item':
            return [x + self.user_length for x in x_list]
        elif old_type == 'feature':
            return [x + self.user_length + self.item_length for x in x_list]
        else:
            return x_list

    def _map_to_old_id(self, x):
        if x >= self.user_length + self.item_length:
            x -= (self.user_length + self.item_length)
        elif x >= self.user_length:
            x -= self.user_length
        return x

