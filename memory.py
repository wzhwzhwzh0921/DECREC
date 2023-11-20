import math
import random
import numpy as np
import os
import sys
from tqdm import tqdm
# sys.path.append('..')

from collections import namedtuple
import argparse
from itertools import count, chain
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from sum_tree import SumTree
from agent import Agent


Transition = namedtuple('Transition',
                        ('state', 'action', 'sorted_actions_feature', 'sorted_actions_item', 'next_state', 'reward', 'next_cand'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayMemoryPER(object):
    # 使用SumTree作为储存结构，存储为 ( s, a, r, s_ )
    def __init__(self, capacity, a=0.6, e=0.01):
        self.tree = SumTree(capacity)  # 创建一个容量为capacity的SumTree
        self.capacity = capacity  # 内存的容量
        self.prio_max = 0.1  # 初始化优先级最大值
        self.a = a  # 控制优先级分布的参数
        self.e = e  # 用于避免优先级为0的情况
        self.beta = 0.4  # IS权重的初始值
        self.beta_increment_per_sampling = 0.001  # 每次采样后增加的IS权重

    def push(self, *args):
        data = Transition(*args)  # 将输入的数据转换为Transition对象
        p = (np.abs(self.prio_max) + self.e) ** self.a  # 计算优先级，采用proportional priority
        self.tree.add(p, data)  # 将数据和优先级添加到SumTree中

    def sample(self, batch_size):
        batch_data = []  # 用于存储采样的数据
        idxs = []  # 存储采样的索引
        segment = self.tree.total() / batch_size  # 计算每个batch的段大小
        priorities = []  # 存储采样的优先级

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)  # 在当前段内随机选择一个数
            idx, p, data = self.tree.get(s)  # 根据选择的数获取对应的数据和优先级

            batch_data.append(data)  # 将数据添加到batch_data中
            priorities.append(p)  # 将优先级添加到priorities中
            idxs.append(idx)  # 将索引添加到idxs中

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # 更新IS权重的增量

        sampling_probabilities = priorities / self.tree.total()  # 计算每个样本的采样概率
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)  # 计算IS权重
        is_weight /= is_weight.max()  # 归一化IS权重

        return idxs, batch_data, is_weight  # 返回采样的索引、数据和IS权重

    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))  # 更新优先级最大值
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a  # 计算更新后的优先级
            self.tree.update(idx, p)  # 更新SumTree中的优先级

    def __len__(self):
        return self.tree.n_entries  # 返回SumTree中存储的数据数量