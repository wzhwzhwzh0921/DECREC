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


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=100):
        super(DQN, self).__init__()
        # 初始化函数，定义网络的结构和参数
        # V(s)
        self.fc2_value = nn.Linear(hidden_size, hidden_size)  # 值函数网络的第二个全连接层
        self.out_value = nn.Linear(hidden_size, 1)  # 值函数网络的输出层，输出状态的值

        # Q(s,a)
        self.fc2_advantage = nn.Linear(hidden_size + action_size, hidden_size)  # 动作优势函数网络的第二个全连接层
        self.out_advantage = nn.Linear(hidden_size, 1)  # 动作优势函数网络的输出层，输出每个动作的优势值

    def forward(self, X, z, choose_action=True, Op=False):
        # 前向传播函数，定义了数据从输入到输出的流程

        m = []  # 用于存储每个样本的 Q(s,a) 值

        for x in X:
            # 计算值函数的值
            value = self.out_value(F.relu(self.fc2_value(x))).squeeze(dim=2)  # 值函数网络的计算结果 [N*1*1]

            if choose_action:
                x = x.repeat(1, z.size(1), 1)  # 如果选择动作，将状态扩展到和动作数量一致
            state_cat_action = torch.cat((x, z), dim=2)  # 将状态和动作连接在一起，形成输入特征
            advantage = self.out_advantage(F.relu(self.fc2_advantage(state_cat_action))).squeeze(
                dim=2)  # 动作优势函数的计算结果 [N*K]

            if choose_action:
                qsa = advantage + value - advantage.mean(dim=1, keepdim=True)  # 计算 Q(s,a) 值，减去动作优势的均值
            else:
                qsa = advantage + value  # 如果不选择动作，直接计算 Q(s) 值
            m.append(qsa)

        qsa, _ = torch.max(torch.stack(m, dim=0).squeeze(0), dim=0)  # 选择每个状态的最大 Q(s,a) 值

        return qsa  # 返回 Q(s,a) 值

