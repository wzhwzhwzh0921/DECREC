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

#TODO select env
from RL.env_multi_choice_question import MultiChoiceRecommendEnv
from RL.RL_evaluate import dqn_evaluate
from multi_interest import GraphEncoder
import time
import warnings
import ipdb
from dqn import DQN

Transition = namedtuple('Transition',
                        ('state', 'action', 'sorted_actions_feature', 'sorted_actions_item', 'next_state', 'reward', 'next_cand'))
class Agent(object):
    def __init__(self, device, memory_feature_longtail, memory_item_longtail, memory_feature_head, memory_item_head, state_size, action_size, hidden_size, gcn_net, learning_rate, l2_norm, PADDING_ID, EPS_START = 0.9, EPS_END = 0.1, EPS_DECAY = 0.0001, tau=0.01):
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.steps_done = 0
        self.device = device
        self.gcn_net = gcn_net
        self.value_net = DQN(state_size, action_size, hidden_size).to(device)
        self.value_net_feature = DQN(state_size, action_size, hidden_size).to(device)
        self.value_net_item = DQN(state_size, action_size, hidden_size).to(device)
        self.value_net_feature.load_state_dict(self.value_net_item.state_dict())

        self.target_net_feature = DQN(state_size, action_size, hidden_size).to(device)
        self.target_net_item = DQN(state_size, action_size, hidden_size).to(device)
        self.target_net_feature.load_state_dict(self.value_net_feature.state_dict())
        self.target_net_item.load_state_dict(self.value_net_item.state_dict())
        self.target_net_feature.eval()
        self.target_net_item.eval()

        self.policy_net = DQN(state_size, action_size, hidden_size).to(device)
        self.rec_or_req = nn.Embedding(2, action_size).to(device)

        self.long_tail_threshold = 30
        self.freq_feature = {}
        self.freq_item = {}
        self.softmax = nn.Softmax()
        self.optimizer = optim.Adam(chain(self.gcn_net.parameters(), self.policy_net.parameters(), self.value_net_feature.parameters(), self.value_net_item.parameters()), lr=learning_rate, weight_decay=l2_norm)
        self.memory_feature_longtail = memory_feature_longtail
        self.memory_item_longtail = memory_item_longtail
        self.memory_feature_head = memory_feature_head
        self.memory_item_head = memory_item_head
        self.loss_func = nn.MSELoss()
        self.PADDING_ID = PADDING_ID
        self.tau = tau
    def select_action(self, state, cand1, action_space, is_test=False, is_last_turn=False):
        # 将状态传入 GCN 网络获取状态的嵌入表示
        state_emb = self.gcn_net([state])
        cand_feature = torch.LongTensor([cand1[0]]).to(self.device)
        cand_item = torch.LongTensor([cand1[1]]).to(self.device)
        # 将候选动作转换为张量，并将其移动到设备上

        if cand_feature.size()[1]==0:
            return None, None, None
        # 获取候选动作的嵌入表示
        cand_feature_emb = self.gcn_net.embedding(cand_feature)
        cand_item_emb = self.gcn_net.embedding(cand_item)


        # 生成一个随机样本以确定是否进行随机动作
        sample = random.random()
        # 计算当前的 ε 贪心阈值，根据当前步数进行衰减
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        # 判断是否选择随机动作还是贪心策略
        if sample > eps_threshold:
            if is_test and (len(action_space[1]) <= 20 or is_last_turn):
                # 在测试模式下，如果候选动作数不超过20个或者是对话的最后一轮，则选择第一个动作
                return torch.tensor(action_space[1][0], device=self.device, dtype=torch.long), action_space, state_emb
            with torch.no_grad():
                # 在策略网络中计算动作值
                #actions_value = self.policy_net(state_emb, cand_emb)
                actions_feature_value = self.value_net_feature(state_emb, cand_feature_emb)
                actions_item_value =self.value_net_item(state_emb, cand_item_emb)
                # if actions_feature_value.max() > actions_item_value.max():
                #     action_type = 0
                # else:
                #     action_type = 1
                policy = self.policy_net(state_emb, torch.cat((cand_feature_emb, cand_item_emb),dim=1))
                policy_feature = policy[:,:len(cand_feature[0])].mean(dim=1)
                policy_item = policy[:, len(cand_feature[0]):].mean(dim=1)
                policy_probs = self.softmax(torch.cat((policy_feature, policy_item)))
                action_type = torch.multinomial(policy_probs, 1)

                if action_type == 0:
                    if cand_feature.size()[1]!= 0:
                        action = cand_feature[0][actions_feature_value.argmax().item()]
                    # 根据动作值排序获取动作的降序排列
                    else:
                        action = cand_item[0][actions_item_value.argmax().item()]
                else:
                    if cand_item.size()[1]!= 0:
                        action = cand_item[0][actions_item_value.argmax().item()]
                    else:
                        action = cand_feature[0][actions_feature_value.argmax().item()]
                    # 根据动作值排序获取动作的降序排列
                sorted_actions_feature = cand_feature[0][actions_feature_value.sort(1, True)[1].tolist()].tolist()
                sorted_actions_item = cand_item[0][actions_item_value.sort(1, True)[1].tolist()].tolist()
                sorted_actions = [sorted_actions_feature, sorted_actions_item]

                return action, sorted_actions, state_emb
        else:
            # 对候选动作进行随机洗牌，并选择第一个动作
            shuffled_cand = action_space[0] + action_space[1]
            random.shuffle(shuffled_cand)
            return torch.tensor(shuffled_cand[0], device=self.device, dtype=torch.long), [action_space[0], action_space[1]], state_emb

    def update_target_model(self):
        #soft assign
        for target_param, param in zip(self.target_net_feature.parameters(), self.value_net_feature.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))
        for target_param, param in zip(self.target_net_item.parameters(), self.value_net_item.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

    def optimize_model(self, BATCH_SIZE, GAMMA, data_type):
        # 如果存储的记忆不足以构成一个批次，就不进行优化
        if data_type == 'feature':
            if len(self.memory_feature_longtail) < BATCH_SIZE:
                return None, None
            if len(self.memory_feature_head) < BATCH_SIZE :
                idxs, transitions, is_weights = self.memory_feature_longtail.sample(BATCH_SIZE)
                split_flag = 0
                head_num = 0
                longtail_num = BATCH_SIZE
            elif len(self.memory_feature_longtail) - len(self.memory_feature_head) < BATCH_SIZE:
                return None, None
            else:
                head_num = int((len(self.memory_feature_head)/len(self.memory_feature_longtail) * BATCH_SIZE))
                longtail_num = BATCH_SIZE - head_num
                idxs_head, transitions_head, is_weights_head = self.memory_feature_head.sample(head_num)
                idxs_longtail, transitions_longtail, is_weights_longtail = self.memory_feature_longtail.sample(longtail_num)
                idxs = idxs_head + idxs_longtail
                transitions = transitions_head + transitions_longtail
                is_weights = np.hstack((is_weights_head, is_weights_longtail))
                split_flag = 1

        else: #data_type == 'item':
            if len(self.memory_item_longtail) < BATCH_SIZE:
                return None, None
            if len(self.memory_item_head) < BATCH_SIZE :
                idxs, transitions, is_weights = self.memory_item_longtail.sample(BATCH_SIZE)
                split_flag = 0
                head_num = 0
                longtail_num = BATCH_SIZE

            elif len(self.memory_item_longtail) - len(self.memory_item_head) < BATCH_SIZE:
                return None, None
            else:
                head_num = int((len(self.memory_item_head)/len(self.memory_item_longtail) * BATCH_SIZE))
                longtail_num = BATCH_SIZE - head_num
                idxs_head, transitions_head, is_weights_head = self.memory_item_head.sample(head_num)
                idxs_longtail, transitions_longtail, is_weights_longtail = self.memory_item_longtail.sample(longtail_num)
                idxs = idxs_head + idxs_longtail
                transitions = transitions_head + transitions_longtail
                is_weights = np.hstack((is_weights_head, is_weights_longtail))
                split_flag = 1

        # 更新目标模型的参数
        self.update_target_model()
        batch = Transition(*zip(*transitions))

        #使用GCN网络计算当前状态的嵌入向量
        state_emb_batch = self.gcn_net(list(batch.state))

        # 转换动作数据为张量并移至CPU，然后根据需要对其进行格式处理
        action_batch = torch.stack(batch.action).detach().cpu()
        action_batch = torch.LongTensor(np.array(action_batch).astype(int).reshape(-1, 1)).to(self.device)  # [N*1]

        # 使用GCN网络计算动作的嵌入向量
        action_emb_batch = self.gcn_net.embedding(action_batch)

        # 转换奖励数据为张量并移至CPU，然后根据需要对其进行格式处理
        reward_batch = torch.stack(batch.reward).detach().cpu()
        reward_batch = torch.FloatTensor(np.array(reward_batch).astype(float).reshape(-1, 1)).to(self.device)

        # 创建一个掩码来标识哪些样本有非空的下一个状态
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.uint8)

        # 分别收集非空的下一个状态和候选动作
        n_states = []
        n_cands_feature = []
        n_cands_item = []
        cands_feature = []
        cands_item = []
        for s, c, m, n in zip(batch.next_state, batch.next_cand, batch.sorted_actions_feature, batch.sorted_actions_item):
            if s is not None:
                n_states.append(s)
                n_cands_feature.append(c[0])
                n_cands_item.append((c[1]))
            cands_feature.append(m)
            cands_item.append(n)
        # 使用GCN网络计算下一个状态的嵌入向量
        next_state_emb_batch = self.gcn_net(n_states)

        # 使用GCN网络计算下一个候选动作的嵌入向量
        next_cand_feature_batch, next_cand_feature_mask_batch = self.padding(n_cands_feature)

        next_cand_feature_emb_batch = self.gcn_net.embedding(next_cand_feature_batch)

        # 使用GCN网络计算下一个候选动作的嵌入向量
        next_cand_item_batch, next_cand_item_mask_batch = self.padding(n_cands_item)

        next_cand_item_emb_batch = self.gcn_net.embedding(next_cand_item_batch)

        # 使用当前策略网络计算当前状态和动作对应的Q值

        if data_type == 'feature':
            q_eval = self.value_net_feature(state_emb_batch, action_emb_batch, choose_action=False)
        else:
            q_eval = self.value_net_item(state_emb_batch, action_emb_batch, choose_action=False)

        # 使用双重DQN算法计算目标Q值，选择下一个候选动作中具有最高Q值的动作
        best_actions_feature = torch.gather(input=next_cand_feature_batch, dim=1,
                                            index=self.value_net_feature(next_state_emb_batch,
                                                                         next_cand_feature_emb_batch,
                                                                         Op=True).argmax(
                                                dim=1).view(len(n_states), 1).to(self.device))
        best_actions_item = torch.gather(input=next_cand_item_batch, dim=1,
                                         index=self.value_net_item(next_state_emb_batch, next_cand_item_emb_batch,
                                                                   Op=True).argmax(
                                             dim=1).view(len(n_states), 1).to(self.device))

        policy_temp_feature = self.policy_net(next_state_emb_batch, next_cand_feature_emb_batch).mean(dim=1).unsqueeze(dim=1)
        policy_temp_item = self.policy_net(next_state_emb_batch, next_cand_item_emb_batch).mean(dim=1).unsqueeze(dim=1)
        policy_probs = self.softmax(torch.cat((policy_temp_feature,policy_temp_item),dim=1))
        best_actions_feature_value = self.target_net_feature(next_state_emb_batch,
                                                             self.gcn_net.embedding(best_actions_feature),
                                                             choose_action=False).detach()
        best_actions_item_value = self.target_net_item(next_state_emb_batch, self.gcn_net.embedding(best_actions_item),
                                                       choose_action=False).detach()

        best_action_value = torch.cat((best_actions_feature_value, best_actions_item_value),dim=1)
        best_action_value = torch.sum(policy_probs * best_action_value,dim=1, keepdim=True)
        q_target = torch.zeros((BATCH_SIZE, 1), device=self.device)
        q_target[non_final_mask] = best_action_value

        # 计算更新后的目标Q值，包括奖励和折扣因子的影响
        q_target = reward_batch + GAMMA * q_target

        # 计算TD误差，并将其用于更新存储记忆的优先级
        errors = (q_eval - q_target).detach().cpu().squeeze().tolist()
        if data_type == 'feature':
            if split_flag == 1:
                self.memory_feature_head.update(idxs[:head_num], errors[:head_num])
                self.memory_feature_longtail.update(idxs[head_num:], errors[head_num:])
            else:
                self.memory_feature_longtail.update(idxs, errors)
        else:
            if split_flag == 1:
                self.memory_item_head.update(idxs[:head_num], errors[:head_num])
                self.memory_item_longtail.update(idxs[head_num:], errors[head_num:])
            else:
                self.memory_item_longtail.update(idxs, errors)

        if data_type == 'feature':
            idx_move = []
            for idx, state, action, sorted_actions_feature, sorted_actions_item, next_state, reward, next_cand in zip(
                    idxs[head_num:], batch.state[head_num:], batch.action[head_num:],
                    batch.sorted_actions_feature[head_num:], batch.sorted_actions_item[head_num:],
                    batch.next_state[head_num:], batch.reward[head_num:], batch.next_cand[head_num:]):
                if action in self.freq_feature.keys():
                    self.freq_feature[action] += 1
                else:
                    self.freq_feature[action] = 1
                if self.freq_feature[action] > self.long_tail_threshold:
                    self.memory_feature_head.push(state, action, sorted_actions_feature, sorted_actions_item,
                                                  next_state, reward, next_cand)
                    idx_move.append(idx)
            if len(idx_move) != 0:
                self.memory_feature_longtail.update(idx_move, [0.00001]*len(idx_move))

        else:
            idx_move = []
            for idx, state, action, sorted_actions_feature, sorted_actions_item, next_state, reward, next_cand in zip(
                    idxs[head_num:], batch.state[head_num:], batch.action[head_num:],
                    batch.sorted_actions_feature[head_num:], batch.sorted_actions_item[head_num:],
                    batch.next_state[head_num:], batch.reward[head_num:], batch.next_cand[head_num:]):
                if action in self.freq_item.keys():
                    self.freq_item[action] += 1
                else:
                    self.freq_item[action] = 1
                if self.freq_item[action] > self.long_tail_threshold:
                    self.memory_item_head.push(state, action, sorted_actions_feature, sorted_actions_item,
                                                 next_state, reward, next_cand)
                    idx_move.append(idx)
            if len(idx_move) != 0:
                self.memory_item_longtail.update(idx_move, [0.00001] * len(idx_move))


        # 计算损失函数，乘以重要性权重，然后进行优化
        loss_value = (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(q_eval, q_target)).mean()

        sorted_actions_feature_batch, sorted_actions_feature_mask_batch = self.padding(cands_feature)
        sorted_actions_feature_emb_batch = self.gcn_net.embedding(sorted_actions_feature_batch)

        sorted_actions_item_batch, sorted_actions_item_mask_batch = self.padding(cands_item)
        sorted_actions_item_emb_batch = self.gcn_net.embedding(sorted_actions_item_batch)

        advantage_now = torch.sum(self.policy_net(state_emb_batch,
                                  torch.cat((sorted_actions_feature_emb_batch, sorted_actions_item_emb_batch),
                                            dim=1)) * torch.cat(
            (sorted_actions_feature_mask_batch, sorted_actions_item_mask_batch), dim=1), dim=1, keepdim=True) / (
                    0.01 + torch.sum(sorted_actions_feature_mask_batch, dim=1, keepdim=True) + torch.sum(
                sorted_actions_item_mask_batch, dim=1, keepdim=True))
        advantage_next = torch.sum(self.policy_net(next_state_emb_batch,
                                  torch.cat((next_cand_feature_emb_batch, next_cand_item_emb_batch),
                                            dim=1)) * torch.cat(
            (next_cand_feature_mask_batch, next_cand_item_mask_batch), dim=1), dim=1, keepdim=True) / (
                    0.001 + torch.sum(next_cand_feature_mask_batch, dim=1, keepdim=True) + torch.sum(
                next_cand_item_mask_batch, dim=1, keepdim=True))


        advantage_next_r = torch.zeros((BATCH_SIZE, 1), device=self.device)
        advantage_next_r[non_final_mask] = advantage_next
        advantage_next_r = reward_batch + advantage_next_r*GAMMA
        loss_policy = (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(advantage_now, advantage_next_r)).mean()

        loss = loss_value + loss_policy
        self.optimizer.zero_grad()
        loss.backward()
        if data_type == 'feature':
            for param in self.value_net_feature.parameters():
                param.grad.data.clamp_(-1, 1)
        else:
            for param in self.value_net_item.parameters():
                param.grad.data.clamp_(-1, 1)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        # 更新模型参数
        self.optimizer.step()

        # 返回损失值
        return loss_value.data, loss_policy.data
    
    def save_model(self, data_name, filename, epoch_user):
        save_rl_agent(dataset=data_name, model=self.policy_net, filename=filename, epoch_user=epoch_user)
    def load_model(self, data_name, filename, epoch_user):
        model_dict = load_rl_agent(dataset=data_name, filename=filename, epoch_user=epoch_user)
        self.policy_net.load_state_dict(model_dict)
    
    def padding(self, cand):
        pad_size = max([len(c) for c in cand])
        padded_cand = []
        masks = torch.zeros(len(cand), pad_size).to(self.device)
        for i in range(len(cand)):
            c = cand[i]
            cur_size = len(c)
            masks[i] = torch.cat((torch.ones(cur_size),torch.zeros(pad_size-cur_size))).to(self.device)
            new_c = np.ones((pad_size)) * self.PADDING_ID
            new_c[:cur_size] = c
            padded_cand.append(new_c)
        return torch.LongTensor(padded_cand).to(self.device), masks
