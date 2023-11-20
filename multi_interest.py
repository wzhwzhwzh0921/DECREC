import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import pickle
import gzip
import numpy as np
import time

from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
import dgl.nn as dglnn
import dgl.function as fn
import dgl
from utils import *


class SelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.,anchor=False):
        super().__init__()
        self.anchor=anchor
        if anchor:
            self.scorer = nn.Linear(d_hid*2, 1)
        else:
            self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters(d_hid)
    def reset_parameters(self,d_hid):
        if self.anchor:
            stdv = 1. / math.sqrt(d_hid*2)
        else:
            stdv = 1. / math.sqrt(d_hid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_seq,anchor=None,s=None):
        batch_size, seq_len, feature_dim = input_seq.size()
        input_seq = self.dropout(input_seq)
        
        if anchor!=None:
            size=input_seq.shape[1]
            anchor=anchor.repeat(1,size,1)
            seq=torch.cat((input_seq,anchor),2)
            # enablePrint()
            # ipdb.set_trace()
            scores = self.scorer(seq.contiguous().view(-1, feature_dim*2)).view(batch_size, seq_len)+s
        else:
            scores = self.scorer(input_seq.contiguous().view(-1, feature_dim)).view(batch_size, seq_len)
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(input_seq).mul(input_seq).sum(1)
        return context,scores # 既然命名为context就应该是整句的表示



class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphEncoder(Module):
    def __init__(self, graph, device, entity, emb_size, kg, embeddings=None, fix_emb=True, seq='rnn', gcn=True, hidden_size=100, layers=1, rnn_layer=1,u=None,v=None,f=None):
        super(GraphEncoder, self).__init__()
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        # self.eps = 0.0
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        rel_names=['interact','friends','like','belong_to']
        self.G = graph.to(device)
        self.conv1 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(emb_size, hidden_size) for rel in rel_names},
                                           aggregate='mean')

        self.embedding = nn.Embedding(entity, emb_size, padding_idx=entity-1)
        if embeddings is not None:
            print("pre-trained embeddings")
            self.embedding.from_pretrained(embeddings,freeze=fix_emb)
        self.layers = layers
        self.user_num = u
        self.item_num = v
        self.PADDING_ID = entity-1
        self.device = device
        self.seq = seq
        self.gcn = gcn
        self.hidden_size=hidden_size

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc_neg = nn.Linear(hidden_size, hidden_size)
        # if self.seq == 'rnn':
        #     self.rnn = nn.GRU(hidden_size, hidden_size, rnn_layer, batch_first=True)
        # elif self.seq == 'transformer':
        #     self.transformer1 = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=400), num_layers=rnn_layer)
            # self.transformer2 = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=400), num_layers=rnn_layer)

        if self.gcn:
            indim, outdim = emb_size, hidden_size
            self.gnns = nn.ModuleList()
            for l in range(layers):
                self.gnns.append(GraphConvolution(indim, outdim))
                indim = outdim
        else:
            self.fc2 = nn.Linear(emb_size, hidden_size)
        self.num_pers=4
        self.multi_head_self_attention_init = nn.ModuleList([SelfAttention(self.hidden_size, 0.3) for _ in range(self.num_pers)])
        self.multi_head_self_attention = nn.ModuleList([SelfAttention(self.hidden_size, 0.3,anchor=True) for _ in range(self.num_pers)])

    def forward(self, b_state):
        """
        前向传播函数，接受一个包含输入状态的批次数据 `b_state` 作为输入，并返回输出张量。

        参数:
        - b_state: 包含输入状态的列表，每个状态对应批次中的一个样本 [N]

        返回值:
        - 输出张量 [N x L x d]
        """

        # 初始化节点嵌入向量，分为'user'、'item'和'attribute'三个部分
        h0 = {
            'user': self.embedding(torch.arange(0, self.user_num).long().to(self.device)),
            'item': self.embedding(torch.arange(self.user_num, self.user_num + self.item_num).long().to(self.device)),
            'attribute': self.embedding(
                torch.arange(self.user_num + self.item_num, self.PADDING_ID + 1).long().to(self.device))
        }

        # 进行第一层卷积操作
        h1 = self.conv1(self.G, h0)

        # 对卷积结果进行ReLU激活函数操作
        h1 = {k: self.relu((v)) for k, v in h1.items()}

        # 将不同类型的节点嵌入向量拼接在一起，形成一个大的嵌入向量
        gnn_embedding = torch.cat((h1['user'], h1['item']), dim=0)
        gnn_embedding = torch.cat((gnn_embedding, h1['attribute']), dim=0)

        batch_output = []

        # 遍历输入批次中的每个样本
        for s in b_state:
            # 从样本中获取相邻节点和邻接矩阵
            neighbors, adj = s['neighbors'].to(self.device), s['adj'].to(self.device)

            # 通过节点嵌入层获取输入状态的嵌入向量
            input_state = self.embedding(neighbors)

            if self.gcn:
                # 如果使用GCN，进行多层GCN操作
                for gnn in self.gnns:
                    output_state = gnn(input_state, adj)
                    input_state = output_state
                batch_output.append(output_state)
            else:
                # 否则，通过全连接层进行激活操作
                output_state = F.relu(self.fc2(input_state))
                batch_output.append(output_state)

        seq_embeddings = []
        rej_feature_embeddings = []
        rej_item_embeddings = []
        user_em = []

        # 遍历每个样本以处理其嵌入和被拒绝的特征/物品
        for s, o in zip(b_state, batch_output):
            # 根据输出状态和当前样本的节点，生成序列嵌入向量
            seq_embeddings.append(
                (1 - self.eps) * o[:len(s['cur_node']), :][None, :] + self.eps * gnn_embedding[s['cur_node']])

            # 计算被拒绝特征的嵌入向量
            if len(s['rej_feature']) > 0:
                rej_feature_embeddings.append(torch.mean(gnn_embedding[s['rej_feature']], dim=0).view(1, -1))
            else:
                rej_feature_embeddings.append(torch.zeros([1, self.hidden_size]).to(self.device))

            # 计算被拒绝物品的嵌入向量
            if len(s['rej_item']) > 0:
                rej_item_embeddings.append(torch.mean(gnn_embedding[s['rej_item']], dim=0).view(1, -1))
            else:
                rej_item_embeddings.append(torch.zeros([1, self.hidden_size]).to(self.device))

            # 获取用户嵌入向量
            user_em.append(gnn_embedding[s['user']])

        if len(batch_output) > 1:
            # 如果输入样本的数量大于1，进行序列的填充，保证它们的长度一致
            seq_embeddings = self.padding_seq(seq_embeddings)
        seq_embeddings = torch.cat(seq_embeddings, dim=0)  # [N x L x d]

        # 用户嵌入向量拼接成 [N x 1 x d] 的形状
        user_em = torch.cat(user_em, dim=0).view(-1, 1, self.hidden_size)
        rej_embedding = None

        # 创建 `interest_emb`，它包含了序列嵌入向量
        interest_emb = seq_embeddings

        if len(rej_feature_embeddings) > 0 and len(rej_item_embeddings) > 0:
            # 如果有被拒绝的特征和物品的嵌入向量，则将它们拼接在一起
            rej_feature_embed = torch.cat(rej_feature_embeddings, dim=0).unsqueeze(1)
            rej_item_embed = torch.cat(rej_item_embeddings, dim=0).unsqueeze(1)
            rej_embedding = rej_item_embed + rej_feature_embed
        elif len(rej_feature_embeddings) > 0:
            # 如果只有被拒绝的特征的嵌入向量，则使用它们
            rej_feature_embed = torch.cat(rej_feature_embeddings, dim=0).unsqueeze(1)
            rej_embedding = rej_feature_embed
        elif len(rej_item_embeddings) > 0:
            # 如果只有被拒绝的物品的嵌入向量，则使用它们
            rej_item_embed = torch.cat(rej_item_embeddings, dim=0).unsqueeze(1)
            rej_embedding = rej_item_embed

        if rej_embedding is not None:
            # 如果存在被拒绝的嵌入向量，则通过全连接层进行激活操作
            mm = F.relu(self.fc_neg(rej_embedding))
            interest_emb = torch.cat((interest_emb, mm), 1)

        seq_embedding = []  # 用于存储自注意力机制的输出
        score = []  # 用于存储注意力权重

        # 多头自注意力机制的计算
        # 遍历初始的多头自注意力机制
        for self_attention in self.multi_head_self_attention_init:
            # 计算自注意力机制的输出和注意力权重
            context, weight = self_attention(interest_emb)
            # 将输出加入到 `seq_embedding` 列表中，维度变为 [N x 1 x d]
            seq_embedding.append(context.view(-1, 1, self.hidden_size))
            # 存储注意力权重
            score.append(weight)

        # 循环迭代多次（此处为1次）
        for _ in range(1):
            i = 0
            for self_attention in self.multi_head_self_attention:
                # 多头自注意力机制的多次迭代
                context, weight = self_attention(interest_emb, seq_embedding[i], score[i])
                # 更新 `seq_embedding` 中的值，维度变为 [N x 1 x d]
                seq_embedding[i] = context.view(-1, 1, self.hidden_size)
                # 更新注意力权重
                score[i] = weight
                i += 1

        # 返回自注意力机制的输出
        return seq_embedding
    
    
    def padding_seq(self, seq):
        padding_size = max([len(x[0]) for x in seq])
        padded_seq = []
        for s in seq:
            cur_size = len(s[0])
            emb_size = len(s[0][0])
            new_s = torch.zeros((padding_size, emb_size)).to(self.device)
            new_s[:cur_size,:] = s[0]
            padded_seq.append(new_s[None,:])
        return padded_seq
