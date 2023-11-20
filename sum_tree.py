import numpy as np

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree(object):
    write = 0  # 初始化写指针为0

    def __init__(self, capacity):
        self.capacity = capacity  # 初始化树的容量
        self.tree = np.zeros(2 * capacity - 1)  # 初始化树的优先级数组
        self.data = np.zeros(capacity, dtype=object)  # 初始化数据存储数组，用于存储数据
        self.n_entries = 0  # 初始化数据条目计数为0

    # 更新到根节点
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2  # 计算当前节点的父节点索引

        self.tree[parent] += change  # 更新父节点的优先级

        if parent != 0:
            self._propagate(parent, change)  # 递归向上更新父节点的祖先节点

    # 在叶子节点上查找样本
    def _retrieve(self, idx, s):
        left = 2 * idx + 1  # 计算左子节点索引
        right = left + 1  # 计算右子节点索引

        if left >= len(self.tree):
            return idx  # 如果超出树的范围，返回当前节点索引

        if s <= self.tree[left]:
            return self._retrieve(left, s)  # 如果要查找的优先级小于等于左子节点的优先级，则向左子树查找
        else:
            return self._retrieve(right, s - self.tree[left])  # 否则，向右子树查找并减去左子节点的优先级

    def total(self):
        return self.tree[0]  # 返回树的根节点（总和）的优先级

    # 存储优先级和样本数据
    def add(self, p, data):
        idx = self.write + self.capacity - 1  # 计算数据应该存储的位置

        self.data[self.write] = data  # 存储数据
        self.update(idx, p)  # 更新存储的节点的优先级

        self.write += 1  # 更新写指针
        if self.write >= self.capacity:
            self.write = 0  # 如果写指针超过容量，重置为0

        if self.n_entries < self.capacity:
            self.n_entries += 1  # 更新数据条目计数，不超过容量

    # 更新节点的优先级
    def update(self, idx, p):
        change = p - self.tree[idx]  # 计算优先级变化值

        self.tree[idx] = p  # 更新节点的优先级
        self._propagate(idx, change)  # 递归更新相关祖先节点的优先级


    def get(self, s):
        idx = self._retrieve(0, s)  # 查找具有给定优先级的节点
        dataIdx = idx - self.capacity + 1  # 计算数据存储数组中的索引

        return idx, self.tree[idx], self.data[dataIdx]  # 返回节点索引、优先级和相关的数据
