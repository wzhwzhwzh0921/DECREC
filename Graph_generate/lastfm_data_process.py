import os
import json
from easydict import EasyDict as edict

# 定义一个名为 LastFmDataset 的类
class LastFmDataset(object):
    def __init__(self, data_dir):
        # 初始化类实例时传入数据目录
        self.data_dir = data_dir + '/Graph_generate_data'
        # 加载实体数据
        self.load_entities()
        # 加载关系数据
        self.load_relations()

    # 获取关系类型和实体类型的映射
    def get_relation(self):
        # 实体类型
        USER = 'user'        # 用户
        ITEM = 'item'        # 物品
        FEATURE = 'feature'  # 特征

        # 关系类型
        INTERACT = 'interact'       # 互动
        FRIEND = 'friends'          # 好友
        LIKE = 'like'               # 喜欢
        BELONG_TO = 'belong_to'     # 属于
        relation_name = [INTERACT, FRIEND, LIKE, BELONG_TO]  # 所有关系类型的列表

        # 定义实体之间的关系映射
        fm_relation = {
            USER: {
                INTERACT: ITEM,     # 用户与物品之间的互动关系
                FRIEND: USER,       # 用户与用户之间的好友关系
                LIKE: FEATURE,      # 用户对特征的喜欢关系
            },
            ITEM: {
                BELONG_TO: FEATURE, # 物品属于特征的关系
                INTERACT: USER      # 物品与用户之间的互动关系
            },
            FEATURE: {
                LIKE: USER,         # 特征被用户喜欢的关系
                BELONG_TO: ITEM     # 特征属于物品的关系
            }
        }

        # 定义关系对应的实体类型
        fm_relation_link_entity_type = {
            INTERACT:  [USER, ITEM],     # 互动关系对应的实体类型为用户和物品
            FRIEND:  [USER, USER],       # 好友关系对应的实体类型为用户
            LIKE:  [USER, FEATURE],      # 喜欢关系对应的实体类型为用户和特征
            BELONG_TO:  [ITEM, FEATURE]  # 属于关系对应的实体类型为物品和特征
        }
        return fm_relation, relation_name, fm_relation_link_entity_type  # 返回关系映射

    # 加载实体数据
    def load_entities(self):
        # 定义各个实体的数据文件
        entity_files = edict(
            user='user_dict.json',
            item='item_dict.json',
            feature='merged_tag_map.json',
        )
        # 遍历实体文件并加载数据
        for entity_name in entity_files:
            # 打开实体数据文件
            with open(os.path.join(self.data_dir, entity_files[entity_name]), encoding='utf-8') as f:
                mydict = json.load(f)
            if entity_name == 'feature':
                entity_id = list(mydict.values())
            else:
                entity_id = list(map(int, list(mydict.keys())))
            # 设置实体的属性，包括ID和值的长度
            setattr(self, entity_name, edict(id=entity_id, value_len=max(entity_id)+1))
            print('Load', entity_name, 'of size', len(entity_id))  # 打印加载的实体类型和数量
            print(entity_name, 'of max id is', max(entity_id))      # 打印实体类型的最大ID

    # 加载关系数据
    def load_relations(self):
        """
        加载各个关系的数据
        """
        # 定义 LastFm_relations，包含不同关系的文件名和关联的实体类型
        LastFm_relations = edict(
            interact=('user_item.json', self.user, self.item),  # 互动关系（文件名，头实体类型，尾实体类型）
            friends=('user_dict.json', self.user, self.user),    # 好友关系
            like=('user_dict.json', self.user, self.feature),    # 喜欢关系
            belong_to=('item_dict.json', self.item, self.feature), # 属于关系
        )
        # 遍历不同关系
        for name in LastFm_relations:
            # 创建关系对象
            relation = edict(
                data=[],
            )
            knowledge = [list([]) for i in range(LastFm_relations[name][1].value_len)]
            # 打开关系数据文件
            with open(os.path.join(self.data_dir, LastFm_relations[name][0]), encoding='utf-8') as f:
                mydict = json.load(f)
            # 根据不同关系类型，将数据加载到对应的实体关系中
            if name in ['interact']:
                for key, value in mydict.items():
                    head_id = int(key)
                    tail_ids = value
                    knowledge[head_id] = tail_ids
            elif name in ['friends', 'like']:
                for key in mydict.keys():
                    head_str = key
                    head_id = int(key)
                    tail_ids = mydict[head_str][name]
                    knowledge[head_id] = tail_ids
            elif name in ['belong_to']:
                for key in mydict.keys():
                    head_str = key
                    head_id = int(key)
                    tail_ids = mydict[head_str]['feature_index']
                    knowledge[head_id] = tail_ids
            relation.data = knowledge
            setattr(self, name, relation)
            tuple_num = 0
            for i in knowledge:
                tuple_num += len(i)
            print('Load', name, 'of size', tuple_num)  # 打印加载的关系类型和关系数量






