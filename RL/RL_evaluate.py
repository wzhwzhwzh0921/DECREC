import time
import argparse
from itertools import count
import torch.nn as nn
import torch
import math
from collections import namedtuple
from utils import *
from RL.env_multi_choice_question import MultiChoiceRecommendEnv
from tqdm import tqdm
EnvDict = {
        LAST_FM_STAR: MultiChoiceRecommendEnv,
        YELP_STAR: MultiChoiceRecommendEnv,
        BOOK:MultiChoiceRecommendEnv,
        MOVIE:MultiChoiceRecommendEnv
    }


def dqn_evaluate(args, kg, dataset, agent, filename, i_episode):
    # 初始化测试环境
    test_env = EnvDict[args.data_name](kg, dataset, args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn,
                                       cand_num=args.cand_num, cand_item_num=args.cand_item_num, attr_num=args.attr_num,
                                       mode='test', ask_num=args.ask_num, entropy_way=args.entropy_method,
                                       fm_epoch=args.fm_epoch)
    set_random_seed(args.seed)
    tt = time.time()
    start = tt
    SR5, SR10, SR15, AvgT, Rank, total_reward = 0, 0, 0, 0, 0, 0
    item_cover_success = dict()
    item_cover = dict()
    feature_cover = dict()
    feature_cover_success =dict()
    SR_turn_15 = [0] * args.max_turn
    turn_result = []
    result = []
    user_size = test_env.ui_array.shape[0]
    print('User size in UI_test: ', user_size)
    test_filename = 'Evaluate-epoch-{}-'.format(i_episode) + filename
    plot_filename = 'Evaluate-'.format(i_episode) + filename
    if args.test_type == 'ALL':
        if args.data_name == 'LAST_FM_STAR':
            test_size = 4000
        elif args.data_name == 'MOVIE':
            test_size = 4000
        elif args.data_name == 'BOOK':
            test_size = 4000
        else:
            test_size = 4000
    else:
        test_size = 500
    user_size = 500
    # if args.data_name in [LAST_FM_STAR, ]:
    #     if args.eval_num == 2:
    #         test_size = 500
    #     else:
    #         test_size = 11892  # 仅进行 4000 次迭代以节省时间
    #     user_size = test_size
    # else:
    #     if args.eval_num == 2:
    #         test_size = 500
    #     else:
    #         test_size = 2500  # 仅进行 2500 次迭代以节省时间
    #     user_size = test_size
    turn_suc_num = [0]*16
    turn_num = [0]*16
    turn_rec =[0]*16
    turn_rec_suc = [0]*16
    print('选择的测试大小：', test_size)
    for user_num in tqdm(range(user_size)):  # 遍历每个用户
        # TODO 取消注释以打印对话过程
        blockPrint()
        print('\n================测试元组：{}===================='.format(user_num))
        if not args.fix_emb:
            state, cand, action_space = test_env.reset(
                agent.gcn_net.embedding.weight.data.cpu().detach().numpy())  # 重置环境并记录起始状态
        else:
            state, cand, action_space = test_env.reset()
        target_item = test_env.target_item
        epi_reward = 0
        is_last_turn = False
        for t in count():  # 用户对话循环
            turn_num[t] += 1
            if t == 14:
                is_last_turn = True
            action, sorted_actions, _ = agent.select_action(state, cand, action_space, is_test=True,
                                                            is_last_turn=is_last_turn)
            if action == None:
                break
            next_state, next_cand, action_space, reward, done, action_type, recom_items, asked_features = test_env.step(action.item(), sorted_actions)

            epi_reward += reward
            reward = torch.tensor([reward], device=args.device, dtype=torch.float)
            if done:
                next_state = None
            state = next_state
            cand = next_cand
            if len(recom_items)!=0:
                turn_rec[t]+=1
            if reward.item() > 0:
                turn_suc_num[t] += 1
            for i in asked_features:
                if i not in feature_cover:
                    feature_cover[i] = 0
                feature_cover[i] += 1
                if reward.item() > 0:  # 成功推荐
                    if i not in feature_cover_success:
                        feature_cover_success[i] = 0
                    feature_cover_success[i] += 1
            rec_item = recom_items
            for i in rec_item:
                if i not in item_cover:
                    item_cover[i] = 0
                item_cover[i] += 1

            if done:
                enablePrint()
                if reward.item() == 1:  # 成功推荐
                    turn_rec_suc[t]+=1
                    for i in rec_item:
                        if i not in item_cover_success:
                            item_cover_success[i] = 0
                        item_cover_success[i] += 1
                    SR_turn_15 = [v + 1 if i > t else v for i, v in enumerate(SR_turn_15)]
                    if t < 5:
                        SR5 += 1
                        SR10 += 1
                        SR15 += 1
                    elif t < 10:
                        SR10 += 1
                        SR15 += 1
                    else:
                        SR15 += 1
                    Rank += (1 / math.log(t + 3, 2) + (1 / math.log(t + 2, 2) - 1 / math.log(t + 3, 2)) / math.log(
                        done + 1, 2))
                else:
                    Rank += 0
                total_reward += epi_reward
                AvgT += t + 1
                break

        if (user_num + 1) % args.observe_num == 0 and user_num > 0:
            SR = [SR5 / args.observe_num, SR10 / args.observe_num, SR15 / args.observe_num, AvgT / args.observe_num,
                  Rank / args.observe_num, total_reward / args.observe_num]
            SR_TURN = [i / args.observe_num for i in SR_turn_15]
            print('总的评估周期用户数：{}'.format(user_num + 1))
            print('完成 {}% 的任务用时：{} 秒'.format(str(time.time() - start), float(user_num) * 100 / user_size))
            print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, reward:{} '
                  '总的周期用户数:{}'.format(SR5 / args.observe_num, SR10 / args.observe_num, SR15 / args.observe_num,
                                             AvgT / args.observe_num, Rank / args.observe_num,
                                             total_reward / args.observe_num, user_num + 1))
            result.append(SR)
            turn_result.append(SR_TURN)
            SR5, SR10, SR15, AvgT, Rank, total_reward = 0, 0, 0, 0, 0, 0
            SR_turn_15 = [0] * args.max_turn
            tt = time.time()
        enablePrint()
    print('turn_num:', turn_num)
    print('turn_suc_num:', turn_suc_num)
    print('turn_rec:', turn_rec)
    print('turn_rec_suc:', turn_rec_suc)
    SR5_mean = np.mean(np.array([item[0] for item in result]))
    SR10_mean = np.mean(np.array([item[1] for item in result]))
    SR15_mean = np.mean(np.array([item[2] for item in result]))
    AvgT_mean = np.mean(np.array([item[3] for item in result]))
    Rank_mean = np.mean(np.array([item[4] for item in result]))
    reward_mean = np.mean(np.array([item[5] for item in result]))
    SR_all = [SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean, reward_mean,len(item_cover.keys()),len(item_cover_success.keys()),len(feature_cover.keys()),len(feature_cover_success.keys())]
    save_rl_mtric(dataset=args.data_name, filename=filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                  mode='test')
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=user_num, SR=SR_all,
                  spend_time=time.time() - start,
                  mode='test')  # 保存 RL SR
    print('测试评估保存成功！')

    print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, reward:{}, item_cover:{}, item_cover_correct:{}, feature_cover:{},feature_cover_correct:{}'.format(SR5_mean, SR10_mean, SR15_mean, AvgT_mean,
                                                                         Rank_mean, reward_mean,len(item_cover.keys()), len(item_cover_success.keys()),len(feature_cover.keys()),len(feature_cover_success.keys())))
    item_num = 16482  # movie:16482
    print('物品覆盖率为:{},推荐成功的物品覆盖率为:{}'.format(len(item_cover.keys()) / item_num,
          len(item_cover_success.keys()) / item_num))
    PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('训练周期:{}\n'.format(i_episode))
        f.write('===========测试轮次===============\n')
        f.write('测试 {} 个用户元组\n'.format(user_num))
        f.write('================================\n')
    PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' + plot_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(i_episode, SR15_mean, AvgT_mean, Rank_mean, reward_mean, len(item_cover.keys()), len(item_cover_success.keys()), len(feature_cover.keys()),len(feature_cover_success.keys())))

    return SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean


