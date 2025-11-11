import torch
import torch.nn as nn
import numpy as np
from model import *
# from dqn import DQN
from prepare_data import *
from Actor_Critic import ActorCritic
from utils import evaluate
from prepare_data import *
import random

class AC_model(nn.Module):
    def __init__(self, opt, embedding_dim):
        super(AC_model, self).__init__()
        self.opt = opt
        self.visit_count = 0
        self.pb_info, self.pb_info_idx, self.pb_nd_update = get_all_pbs_dimension(opt.pb_path)
        # self.paint_to_course, self.course_to_dimension, self.dimension_to_idx, self.dimension_adj_martix = get_all_reflect_relation()

        self.paint_to_course, self.dimension_adj_martix = get_all_reflect_relation(opt.data_path)
        self.aclist = [ActorCritic(7 + opt.embedding_dim, opt.hidden_dim, 2, opt.lr, opt.gamma, opt.epsilon, 
                       self.pb_info, self.pb_info_idx, opt.least_score, opt.batch_size).cuda() for _ in range(opt.state_nums)]
        self.aes = AesModel(embedding_dim, opt)
        self.train_agents, self.test_agents = {}, {}
        self.state_embedding = nn.Embedding(opt.state_nums, opt.embedding_dim).cuda()
        self.state_embed = self.state_embedding(torch.LongTensor([_ for _ in range(opt.state_nums)]).cuda()).cuda()
        self.GAT = GAT(self.opt.embedding_dim, self.opt.embedding_dim, self.opt.embedding_dim, 0.7, 0.2, self.opt.num_heads).cuda()

    def cal_s1(self, chosen_dimensions_val):
        std_val, avg_val, min_val, max_val = torch.std(chosen_dimensions_val), torch.mean(chosen_dimensions_val), torch.min(chosen_dimensions_val), torch.max(chosen_dimensions_val)
        m1, m2, m3 = torch.quantile(chosen_dimensions_val, 0.25), torch.quantile(chosen_dimensions_val, 0.50), torch.quantile(chosen_dimensions_val, 0.75)
        return torch.tensor([std_val, avg_val, min_val, max_val, m1, m2, m3], dtype=torch.float).unsqueeze(0).cuda()

    def cal_s2(self, mark_id_list, state_embed):
        s2 = self.state_embed[mark_id_list]
        s2 = torch.mean(s2, dim=0).unsqueeze(0).cuda()
        return s2

    def cal_new_state(self, action_list, aes_ability):
        mark_id_list = []
        for _ in range(len(action_list)):
            if action_list[_] == 1:
                mark_id_list.append(_)
        mark_id_list = torch.LongTensor(mark_id_list)
        s1 = []
        if len(mark_id_list) == 0:
            s1 = torch.tensor([0 for _ in range(7)]).unsqueeze(0).cuda()
        else:
            chosen_dimensions_val = aes_ability[:, mark_id_list]
            s1 = self.cal_s1(chosen_dimensions_val)
        s2 = self.cal_s2(mark_id_list, self.state_embed)
        new_state = torch.cat((s1, s2), dim=1).cuda()
        return new_state

    def init_state(self, aes_ability):
        action_list = [random.randint(0, 1) for _ in range(self.opt.state_nums)]
        ini_state = self.cal_new_state(action_list, aes_ability)
        return ini_state, action_list

    def cal_new_ability(self, action_lsit, aes_ability, mark_ability):
        aes_ability, mark_ability = aes_ability.tolist()[0], mark_ability.tolist()[0]
        cur_ability = aes_ability
        for idx in range(len(action_lsit)):
            if action_lsit[idx] == 1:
                cur_ability[idx] = mark_ability[idx]
            else:
                cur_ability[idx] = aes_ability[idx]
        return torch.tensor(cur_ability).unsqueeze(0).cuda()

    def pearson_correlation(self, martix):
        martix = martix.cpu().detach().numpy()
        correlation_matrix = np.corrcoef(martix)

        # 提取上三角部分的相关性（不包括对角线）
        upper_triangle_indices = np.triu_indices_from(correlation_matrix, k=1)
        upper_triangle_values = correlation_matrix[upper_triangle_indices]

        # 计算平均相关性值
        average_correlation = np.mean(upper_triangle_values)
        return average_correlation

    def cal_reward(self, init_mse, action_list, mark_ability, new_ability, discount):
        mae, mse, rmse = evaluate(new_ability.tolist(), mark_ability.unsqueeze(0).tolist())
        mse_improve = abs(mse - init_mse) * self.opt.msepara
        mark_id_list = []
        for _ in range(len(action_list)):
            if action_list[_] == 1:
                mark_id_list.append(_)
        sim = 0
        if len(mark_id_list) < 2:
            sim = 10
        else:
            sim_martix = self.state_embed[torch.tensor(mark_id_list)].cuda()
            sim = self.pearson_correlation(sim_martix)
        punish_val = sum(action_list) * self.opt.mu

        reward = (mse_improve + sim + punish_val) * discount
        record_list = [mse_improve, sim, punish_val]
        return reward, record_list

    def cal_reward(self, old_ability, new_ability, discount):
        rsum = sum([(float(new_ability[_].item()) - float(old_ability)) / (1 - float(old_ability)) for _ in range(len(new_ability))]) / len(new_ability)
        reward = discount * rsum
        return reward

    def fit(self, train_loader):
        self.aes.load_state_dict(torch.load('./aes_model_weight.pth'))
        num_visits, tl_visits = 0, int(train_loader.batch_size) *  len(train_loader) * self.opt.path_len
        self.train_agents = {}
        while num_visits < tl_visits:
            for img, name, labels in train_loader:
                for i in range(len(img)):
                    if name[i] not in self.train_agents:
                        self.train_agents[name[i]] = {}
                        one_picture = img[i].unsqueeze(0).cuda()
                        aes_ability, _ = self.aes(one_picture)
                        min_val = aes_ability.min()
                        max_val = aes_ability.max()

                        # min-max 归一化到 [0.2, 0.8]
                        aes_ability = (aes_ability - min_val) * (0.8 - 0.2) / (max_val - min_val) + 0.2

                        self.train_agents[name[i]]['cur_state'], self.train_agents[name[i]]['last_actions'] = self.init_state(aes_ability)
                        self.train_agents[name[i]]['mark_ability'] = torch.tensor(labels[i]).unsqueeze(0).cuda()
                        self.train_agents[name[i]]['aes_ability'] = aes_ability
                        score, label = aes_ability.tolist(), labels[i].unsqueeze(0).tolist()
                        mae, mse, rmse = evaluate(score, label)

                        self.train_agents[name[i]]['init_mse'] = mse
                        self.train_agents[name[i]]['discount'] = self.opt.gamma
                        self.train_agents[name[i]]['iter_count'] = 0
                        self.train_agents[name[i]]['rewards'] = []
                    if self.train_agents[name[i]]['iter_count'] >= self.opt.path_len:
                        continue
                    action_list = [ac_model.choose_action(self.train_agents[name[i]]['cur_state']) for ac_model in self.aclist]
                    new_state = self.cal_new_state(action_list, aes_ability)
                    new_ability = self.cal_new_ability(action_list, self.train_agents[name[i]]['aes_ability'], self.train_agents[name[i]]['mark_ability'])

                    reward, record_list = self.cal_reward(self.train_agents[name[i]]['init_mse'], action_list, self.train_agents[name[i]]['mark_ability'], new_ability, self.train_agents[name[i]]['discount'])
                    self.train_agents[name[i]]['rewards'].append(record_list)
                    
                    for idx, ac_model in enumerate(self.aclist):
                        if action_list[idx] == 1 or self.train_agents[name[i]]['last_actions'][idx] == 1:
                            ac_model.update(self.train_agents[name[i]]['cur_state'], reward, new_state)
                    
                    self.train_agents[name[i]]['cur_state'], self.train_agents[name[i]]['discount'] = new_state, self.train_agents[name[i]]['discount'] * self.opt.gamma
                    self.train_agents[name[i]]['last_actions'] = action_list
                    self.train_agents[name[i]]['iter_count'] += 1
                    num_visits += 1

            self.state_embed = self.GAT(self.state_embed, self.dimension_adj_martix).cuda()


    def evaluate(self, test_loader):
        num_visits, tl_visits = 0, int(test_loader.batch_size) *  len(test_loader) * 1
        while num_visits < tl_visits:
            for img, name, labels in test_loader:
                for i in range(len(img)):
                    one_picture = img[i].unsqueeze(0).cuda()
                    if name[i] not in self.test_agents:
                        self.test_agents[name[i]] = {}
                        aes_ability, _ = self.aes(one_picture)
                        min_val = aes_ability.min()
                        max_val = aes_ability.max()
                        # min-max 归一化到 [0.2, 0.8]
                        aes_ability = (aes_ability - min_val) * (0.8 - 0.2) / (max_val - min_val) + 0.2

                        self.test_agents[name[i]]['cur_state'], self.test_agents[name[i]]['last_actions'] = self.init_state(aes_ability)

                        self.test_agents[name[i]]['mark_ability'] = torch.tensor(labels[i]).unsqueeze(0).cuda()
                        self.test_agents[name[i]]['aes_ability'] = aes_ability
                        self.test_agents[name[i]]['cur_ability'] = aes_ability

                        self.test_agents[name[i]]['iter_count'] = 0
                    if self.test_agents[name[i]]['iter_count'] >= self.opt.path_len:
                        continue

                    action_list = [ac_model.choose_action(self.test_agents[name[i]]['cur_state']) for ac_model in self.aclist]
                    self.test_agents[name[i]]['cur_ability'] = self.cal_new_ability(action_list, self.test_agents[name[i]]['aes_ability'], self.test_agents[name[i]]['mark_ability'])
                    self.test_agents[name[i]]['iter_count'] += 1
                    num_visits += 1
        # with open(f"reason.txt", mode='a') as file:
        score, label, count = [], [], 0
        for ke, va in self.test_agents.items():
            score.append(va['cur_ability'].tolist()[0])
            label.append(va['mark_ability'].tolist()[0])
        for idx in range(len(score)):
            for _ in range(len(score[idx])):
                if abs(score[idx][_] - label[idx][_]) <= 0:
                    count += 1

        # with open(f"recod_list_{self.opt.mu}.txt", mode='a', encoding='utf-8') as file:
        #     for ke, va in self.train_agents.items():
        #         for idx in range(len(va['rewards'])):
        #             ans = ke + str(' ')
        #             for _ in range(len(va['rewards'][idx])):
        #                 ans = ans + str(va['rewards'][idx][_][0].item()) + str(' ')
        #             ans = ans + str('\n')
        #             file.write(ans)
        #             ans = ke + str(' ')
        #             for _ in range(len(va['rewards'][idx])):
        #                 # ans = ans + str(va['rewards'][idx][_][1].item()) + str(' ')
        #                 ans = ans + str(va['rewards'][idx][_][1]) + str(' ')
        #             ans = ans + str('\n')
        #             file.write(ans)
        #             file.write('\n')

        score, label = torch.tensor(score), torch.tensor(label)
        mae, mse, rmse = evaluate(score, label)
        if self.opt.is_adjust_parameter == str('true'):
            adjust_parameter_val = getattr(self.opt, self.opt.expand_name)
            with open(f"parameter_{self.opt.expand_name}.txt", mode='a', encoding='utf-8') as file:
                file.write(str(f'{self.opt.expand_name} {adjust_parameter_val} {[mae, mse, count / len(score)]}'))
                file.write('\n')
        print(f'mae {mae} mse {mse} rmse {rmse}')
        print('avg is', count / len(score))