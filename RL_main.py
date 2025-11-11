import torch
import argparse
from model import *
# from prepare_data import *
from Actor_Critic_model import AC_model  # 使用新的Actor-Critic模型
import warnings
import os
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print(torch.cuda.is_available())
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    parser = argparse.ArgumentParser()

    # 定义参数
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--sample_size', type=int, default=2, help='input batch size')
    parser.add_argument('--hidden_dim', type=int, default=100, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_layers', type=int, default=6, help='layers nums')
    parser.add_argument('--num_heads', type=int, default=6, help='attention heads nums')
    parser.add_argument('--mlp_ratio', type=float, default=1, help='the ratio of hidden layers in the middle')
    parser.add_argument('--Kernel_size1', type=int, default=2, help='the first layer convolution kernel size')
    parser.add_argument('--Kernel_size2', type=int, default=2, help='the second layer convolution kernel size')
    parser.add_argument('--Stride1', type=int, default=2, help='the second layer convolution stride size')
    parser.add_argument('--Stride2', type=int, default=2, help='the second layer convolution stride size')
    parser.add_argument('--num_classes', type=int, default=77, help='the number of categories')
    parser.add_argument('--state_nums', type=int, default=77, help='the number of agents/dimensions')  # 新增state_nums参数
    # parser.add_argument('--pb_path', type=str, default='your_pb_path_here', help='the path to the pb file')  # 新增pb_path参数
    parser.add_argument('--pb_path', type=str, default='/home/dlg/li/OK/#2213_code/data/画/new_1/打分1.xlsx', help='the path to the pb file')
    parser.add_argument('--least_score', type=float, default=0.2, help='some score threshold')  # 新增least_score参数
    parser.add_argument('--photo_size', type=int, default=128, help='photo size')
    parser.add_argument('--Linear_nums', type=int, default=3, help='the number of categories to the last label')
    # parser.add_argument('--data_path', type=str, default='/home/dlg/li/#2213_code/#2213_code/data', help='the path to the data folder')
    parser.add_argument('--data_path', type=str, default='/home/dlg/li/OK/#2213_code/data', help='the path to the data folder')
    parser.add_argument('--agent_nums', type=int, default=77, help='the number of categories to the last label')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=0.8, help='epsilon for epsilon-greedy strategy')
    parser.add_argument('--target_update_nums', type=int, default=5, help='target update frequency')
    parser.add_argument('--ReplayBuffer_capacity', type=int, default=100, help='replay buffer capacity')
    parser.add_argument('--min_size', type=int, default=50, help='minimum size for the replay buffer')
    parser.add_argument('--path_len', type=int, default=5, help='path length for episodes')
    parser.add_argument('--model_name', type=str, default='Aes', help='model name')
    parser.add_argument('--mu', type=float, default=1, help='parameter mu')
    parser.add_argument('--embedding_dim', type=int, default=100, help='embedding dimension')
    parser.add_argument('--msepara', type=int, default=100, help='parameter msepara')

    opt = parser.parse_args()

    # 加载数据
    train_set, test_set = get_RL_data(opt.photo_size, opt.data_path)
    train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate_RL, pin_memory=True, num_workers=0, drop_last=True)
    test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=collate_RL, pin_memory=True, num_workers=0, drop_last=True)

    # 计算embedding维度
    embedding_dim = int((opt.photo_size - opt.Kernel_size1) // opt.Stride1 + 1)
    embedding_dim = int((embedding_dim - 2) // 2 + 1)
    embedding_dim = int((embedding_dim - opt.Kernel_size2 + 2) // opt.Stride2 + 1)
    embedding_dim = int((embedding_dim - 2) // 2 + 1)
    embedding_dim = int(embedding_dim * embedding_dim * 3)

    # 使用新的Actor-Critic模型
    model = AC_model(opt, embedding_dim).cuda()

    # 训练和评估
    for epoch in range(opt.epoch):
        model.fit(train_data)
        if epoch == opt.epoch - 1:
            model.evaluate(test_data)