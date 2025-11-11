import torch
import argparse
from model import *
from Actor_Critic_model import AC_model  # 使用新的Actor-Critic模型
import warnings
import os
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import random

# ==============================================================================
#  第一部分：将所有必要的函数定义放在这里（主程序之前）
# ==============================================================================

def get_all_pbs_dimension(path):
    """
    从Excel文件中读取并处理数据，返回三个字典
    """
    data = pd.read_excel(path, index_col=0)
    data = data.dropna(axis=0, how='any')
    column_names = data.columns.tolist()[:-1]
    
    pb_info = {}
    pb_info_idx = {}
    pb_nd_update = {}

    for idx, name in enumerate(column_names):
        name = name.replace(' ', '')
        pb_info[name] = idx
        pb_info_idx[idx] = name
        
    for index, row in data.iterrows():
        name = index.replace(' ', '')
        if name not in pb_info:
            continue
        row_values = row.iloc[:-1].values
        pb_nd_update[name] = row_values

    return pb_info, pb_info_idx, pb_nd_update


def get_all_reflect_relation(data_path):
    """
    获取维度邻接矩阵和课程关系
    """
    # 注意：这里的路径是硬编码的，请确保它正确
    relation_path = os.path.join(data_path, '画/new_1/打分1.xlsx')
    
    try:
        data = pd.read_excel(relation_path, index_col=0)
    except FileNotFoundError:
        print(f"错误：在 get_all_reflect_relation 中找不到文件: {relation_path}")
        # 返回一个默认值让代码能继续运行，但你需要检查路径
        return {}, torch.zeros(77, 77)

    data = data.dropna(axis=0, how='any')
    
    dimension_names = data.columns.tolist()[:-1]
    num_dimensions = len(dimension_names)
    
    dimension_adj_matrix = torch.zeros(num_dimensions, num_dimensions)
    
    paint_to_course = {}
    for index in data.index:
        if isinstance(index, str):
            paint_to_course[index] = index
    
    for i in range(num_dimensions):
        for j in range(num_dimensions):
            if i != j:
                dimension_adj_matrix[i][j] = 1.0
    
    return paint_to_course, dimension_adj_matrix


# RL_main.py 中

# --- 请用下面的代码完全替换掉你原来的 get_RL_data 函数 ---

def get_RL_data(image_size, data_path):
    """
    加载训练集和测试集数据（最终修正版）。
    根据用户提供的确切路径加载数据。
    """
    import torch
    from PIL import Image
    import os
    import pandas as pd
    import numpy as np
    import random

    # --- 1. 根据你的描述，硬编码确切的文件和文件夹路径 ---
    image_folder_path = os.path.join(data_path, '画', 'new_1')
    label_file_path = os.path.join(image_folder_path, '打分1.xlsx')

    # 这是一个辅助函数，用于从Excel中加载标签
    def get_labels(path):
        try:
            data = pd.read_excel(path, index_col=0)
        except FileNotFoundError:
            print(f"错误：在 get_labels 中找不到标签文件: {path}")
            return {}
        data = data.dropna(axis=0, how='any')
        labels = {}
        convert_dict = {i: i * 0.15 + 0.2 for i in range(5)}
        for name in data.index.values[1:]: # 从第二行开始读取
            if not isinstance(name, str): 
                continue
            try:
                row_values = data.loc[name].values.tolist()
                # 检查 row_values 是否足够长
                if len(row_values) > 0:
                    # 假设最后一列是文本或其他非数值列，所以忽略
                    numeric_values = row_values[:-1]
                    temp = [int(i) for i in numeric_values]
                    labels[name] = [convert_dict[min(int(i), 4)] for i in temp]
                else:
                    print(f"警告：处理标签时，行 '{name}' 数据为空。")
            except (ValueError, IndexError, TypeError) as e:
                print(f"警告：处理标签时跳过行 '{name}'，错误: {e}")
                continue
        return labels

    # --- 2. 直接从确切路径加载标签 ---
    print(f"正在从 {label_file_path} 加载标签...")
    all_labels = get_labels(label_file_path)
    if not all_labels:
        print("错误：未能加载任何标签，请检查标签文件路径和内容。")
        return [], []
    
    # --- 3. 直接从确切路径加载图片 ---
    print(f"正在从 {image_folder_path} 加载图片...")
    image_list = {}
    if not os.path.exists(image_folder_path):
        print(f"错误：找不到图片目录: {image_folder_path}")
        return [], []
        
    for filename in os.listdir(image_folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img_pil = Image.open(os.path.join(image_folder_path, filename)).convert('RGB')
                img_pil = img_pil.resize((image_size, image_size))
                img_tensor = torch.tensor(np.array(img_pil)).permute(2, 0, 1).float()
                name_without_ext = os.path.splitext(filename)[0]
                image_list[name_without_ext] = img_tensor
            except Exception as e:
                print(f"警告：无法加载或处理图片 {filename}，错误: {e}")
    
    # --- 4. 匹配图像和标签 (这部分逻辑保持不变) ---
    final_data = []
    for name, img in image_list.items():
        if name in all_labels:
            final_data.append([img, name, all_labels[name]])
        else:
            print(f"提示：图片 '{name}' 找到了，但在标签文件中没有找到对应的标签。")

    if not final_data:
        print("错误：找到了图片和标签，但没有任何一对能够匹配上。请检查图片文件名是否与Excel文件中的索引列完全一致（不包括.jpg等扩展名）。")
        return [], []

    # 分割训练集和测试集
    random.shuffle(final_data)
    split_point = int(len(final_data) * 0.8)
    train_set = final_data[:split_point]
    test_set = final_data[split_point:]

    print(f'成功加载并匹配数据：train_set is {len(train_set)}, test_set is {len(test_set)}')
    
    return train_set, test_set


def collate_RL(data):
    """
    自定义 collate 函数，用于打包图像、名称和标签。
    """
    if not data:
        return None, [], None
    
    imgs = [item[0] for item in data]
    names = [item[1] for item in data]
    labels = [torch.tensor(item[2], dtype=torch.float32) for item in data]
    
    return torch.stack(imgs, 0), names, torch.stack(labels, 0)


# ==============================================================================
#  第二部分：主程序执行代码
# ==============================================================================

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print(torch.cuda.is_available())
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    parser = argparse.ArgumentParser()
    
    # ... (你的所有 parser.add_argument 代码保持不变) ...
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
    parser.add_argument('--state_nums', type=int, default=77, help='the number of agents/dimensions')
    parser.add_argument('--pb_path', type=str, default='/home/dlg/li/OK/#2213_code/data/画/new_1/打分1.xlsx', help='the path to the pb file')
    parser.add_argument('--least_score', type=float, default=0.2, help='some score threshold')
    parser.add_argument('--photo_size', type=int, default=128, help='photo size')
    parser.add_argument('--Linear_nums', type=int, default=3, help='the number of categories to the last label')
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

    # --- 现在调用这些函数是绝对安全的 ---
    pb_info, pb_info_idx, pb_nd_update = get_all_pbs_dimension(opt.pb_path)
    paint_to_course, dimension_adj_martix = get_all_reflect_relation(opt.data_path)
    train_set, test_set = get_RL_data(opt.photo_size, opt.data_path)
    
    # 检查数据集是否为空
    if not train_set or not test_set:
        print("错误：训练集或测试集为空，请检查你的数据路径和文件！")
        exit()

    train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate_RL, pin_memory=True, num_workers=0, drop_last=True)
    test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=collate_RL, pin_memory=True, num_workers=0, drop_last=True)

    embedding_dim = int((opt.photo_size - opt.Kernel_size1) // opt.Stride1 + 1)
    embedding_dim = int((embedding_dim - 2) // 2 + 1)
    embedding_dim = int((embedding_dim - opt.Kernel_size2 + 2) // opt.Stride2 + 1)
    embedding_dim = int((embedding_dim - 2) // 2 + 1)
    embedding_dim = int(embedding_dim * embedding_dim * 3)

    model = AC_model(opt, embedding_dim, pb_info, pb_info_idx, pb_nd_update, dimension_adj_martix.cuda()).cuda()

    for epoch in range(opt.epoch):
        model.fit(train_data)
        if epoch == opt.epoch - 1:
            model.evaluate(test_data)
