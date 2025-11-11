import torch
import numpy as np

def evaluate(score, label):
    """
    计算模型预测的评估指标。
    这个函数原本在 RL_model.py 中叫做 test。
    """
    score, label = torch.tensor(score), torch.tensor(label)
    mse = torch.mean(torch.abs(score - label) ** 2)
    mae = torch.mean(torch.abs(score - label))
    rmse = np.sqrt(mse.cpu().detach().numpy())  # 确保在转换前将tensor移到cpu
    
    return mae, mse, rmse
