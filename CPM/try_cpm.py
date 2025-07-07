import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


file_path = r"D:\tsc_ddqn_prb\data\CPMData_final.xlsx"
data = pd.read_excel(file_path)

# 将列解析为数值数组
def parse_state_column(column):
    return [list(map(float, row.split(', '))) for row in column]

states = []
for col in ['state_0', 'state_1', 'state_2', 'state_3']:
    states.append(parse_state_column(data[col]))

# 合并所有状态列
states = np.concatenate(states, axis=1)  # 合并为一个大的状态矩阵
y = data['y'].values


# 滑动窗口处理输入数据
def create_sliding_window(states, y, window_size):
    inputs = []
    conditions = []
    for i in range(len(states) - window_size + 1):
        inputs.append(states[i:i + window_size])  # 滑动窗口
        conditions.append(y[i + window_size - 1])  # 取窗口最后的条件
    return np.array(inputs), np.array(conditions)

window_size = 5
input_states, input_y = create_sliding_window(states, y, window_size)
print('input_states',input_states)
# 转换为 Tensor
states_tensor = torch.tensor(input_states, dtype=torch.float32)
y_tensor = torch.tensor(input_y, dtype=torch.float32)


