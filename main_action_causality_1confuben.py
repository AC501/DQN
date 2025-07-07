from openpyxl.reader.excel import load_workbook

from environment.sumotl_action_causality_yuanshi import SUMOTrafficLights
from learner.q_learning import QLearner
from exploration.epsilon_greedy import EpsilonGreedy
import datetime
import warnings
from replay import ReplayBuffer
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import torch.optim as optim
from RewardFunction.FunctionFitting_fuben3 import rewardfitmain
from CPM.cpmDataStructure import process_cpm_data
from datetime import datetime
import time
import  matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore')




# 将列解析为数值数组
def parse_state_column(column):
    return [list(map(float, str(row).split(','))) for row in column]



# 滑动窗口处理输入数据
def create_sliding_window(states_rewards, y, window_size):
    inputs = []
    conditions = []
    for i in range(len(states_rewards) - window_size + 1):
        inputs.append(states_rewards[i:i + window_size])  # 滑动窗口
        # sum_y = np.sum(y[i:i + window_size])  # 计算每个窗口内y值的总和
        conditions.append(y[i:i + window_size])
    return np.array(inputs), np.array(conditions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 数据集和 DataLoader
class TrafficDataset(Dataset):
    def __init__(self, states_rewards, conditions):
        self.states = states_rewards
        self.conditions = conditions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.conditions[idx]



#定义完整得扩散模型
class CustomDDPMScheduler:
    def __init__(
            self,
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            device="cuda"
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.device = device

        # 初始化 beta 序列，并移动到指定设备
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps).to(device)

        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
            self.betas = self.betas.to(device)
        else:
            raise ValueError(f"未知的 beta schedule: {beta_schedule}")

        # 计算其他参数并移动到指定设备
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    def add_noise(self, original_samples, noise, timesteps):
        """添加噪声到样本中"""
        # 确保所有输入都在正确的设备上
        timesteps = timesteps.to(self.device)
        original_samples = original_samples.to(self.device)
        noise = noise.to(self.device)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1)

        # print(sqrt_alphas_cumprod_t.shape)
        # print(original_samples.shape)
        # 扩展 sqrt_alphas_cumprod_t 的维度，确保其可以与 original_samples 进行广播
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(1)  # 变为 [32, 1, 1]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(1)  # 变为 [32, 1, 1]


        # 将 sqrt_alphas_cumprod_t 广播到 [32, 5, 24] 形状
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.expand(-1, original_samples.shape[1], original_samples.shape[2])
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.expand(-1, original_samples.shape[1],
                                                                                 original_samples.shape[2])

        noisy_samples = sqrt_alphas_cumprod_t * original_samples + \
                        sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_samples

    def step(self, model_output, timestep, sample):
        """执行一步去噪过程"""
        t = timestep
        prev_t = t - 1 if t > 0 else t

        # 获取当前时间步的 alpha 值
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_prod_t = 1 - alpha_prod_t

        # 计算去噪系数
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # 计算方差
        variance = 0
        if t > 0:
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]

        # 如果不是最后一步，添加噪声
        if t > 0:
            noise = torch.randn_like(sample)
            pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + \
                               variance ** 0.5 * noise
        else:
            pred_prev_sample = pred_original_sample

        return type('PrevSampleOutput', (), {'prev_sample': pred_prev_sample})()

    def set_timesteps(self, num_inference_steps):
        """设置推理时的时间步"""
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long)





class DiffusionModel(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim=512, num_heads=8, num_lstm_layers=1,dropout = 0.2):
        super(DiffusionModel, self).__init__()
        # 修改：2023-11-01 - 调整输入维度以适应单个交通信号灯
        # 原因：由于只处理一个交通信号灯的数据，需要调整模型结构
        self.input_dim = input_dim + condition_dim + 1  # 新增+1维度给timesteps
        self.hidden_dim = hidden_dim

        # 添加Dropout层
        self.dropout = nn.Dropout(p=dropout)

        # 修改LSTM定义
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0  # 多层LSTM时启用
        )

        # 自注意力机制
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # 最后的线性层
        # 修改：2023-11-01 - 修改输出维度为1
        # 原因：由于只处理一个交通信号灯的数据，输出维度需要从4改为1
        self.fc1 = nn.Linear(self.hidden_dim, 1)  # 输出维度改为1

    def forward(self, x, condition, timesteps):  # 新增timesteps参数
        # 处理时间步维度 [batch_size] -> [batch_size, seq_len, 1]
        timesteps = timesteps.unsqueeze(-1).unsqueeze(-1).expand(-1, x.shape[1], 1)

        # 合并 x（状态）、condition（条件）、timesteps（时间步）
        combined = torch.cat([x, condition, timesteps], dim=2)  # 形状 [batch, seq_len, input_dim+condition+1]
        print('combined',combined.shape)
        # LSTM处理（添加Dropout）
        lstm_out, _ = self.lstm(combined)
        lstm_out = self.dropout(lstm_out)  # 新增

        # 自注意力处理（添加Dropout）
        attn_output, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_output = self.dropout(attn_output)  # 新增

        # 最终输出
        output = self.fc1(attn_output)
        print('output',output.shape)
        return output




# 定义扩散过程中的噪声函数
def add_noise(x, sigma,device):
    noise = torch.randn_like(x).to(device)
    return x + sigma * noise


# 训练扩散模型的函数
def train_diffusion_model(model, dataloader, optimizer, num_epochs, device):
    # 用于记录训练损失
    losses = []
    if os.path.exists("CPM/BestDDPModel.pth"):
        model.load_state_dict(torch.load("CPM/BestDDPModel.pth"))

    # 使用自定义噪声调度器，并传入device
    noise_scheduler = CustomDDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )
    # 设置模型为训练模式
    model.train()
    best_loss = float('inf')  # 初始化最小损失值为正无穷
    best_model_path = "CPM/BestDDPModel.pth"

    # 开始训练循环
    for epoch in range(num_epochs):
        epoch_losses = []  # 记录每个epoch的损失
        epoch_loss = 0
        # 遍历数据加载器
        for batch_x, batch_c in dataloader:
            # 将数据移到指定设备
            batch_x, batch_c = batch_x.to(device), batch_c.to(device)
            # 随机采样时间步
            timesteps = torch.randint(0, 1000, (batch_x.shape[0],), device=device)
            # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_x.shape[0],), device=device)
            # 生成随机噪声
            noise = torch.randn_like(batch_x).to(device)
            # 添加噪声到输入数据
            noisy_x = noise_scheduler.add_noise(batch_x, noise, timesteps)

            # 预测噪声
            pred_noise = model(noisy_x, batch_c, timesteps)
            # 计算均方误差损失
            loss = torch.mean((pred_noise - noise) ** 2)

            # 反向传播和优化
            optimizer.zero_grad()  # 清除梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

            # 记录损失
            epoch_losses.append(loss.item())

        # 计算平均损失
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)

        # 每10个epoch打印一次损失
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # 如果当前epoch的损失小于最优损失，则保存模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)  # 保存最优模型
            print(f"模型参数已保存，当前最佳损失：{best_loss:.4f}")


# 新增函数：计算认知不确定性
def compute_epistemic_uncertainty(model, condition, num_models=5, num_samples=10, device="cuda"):
    """
    通过多次采样模型预测结果计算认知不确定性
    :param condition: 条件张量 [1, condition_dim]
    :param num_models: 虚拟模型数量（通过MC Dropout实现）
    :param num_samples: 每个模型的采样次数
    :return: 认知不确定性值（标量）
    """
    model.train()  # 启用Dropout
    all_means = []

    with torch.no_grad():
        for _ in range(num_models):
            # 生成多组样本
            samples = generate_new_samples(model, condition, num_samples, device)
            # 计算每批样本的均值（假设关注最后一个维度的平均值）
            batch_means = torch.mean(samples, dim=1)  # [num_samples, feature_dim]
            all_means.append(batch_means.cpu().numpy())

    # 计算不同模型预测均值的方差
    all_means = np.concatenate(all_means, axis=0)
    epistemic_uncertainty = np.var(all_means, axis=0).mean()  # 取特征维度平均
    return epistemic_uncertainty


# 新增函数：构建候选条件集
def build_candidate_conditions(current_max_y, scale_factors=[0.6, 0.8, 1.2, 1.4, 1.8]):
    """
    基于当前最大目标值构建候选条件集
    :param current_max_y: 当前数据集中的最大y值（标量）
    :return: 候选条件张量 [num_candidates, 1]
    """
    candidates = [factor * current_max_y for factor in scale_factors]
    return torch.tensor(candidates, dtype=torch.float32).unsqueeze(-1)  # [num_candidates, 1]

def generate_new_samples(model, condition, num_samples, device):
    """
    生成新的样本数据
    修改：2023-11-01 - 优化条件张量处理逻辑
    原因：解决条件张量维度不匹配问题，支持不同维度的输入条件
    """
    # 如果存在预训练模型则加载
    if os.path.exists("CPM/BestDDPModel.pth"):
        model.load_state_dict(torch.load("CPM/BestDDPModel.pth"))

    # 使用自定义噪声调度器，并传入device
    noise_scheduler = CustomDDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )

    # 设置推理时间步
    noise_scheduler.set_timesteps(num_inference_steps=1000)
    # 设置模型为评估模式
    model.eval()
    
    # 修改：2023-11-01 - 优化噪声张量生成
    # 原因：确保生成的噪声张量维度与模型输入要求一致
    x = torch.randn(num_samples, 100).to(device)
    x = x.unsqueeze(-1)  # shape: [num_samples, 100, 1]

    # 修改：2023-11-01 - 增加条件张量维度检查和处理
    # 原因：处理不同维度的条件输入，确保与模型输入要求匹配
    if condition.dim() == 3:  # 如果条件已经是 [batch, seq_len, feature] 格式
        condition_expanded = condition.to(device)
    else:  # 如果条件需要扩展
        condition_expanded = condition.unsqueeze(0).expand(num_samples, 100, -1).to(device)

    # 逐步去噪过程
    generated_samples = []
    with torch.no_grad():  # 不计算梯度
        for t in range(1000 - 1, -1, -1):
            timesteps = torch.full((num_samples,), t, device=device)
            # 预测噪声
            pred_noise = model(x, condition_expanded, timesteps)
            # 使用调度器进行去噪
            x = noise_scheduler.step(pred_noise, t, x).prev_sample
        generated_samples.append(x)

    # 将生成的样本拼接成一个张量
    generated_samples = torch.cat(generated_samples, dim=0)
    return generated_samples


def main():
    global generated_samples
    epoches = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hc_means_file = 'data/hc_means.xlsx'
    plot_file = 'data/hc_means_plot.png'
    best_con_file = 'data/best_con.xlsx'
    
    # 新增：2023-11-01 - 定义时间周期数量
    num_periods = 5  # 定义5个不同的100秒周期
    
    # 确定断点位置
    start_epoch = 0
    if os.path.exists(hc_means_file):
        try:
            existing_df = pd.read_excel(hc_means_file)
            if not existing_df.empty and 'Epoch' in existing_df.columns:
                start_epoch = existing_df['Epoch'].max() + 1
                print(f"检测到历史记录，将从 epoch={start_epoch} 开始")
        except Exception as e:
            print(f"读取历史记录失败: {str(e)}，从头开始运行")


    for epoch in range(start_epoch,epoches):
        starttime = datetime.now().time()
        start_seconds = starttime.hour * 3600 + starttime.minute * 60 + starttime.second
        print("epoch =", epoch, "====================")
        
        # 新增：2023-11-01 - 计算当前周期号
        current_period = epoch % num_periods
        print(f"当前使用时间周期: {current_period}")
        
        # 创建SUMO环境
        env = SUMOTrafficLights('nets/22grid_fuben_e1.sumocfg', 8813, False, 32)
        # 创建经验回放缓冲区
        learners = {}
        replay_buffers = {}
        # 为每个交通信号灯创建学习者
        for tlID in env.get_trafficlights_ID_list():
            replay_buffers[tlID] = ReplayBuffer(capacity=10000)
            learners[tlID] = QLearner(tlID, env, 0, 0, 0.01, 0.01, 'train', replay_buffers[tlID], 15, 1.0, 0.1, 0.001,
                                      5,
                                      2, 32, "weights/best_model.pth")
        # 设置 learners 和 replay_buffers
        env.learners = learners
        env.replay_buffers = replay_buffers
        
        # 新增：2023-11-01 - 告诉环境当前周期号
        env.current_period = current_period
        
        # 设置回合数
        n_episodes = 1
        # 运行回合
        for i in range(n_episodes):
            # print("Episode =", i+1, "====================")
            arq_avg_nome = 'tl_%d.txt' % i
            arq_tl = open(arq_avg_nome, 'w')
            arq_tl.writelines('##%s## \n' % datetime.now().time())

            env.run_episode(110, None)

            arq_tl.close()


        # 动态确定n_episodes
        # cpm_final_path = 'data/CPMData_final.xlsx'
        # try:
        #     if os.path.exists(cpm_final_path):
        #         df = pd.read_excel(cpm_final_path)
        #         current_rows = len(df)
        #     else:
        #         current_rows = 0
        # except Exception as e:
        #     print(f"读取{cpm_final_path}失败: {e}, 将current_rows设为0")
        #     current_rows = 0
        #
        # # 计算需要运行的episode次数
        # if current_rows >= 80000:
        #     n_episodes = 1
        # else:
        #     needed = 80000 - current_rows
        #     n_episodes = (needed + 99) // 100  # 向上取整
        #
        # print(f"当前数据行数: {current_rows}, 需要运行 {n_episodes} 个episodes")
        #
        # # 运行回合
        # for i in range(n_episodes):
        #     arq_avg_nome = f'tl_{i}.txt'
        #     with open(arq_avg_nome, 'w') as arq_tl:
        #         arq_tl.writelines(f'##{datetime.now().time()}## \n')
        #         env.run_episode(110, None)

        #训练条件扩散模型和拟合奖励函数
        input_filepath = 'data/CPMData.xlsx'
        output_filepath = 'data/CPMData_final.xlsx'
        process_cpm_data(input_filepath, output_filepath)
        # 读取state和reward数据
        SRData = pd.read_excel('data/CPMData_final.xlsx')

        # 创建新的列来保存每列的最后一个元素
        #保存reward
        # 修改：2023-11-01 - 只处理一个交通信号灯的数据
        SRData['reward_0'] = SRData['state_reward_0'].apply(lambda x: x.split(',')[-1].strip())

        # 删除每列的最后一个元素和倒数第二个元素
        SRData['state_reward_0'] = SRData['state_reward_0'].apply(lambda x: ','.join(x.split(',')[:-1]).strip())

        # 读取对应时间内区域产生的车辆数
        ProductData = pd.read_excel('data/produced_vehicles.xlsx')
        # 读取对应时间内区域离开的车辆数
        LeaveData = pd.read_excel('data/departed_vehicles.xlsx')
        #惩罚因子
        M = 5
        # 计算进出车辆总和是否为0（返回布尔值，True/False对应1/0）
        is_congested = (ProductData["Produced Vehicles"] + LeaveData["Departed Vehicles"]) <= 1

        # 计算拥堵指标
        ProductData["HC"] = (
                LeaveData["Departed Vehicles"] - ProductData["Produced Vehicles"]  # 正常差值
                - M * is_congested  # 仅当完全堵死时减去M
        )

        SRData['y'] = ProductData['HC']
        # 计算当前 epoch 的 HC 平均值（使用原始数据，未过滤）
        hc_mean = ProductData['HC'].mean()

        # 保存到 Excel（追加模式，保持不变）
        new_row = pd.DataFrame({'Epoch': [epoch], 'HC_Mean': [hc_mean]})
        if os.path.exists(hc_means_file):
            existing_df = pd.read_excel(hc_means_file)
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        else:
            updated_df = new_row
        updated_df.to_excel(hc_means_file, index=False)

        # 使用筛选后的数据构造 rewards 和 cond
        rewards = []
        # 修改：2023-11-01 - 只处理一个reward列
        for col in ['reward_0']:
            rewards.append(parse_state_column(SRData[col]))

        cond = []
        for col in ['y']:
            cond.append(parse_state_column(SRData[col]))

        rewards = np.concatenate(rewards, axis=1)  # 合并为一个大的状态矩阵
        y = np.concatenate(cond,axis=1)
        # window_size = 5
        # input_rewards, input_y = create_sliding_window(rewards, y, window_size)

        # 修改后的数据处理部分（替换原来的滑动窗口代码）
        block_size = 100
        n_samples = rewards.shape[0]
        num_blocks = n_samples // block_size

        # 截断数据以匹配块大小
        rewards = rewards[:num_blocks * block_size, :]
        y = y[:num_blocks * block_size, :]
        # 分割数据为块并计算y的平均
        input_rewards = rewards.reshape((num_blocks, block_size, rewards.shape[1]))
        input_y = y.reshape((num_blocks, block_size, -1)).mean(axis=1)

        # 转换为 Tensor
        rewards_tensor = torch.tensor(input_rewards, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(input_y, dtype=torch.float32).to(device)
        y_tensor = y_tensor.unsqueeze(1)
        y_tensor = y_tensor.repeat(1,rewards_tensor.size(1),1)

        # 超参数定义
        input_dim = input_rewards.shape[2]
        condition_dim = y_tensor.shape[2]
        hidden_dim = 256
        num_epochs = 300
        batch_size = 32
        learning_rate = 0.0001
        num_samples = 1  # 生成样本的数量
        # 创建模型、定义优化器
        model = DiffusionModel(input_dim, condition_dim, hidden_dim,dropout=0.3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


        # 将数据集包装成DataLoader
        dataset = TrafficDataset(rewards_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size, shuffle=True)

        # 预训练扩散模型
        train_diffusion_model(model, data_loader, optimizer, num_epochs, device)

        # === 修改后 ===
        # 获取当前数据集中的最大y值
        current_max_y = torch.max(y_tensor).item()
        print('torch.max(y_tensor).item()',torch.max(y_tensor).item())
        
        # 修改：2023-11-01 - 为每个周期生成不同的奖励数据
        for period in range(num_periods):
            # 为每个周期设置稍微不同的条件
            period_conditions = y_tensor.clone()
            period_conditions[-1] = current_max_y + period * 0.5  # 为每个周期调整条件
            
            # 生成这个周期的奖励数据
            period_samples = generate_new_samples(model, period_conditions, num_samples, device)
            
            # 处理生成的数据
            period_samples_cpu = period_samples.cpu().numpy()
            period_samples_cpu = np.round(period_samples_cpu).astype(int)
            # 修改：2023-11-01 - 修改为1列数据
            reshaped_data = period_samples_cpu.reshape(-1, 1)
            
            # 创建DataFrame
            # 修改：2023-11-01 - 只保留一个reward列
            df = pd.DataFrame(reshaped_data, columns=["reward0"])
            
            # 保存到对应周期的文件
            file_path = f'data/generated_reward_period_{period}.xlsx'
            df.to_excel(file_path, index=False)
            print(f"已为周期 {period} 生成奖励文件: {file_path}")

        # 最佳条件保存到 Excel（追加模式，保持不变）
        new_row1 = pd.DataFrame({'Epoch': [epoch], 'Best_Gen_Con': [current_max_y]})
        if os.path.exists(best_con_file):
            existing_df1 = pd.read_excel(best_con_file)
            updated_df1 = pd.concat([existing_df1, new_row1], ignore_index=True)
        else:
            updated_df1 = new_row1
        updated_df1.to_excel(best_con_file, index=False)

        # Convert to DataFrame
        # 修改：2023-11-01 - 只保留一个reward列
        df = pd.DataFrame(reshaped_data, columns=["reward0"])
        # Save to Excel
        file_path = 'data/generated_reward.xlsx'
        df.to_excel(file_path, index=False)

        # 调用 FunctionFitting.py 进行状态奖励数据处理和模型训练
        # rewardfitmain()
        endtime = datetime.now().time()
        end_seconds = endtime.hour * 3600 + endtime.minute * 60 + endtime.second
        # 计算秒数差
        spendtime_seconds = end_seconds - start_seconds
        print(f'持续时间：{spendtime_seconds}秒')

        # 所有epoch结束后绘制折线图
    if os.path.exists(hc_means_file):
        hc_means_df = pd.read_excel(hc_means_file)
        plt.figure(figsize=(10, 6))
        plt.plot(hc_means_df['Epoch'], hc_means_df['HC_Mean'],
                 marker='o', markersize=10, linestyle='-', color='b')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Average HC', fontsize=12)
        plt.title('Average HC per Epoch', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(hc_means_df['Epoch'])  # 明确显示每个epoch的刻度
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.show()
    else:
        print("未找到HC平均值数据文件，无法绘图。")



if __name__ == '__main__':
    main()
