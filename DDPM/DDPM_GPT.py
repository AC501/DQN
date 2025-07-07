import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from DDPM.DDPM_model_gpt import RewardDiffusionModel
import ast


device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

# 读取数据
reward_data = pd.read_excel("D:\\tsc_ddqn_prb_1Con_new\\data\\diffusion_data_ctde1s.xlsx")
reward_data = reward_data.iloc[:, 1:3]
reward_data['nex_state'] = reward_data['nex_state'].apply(
    lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x
)
states = reward_data['nex_state'].tolist()
rewards = reward_data['reward'].tolist()

states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
rewards_tensor = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(device)

# 创建数据加载器
dataset = TensorDataset(states_tensor, rewards_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型和优化器
model = RewardDiffusionModel(input_dim=len(states[0]), hidden_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 训练模型并保存最优参数
best_loss = float('inf')

# 训练函数
def train_diffusion_model(model, optimizer, data_loader, num_epochs=10):
    global best_loss
    best_model_path = "D:\\tsc_ddqn_prb_1Con_new\\weights\\best_model_ddpm_GPT_ctde1s.pth"
    model.train()
    for epoch in range(num_epochs):
        for states, rewards in data_loader:
            t = torch.randint(1, 10, (states.size(0),)).to(device)  # 随机时间步
            optimizer.zero_grad()

            # 正向扩散
            noisy_states = model(states, t)
            # 逆向扩散并预测奖励
            predicted_rewards = model.reverse(noisy_states, t)

            # 计算损失
            loss = nn.MSELoss()(predicted_rewards, rewards)  # 与真实奖励比较
            loss.backward()
            optimizer.step()
            print(f'第 [{epoch + 1}/{num_epochs}] 轮, 损失: {loss.item():.4f}')

            # 保存最优模型
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), best_model_path)
                print(f'保存最优模型，损失: {best_loss:.4f}')


train_diffusion_model(model, optimizer, data_loader, num_epochs=1000)
