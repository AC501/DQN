import torch
from full_ddpm import MLPDiffusion
import pandas as pd
import ast
from torch.utils.data import DataLoader, TensorDataset



#超参数
num_steps = 100

#制定每一步的beta
betas = torch.linspace(-6,6,num_steps)
betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

#计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
alphas = 1-betas
alphas_prod = torch.cumprod(alphas,0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
==one_minus_alphas_bar_sqrt.shape
print("all the same shape",betas.shape)

def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps, device):
    batch_size = x_0.shape[0]

    # 对一个batchsize样本生成随机的时刻t
    t = torch.randint(0, n_steps, size=(batch_size // 2,))
    t = torch.cat([t, n_steps - 1 - t], dim=0)
    t = t.unsqueeze(-1).to(device)  # 将 t 移动到 GPU

    # x0的系数
    a = alphas_bar_sqrt[t].to(device)  # 将 a 移动到 GPU

    # eps的系数
    aml = one_minus_alphas_bar_sqrt[t].to(device)  # 将 aml 移动到 GPU

    # 生成随机噪音eps
    e = torch.randn_like(x_0).to(device)  # 将 e 移动到 GPU

    # 构造模型的输入
    x = x_0 * a + e * aml

    # 送入模型，得到t时刻的随机噪声预测值
    output = model(x, t.squeeze(-1))

    # 与真实噪声一起计算误差，求平均值
    return (e - output).square().mean()


seed = 1234
print('Training model...')
batch_size = 128
reward_data = pd.read_excel("D:\\tsc_ddqn_prb_1Con_new\\data\\diffusion_data.xlsx")
reward_data = reward_data.iloc[:, 1:3]
# 转换为Tensor
# 确保所有的 nex_state 都是列表，而不是字符串
reward_data['nex_state'] = reward_data['nex_state'].apply(
    lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x)
print('reward_data[]', reward_data['nex_state'])
# 提取状态和奖励
states = reward_data['nex_state'].tolist()
rewards = reward_data['reward'].tolist()
states_tensor = torch.tensor(states, dtype=torch.float32)
rewards_tensor = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)

# 创建数据加载器
dataset = TensorDataset(states_tensor, rewards_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
num_epoch = 4000

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPDiffusion(num_steps).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
best_loss = float('inf')
for t in range(num_epoch):
    best_model_path = "D:\\tsc_ddqn_prb_1Con_new\\weights\\best_model_fullddpm.pth"
    for states, rewards in data_loader:
        states = states.to(device)
        optimizer.zero_grad()
        loss = diffusion_loss_fn(model, states, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps,device)
        loss.backward()
        optimizer.step()

        # 每100个epoch打印一次损失
        if (t + 1) % 100 == 0:
            print(f'Epoch [{t + 1}/{num_epoch}], Loss: {loss.item():.4f}')

        # 保存最优模型
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model with loss: {best_loss:.4f}')

