import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# 自定义数据集
class ConditionalDataset(Dataset):
    def __init__(self, inputs, conditions):
        self.inputs = inputs
        self.conditions = conditions

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.conditions[idx]


# 简单的条件扩散模型网络
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim=128):
        super(ConditionalDiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, condition):
        # 检查 condition 是否是 1D 张量
        if condition.dim() == 1:
            condition = condition.view(-1, 1, 1)  # 添加两个新维度 [batch_size, 1, 1]

        # 扩展 condition 的形状以匹配 x
        condition = condition.expand(x.size(0), x.size(1), x.size(2))  # 广播到 [batch_size, 5, 24]

        print('condition',condition.shape)
        print('x',x.shape)

        # 拼接 x 和 condition
        x = torch.cat([x, condition], dim=-1)
        print('x1',x.shape)
        print('x2',x)
        x = self.net(x)
        print('/---/-/-/-/-/')
        return x


# 定义扩散过程的函数
class Diffusion:
    def __init__(self, model, timesteps=1000, device="cpu"):
        self.model = model
        self.timesteps = timesteps
        self.device = device

        # 线性噪声调度
        self.beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def forward_diffusion(self, x0, t):
        """正向扩散，添加噪声"""
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t]).view(-1, 1, 1)
        noise = torch.randn_like(x0)
        xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return xt, noise

    def reverse_diffusion(self, x_t, condition, t):
        """逆向扩散，去除噪声"""
        pred_noise = self.model(x_t, condition)
        alpha_t = self.alpha[t].view(-1, 1, 1)
        beta_t = self.beta[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - self.alpha[t]).view(-1, 1, 1)

        x_prev = (1 / torch.sqrt(alpha_t)) * (x_t - beta_t / sqrt_one_minus_alpha_t * pred_noise)
        return x_prev


# 数据处理函数
def prepare_data(file_path, window_size=5):
    import pandas as pd
    import numpy as np

    # 读取数据
    data = pd.read_excel(file_path)

    # 检查并强制转换必要列的数据类型
    required_columns = ['state_0', 'state_1', 'state_2', 'state_3', 'y']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"缺少必要的列：{col}")

    # 尝试将 state_x 和 y 列转换为数值类型
    for col in required_columns:
        data[col] = data[col].apply(
            lambda x: np.array(eval(x), dtype=np.float32) if isinstance(x, str) else np.array(x, dtype=np.float32)
        )

    states = data[['state_0', 'state_1', 'state_2', 'state_3']].values
    y = data['y'].values.astype(np.float32)  # 转换 y 为浮点数

    # 拼接 states
    states_flat = []
    for row in states:
        try:
            # 保证每一个 state 是数值数组
            states_flat.append(np.concatenate([np.array(state, dtype=np.float32) for state in row]))
        except Exception as e:
            print(f"错误拼接行：{row}，错误信息：{e}")
            continue

    # 构造滑动窗口的输入和条件
    inputs, conditions = [], []
    for i in range(len(states_flat) - window_size + 1):
        input_data = states_flat[i:i + window_size]
        condition_value = y[i + window_size - 1]
        inputs.append(input_data)
        conditions.append(condition_value)

    inputs = np.array(inputs)
    conditions = np.array(conditions)

    # 保存为 Excel 文件
    inputs_flat = inputs.reshape(inputs.shape[0], -1)  # 展平成二维数组
    df = pd.DataFrame(inputs_flat)
    df['condition'] = conditions  # 添加条件列
    output_path =r"D:\tsc_ddqn_prb\data\test_CPMData.xlsx"
    df.to_excel(output_path, index=False)
    return inputs, conditions


# 训练模型
def train_model(model, dataloader, diffusion, optimizer, epochs=10, device="cpu"):
    model.train()
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        epoch_loss = 0
        for x, condition in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x, condition = x.to(device), condition.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),)).to(device)
            x_t, noise = diffusion.forward_diffusion(x, t)

            pred_noise = model(x_t, condition)
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")


# 基于条件生成数据
def generate_data(diffusion, condition_value, num_samples, input_dim, device="cpu"):
    model = diffusion.model
    model.eval()

    condition = torch.tensor([condition_value] * num_samples, dtype=torch.float32).to(device)
    x_t = torch.randn((num_samples, 5, input_dim)).to(device)

    with torch.no_grad():
        for t in range(diffusion.timesteps - 1, -1, -1):
            t_tensor = torch.tensor([t] * num_samples).to(device)
            x_t = diffusion.reverse_diffusion(x_t, condition.unsqueeze(-1), t_tensor)

    return x_t.cpu().numpy()


# 主程序
if __name__ == "__main__":
    file_path = r"D:\tsc_ddqn_prb\data\CPMData_final.xlsx"
    # 加载数据
    inputs, conditions = prepare_data(file_path)
    dataset = ConditionalDataset(torch.tensor(inputs, dtype=torch.float32), torch.tensor(conditions, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 模型参数
    input_dim = inputs.shape[-1]
    condition_dim = 1
    model = ConditionalDiffusionModel(input_dim=input_dim, condition_dim=condition_dim).to("cpu")
    diffusion = Diffusion(model, timesteps=1000, device="cpu")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
    train_model(model, dataloader, diffusion, optimizer, epochs=10, device="cpu")

    # 基于条件生成数据
    generated_data = generate_data(diffusion, condition_value=0, num_samples=4000, input_dim=input_dim, device="cpu")
    print("生成数据形状:", generated_data.shape)

    # 保存生成的数据
    np.savetxt("generated_data.csv", generated_data.reshape(4000, -1), delimiter=",")
