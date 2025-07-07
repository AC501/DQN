import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np


# 定义条件扩散模型
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, cond_dim, num_steps=1000, target_condition=None):
        super(ConditionalDiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.num_steps = num_steps
        self.target_condition = target_condition

        # 定义UNet-like结构，用于去噪
        self.unet = UNet(input_dim, cond_dim)
        # 定义扩散过程的beta和alpha
        betas = torch.linspace(1e-4, 0.02, num_steps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)

    def forward(self, x, t, cond):
        noisy_x = self.noise_schedule(x, t)
        return self.unet(noisy_x, t, cond)

    # def noise_schedule(self, x, t):
    #     """根据时间步t将噪声添加到数据x中"""
    #     alpha_cumprod_t = self.alpha_cumprod[t]  # 这里alpha_cumprod_t是一个标量
    #     alpha_cumprod_t = alpha_cumprod_t.unsqueeze(1)  # 转换成(1,)维度
    #     alpha_cumprod_t = alpha_cumprod_t.expand(-1, x.size(1))  # 扩展为(1, input_dim)形状
    #
    #     noisy_x = torch.sqrt(alpha_cumprod_t) * x + torch.sqrt(1 - alpha_cumprod_t) * torch.randn_like(x)
    #     return noisy_x

    def noise_schedule(self, x, t):
        """简单的噪声添加方法"""
        # 生成一个与输入数据x形状相同的噪声
        noise = torch.randn_like(x)  # 生成与x形状相同的噪声
        noisy_x = x + noise * 0.1  # 使用一个固定的噪声标准差（例如 0.1）
        return noisy_x

    def generate_with_condition(self, cond, num_samples=64, num_steps=1000):
        """生成目标条件下的样本"""
        x = torch.randn(num_samples, self.input_dim).cuda()  # 初始化为随机噪声
        for t in reversed(range(num_steps)):
            x = self.forward(x, t, cond)

        # 调整生成的结果使其接近目标条件
        x = self.adjust_to_target_condition(x, cond)
        return x

    def adjust_to_target_condition(self, x, cond):
        """通过反向调整生成的结果，使其逼近目标条件"""
        adjustment_factor = 0.1  # 控制调整的强度
        adjusted_x = x + adjustment_factor * (self.target_condition - cond.unsqueeze(1))
        return adjusted_x


# 定义UNet-like去噪网络结构
class UNet(nn.Module):
    def __init__(self, input_dim, cond_dim):
        super(UNet, self).__init__()
        self.fc1 = nn.Linear(input_dim + cond_dim, 128)  # 拼接后的输入维度是 input_dim + cond_dim
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, input_dim)

        self.relu = nn.ReLU()

    def forward(self, x, t, cond):
        cond = cond.unsqueeze(1)
        # 拼接x和cond
        x = torch.cat((x, cond), dim=1)  # 拼接后得到 (64, 5)

        # # 将条件信息 cond 与输入特征 x 拼接
        # x = torch.cat((x, cond.unsqueeze(1).expand(-1, x.size(1))), dim=1)  # 拼接后维度应该是 (batch_size, input_dim + cond_dim)
        x = self.relu(self.fc1(x))  # 确保 fc1 的输入维度是 (batch_size, input_dim + cond_dim)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 定义数据集类，读取和预处理Excel数据
class ExcelDataset(Dataset):
    def __init__(self, file_path, condition_col=None, transform=None):
        self.data = pd.read_excel(file_path)  # 假设数据是Excel格式
        self.transform = transform

        # 假设所有列都是输入特征，除了条件列
        self.features = self.data.drop(columns=[condition_col]).values  # 除去条件列作为特征
        self.conditions = self.data[condition_col].values  # 作为条件列

        # # 标准化特征
        # self.scaler = StandardScaler()
        # self.features = self.scaler.fit_transform(self.features)

        # 转换为torch张量
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.conditions = torch.tensor(self.conditions, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.conditions[idx]


# 计算损失函数
def diffusion_loss(model, x, t, cond):
    predicted_x = model(x, t, cond)
    loss = nn.MSELoss()(predicted_x, x)  # 使用输入数据本身作为目标，进行自监督训练
    return loss


# 训练模型
def train_diffusion_model(model, train_loader, optimizer, num_epochs=10):
    model.train()
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data, cond) in enumerate(train_loader):
            data, cond = data.cuda(), cond.cuda()
            t = torch.randint(0, model.num_steps, (data.size(0),), device=data.device)  # 随机选择时间步
            optimizer.zero_grad()

            # 计算损失并反向传播
            loss = diffusion_loss(model, data, t, cond)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}')

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_epoch_loss}')

        # 如果当前epoch的loss优于最优loss，则保存模型参数
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_state = model.state_dict()
            print(f"New best model found, saving parameters...")

    # 保存最优模型参数
    if best_model_state is not None:
        torch.save(best_model_state, r'D:\tsc_ddqn_prb\weights\cdm_best_model.pth')
        print("Best model parameters saved to 'best_model.pth'.")


# 生成预测结果
def generate(model, cond, num_samples=64, num_steps=1000):
    model.eval()
    with torch.no_grad():
        return model.generate_with_condition(cond, num_samples, num_steps)


# 加载最优模型参数
def load_best_model(model, path=r'D:\tsc_ddqn_prb\weights\cdm_best_model.pth'):
    model.load_state_dict(torch.load(path))
    print(f"Model parameters loaded from {path}")


# 示例代码运行
def main():
    # 加载数据
    dataset = ExcelDataset(file_path=r"D:\tsc_ddqn_prb\data\模拟条件扩散模型数据.xlsx",
                           condition_col="y")  # 替换为您的Excel文件路径
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 初始化模型和优化器
    model = ConditionalDiffusionModel(input_dim=dataset.features.shape[1], cond_dim=1,
                                      target_condition=0.5).cuda()  # 目标条件值设定为0.5
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练模型并保存最优模型参数
    num_epochs = 10
    train_diffusion_model(model, train_loader, optimizer, num_epochs=num_epochs)

    # # 加载最优模型
    # load_best_model(model)

    # 生成最终预测结果
    target_condition = torch.tensor([0.5]).cuda()  # 设置条件目标值
    target_condition = target_condition.repeat(64)

    generated_samples = generate(model, target_condition, num_samples=64)
    print(generated_samples)


if __name__ == "__main__":
    main()
