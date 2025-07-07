import socket
import torch
import torch.nn as nn
import os
import ast
import random
import math
import torch.optim as optim
import numpy as np
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self._capacity = capacity
        self._storage = []
        self._num_added = 0
        self._priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # 用于控制优先级的平滑程度，alpha=0时退化为均匀采样

    def add(self, state, next_state, action, reward, td_error: float):
        """添加样本，并根据TD-error计算优先级"""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        action = torch.tensor(action, dtype=torch.int64).unsqueeze(0).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(device)

        transition = Transition(state, action, next_state, reward)

        if len(self._storage) < self._capacity:
            self._storage.append(transition)
        else:
            self._storage[self._num_added % self._capacity] = transition

        # 优先级设置为TD-error的幂次，确保为正
        self._priorities[self._num_added % self._capacity] = (abs(td_error) + 1e-5) ** self.alpha
        self._num_added += 1

    def sample(self, batch_size: int, beta: float = 0.4):
        """根据优先级进行样本采样"""
        if len(self._storage) == 0:
            return None

        # 计算采样的概率分布
        priorities = self._priorities[:len(self._storage)]
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self._storage), batch_size, p=probabilities)
        batch = [self._storage[i] for i in indices]

        # 计算权重 (importance sampling weight)
        total = len(self._storage)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # 归一化

        batch = Transition(*zip(*batch))  # 解包为Transition命名元组
        return batch, torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device), indices

    def update_priorities(self, indices, td_errors):
        """更新样本的优先级"""
        for i, td_error in zip(indices, td_errors):
            self._priorities[i] = (abs(td_error) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self._storage)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return min(self._num_added, self._capacity)

    @property
    def steps_done(self) -> int:
        return self._num_added


# 强化学习智能体
class Agent:
    def __init__(self, model_path=None, cuda=False,random_threshold=0.5, eps_start=1.0, eps_end=0.1, eps_decay=0.001,
                 target_update=15,gamma=0.01,batch_size=32,buffer_capacity = 1000):
        self.policy_net = DQN()
        self.target_net = DQN()
        if model_path and os.path.exists(model_path):
            self.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.cuda = cuda
        if cuda:
            self.policy_net = self.policy_net.cuda()
            self.target_net = self.target_net.cuda()
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.random_threshold = random_threshold
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = 0.01
        self.batch_size = 32
        self.steps_done = 0
        self.model = 'train'
        self.target_update = target_update
        self.learn_steps = 0
        self.best_model_path = "./best_model_aibox.pth"

        self.best_loss = float('inf')
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.last_action = 0

    def get_action_randomly(self):
        action = 0 if random.random() < self.random_threshold else 1
        return action

    def get_optim_action(self, state):
        if self.cuda:
            state = torch.tensor(state, dtype=torch.float32).cuda()
        else:
            state = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            q_value = self.policy_net(state)

        _, action_index = torch.max(q_value, dim=0)
        action = action_index.cpu().item()
        return action

    def get_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        # if random.random() <= eps_threshold:
        #     act = self.get_action_randomly()
        act = self.get_optim_action(state)

        return act

    def learn(self, experiences):
        if self.model == 'train':
            print('---------------------')
            loss_fn = nn.MSELoss()

            if self.replay_buffer.steps_done <= 10:
                return

            batch, _, _ = self.replay_buffer.sample(self.batch_size)

            state_batch = torch.cat(batch.state).float().to(device)
            action_batch = torch.cat(batch.action).view(self.batch_size, 1).to(device)
            next_state_batch = torch.cat(batch.next_state).float().to(device)
            reward_batch = torch.cat(batch.reward).view(self.batch_size, 1).float().to(device)
            # print(f"state:{state_batch},next_state:{next_state_batch},action:{action_batch},reward:{reward_batch}")

            num_actions = self.policy_net(state_batch).size(1)
            if torch.any(action_batch >= num_actions):
                print("Invalid action index found in action_batch!")
                print("action_batch:", action_batch)
                return

            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Double DQN 的目标Q值计算：选择动作使用 policy_net，评估动作使用 target_net
            with torch.no_grad():
                argmax_actions = self.policy_net(next_state_batch).max(1)[1].view(self.batch_size, 1)  # 通过行为网络选择最大动作
                next_state_values = self.target_net(next_state_batch).gather(1, argmax_actions)  # 使用目标网络计算Q值
                expected_state_action_values = reward_batch + self.gamma * next_state_values
            # #DQN
            # with torch.no_grad():
            #     argmax_action = self.policy_net(next_state_batch).max(1)[1].view(self.batch_size, 1)
            #     expected_state_action_values = reward_batch + self.gamma * self.target_net(next_state_batch).gather(1, argmax_action)

            loss = loss_fn(state_action_values, expected_state_action_values)

            self.optimizer.zero_grad()
            loss.backward()

            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optimizer.step()
            self.learn_steps += 1
            # if self.learn_steps % self.target_update == 0:
            #     self.target_net.load_state_dict(self.policy_net.state_dict())
            #     time = str(datetime.now()).split('.')[0]
            #     time = time.replace('-', '').replace(' ', '_').replace(':', '')
            #     torch.save(self.policy_net.state_dict(), 'weights/weights_{0}_{1}.pth'.format(time, self.learn_steps))

            if self.learn_steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if self.learn_steps == 1 or loss.item() < self.best_loss:
                self.best_loss = loss.item()
                torch.save(self.policy_net.state_dict(), self.best_model_path)

    def act_last(self, state):
        action = self.get_action(state)
        self.last_action = action  # Store the last action
        return state, action

    def feedback_last(self, reward, next_state, state):
        # Convert state and next_state to tensors if they are not already
        if isinstance(state, list):
            state = torch.tensor(state, dtype=torch.float32).to(device)

        if isinstance(next_state, list):
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

        # Get current Q-values from policy network
        current_q_values = self.policy_net(state)

        # Handle 1D and 2D cases
        # 获取当前状态的Q值
        if current_q_values.dim() == 1:  # 1D情形
            current_q = current_q_values[self.last_action].item()  # 使用.item()提取标量
        else:  # 2D情形
            current_q = current_q_values.gather(1, torch.tensor([[self.last_action]], dtype=torch.int64).to(
                device)).squeeze(1).item()  # 使用.item()提取标量

        # Get Q-values for next state from target network
        next_q_values = self.target_net(next_state)
        next_max_q = torch.max(next_q_values).item()

        # Calculate TD error
        td_error = reward + self.gamma * next_max_q - current_q


        # Add experience to replay buffer
        self.replay_buffer.add(state, next_state, self.last_action, reward, td_error)

        # Sample from replay buffer if it's large enough
        if len(self.replay_buffer) > self.batch_size:
            experiences = self.replay_buffer.sample(self.batch_size)
            self.learn(experiences)

    def update_target_network(self):
        # 将策略网络的参数复制到目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("Target network updated!")


# 建立Socket服务器
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = '222.18.156.234'
port = 8000

serversocket.bind((host, port))

serversocket.listen(5)
agent = Agent(model_path='/home/linaro/best_model_4.pth', cuda=False)


while True:
    clientsocket, addr = serversocket.accept()
    print("addr: %s" % str(addr))

    try:
        while True:
            data = clientsocket.recv(1024)
            redata = ast.literal_eval(data.decode('utf-8'))
            if not data:
                break
            print("receive data: %s" % data.decode('utf-8'))

            if len(redata) == 5:
                # action = agent.get_action(redata)
                action = agent.act_last(redata)
                clientsocket.send(str(action).encode())
            else:
                feedback_data = redata
                reward = feedback_data[0]
                new_state = feedback_data[1]
                state = feedback_data[2]
                print(f'reward:{reward},new_state:{new_state},state:{state}')
                agent.feedback_last(reward, new_state, state)
                clientsocket.send(str("1").encode())

    except ConnectionResetError:
        print("addr update")
    finally:
        clientsocket.close()
