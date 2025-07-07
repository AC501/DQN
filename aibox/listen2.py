import socket
import torch
import torch.nn as nn
import os
import ast
import random
import math

# DQN 神经网络定义
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

# 强化学习智能体定义
class Agent:
    def __init__(self, model_path=None, cuda=False, random_threshold=0.5, eps_start=1.0, eps_end=0.1, eps_decay=0.001, target_update = 5):
        self.policy_net = DQN()
        self.target_net = DQN()
        if model_path and os.path.exists(model_path):
            self.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.cuda = cuda
        if cuda:
            self.policy_net = self.policy_net.cuda()
            self.target_net = self.target_net.cuda()
        self.random_threshold = random_threshold
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.target_update = target_update  # 每5步更新网络
        self.learn_steps = 0  # 记录学习步数

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
            print('q_value', q_value)

        _, action_index = torch.max(q_value, dim=0)
        action = action_index.cpu().item()
        return action

    def get_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * self.steps_done / self.eps_decay)
        self.steps_done += 1


        if random.random() <= eps_threshold:
            act = self.get_action_randomly()
        else:
            act = self.get_optim_action(state)

        # 每 10 步更新一次 target 网络
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.update_target_network()

        return act

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

            # 加载模型并使用 Agent 类获取动作
            agent = Agent(model_path='/home/linaro/best_model.pth', cuda=False)

            action = agent.get_action(redata)
            print("action:", action)

            clientsocket.send(str(action).encode())

    except ConnectionResetError:
        print("addr update")
    finally:
        clientsocket.close()
