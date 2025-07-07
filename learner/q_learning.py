'''
Created on 04/08/2014

@author: Gabriel de O. Ramos <goramos@inf.ufrgs.br>
'''
from turtle import done

from torch import nn
from learner import Learner
import torch.optim as optim
from replay import ReplayBuffer
import torch
from DQN import DQN
import random
import numpy as np
import math
import os
from datetime import datetime
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QLearner(Learner):

    def __init__(self, name, env, starting_state, goal_state, alpha, gamma, model, replay,
                 target_update, eps_start, eps_end, eps_decay, input_dim, output_dim, batch_size, network_file):

        super(QLearner, self).__init__(name, env, self)
        self.output_dim = output_dim
        self.train = None
        self._starting_state = starting_state
        self._goal_state = goal_state
        self.cuda = True

        self.steps_done = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self._alpha = alpha
        self._gamma = gamma
        self.model = model
        self.replay = replay
        self.target_update = target_update
        self.gamma = gamma

        self.n_actions = output_dim
        self.batch_size = batch_size
        self.random_threshold = 0.6

        self.network_file = network_file
        self.policy_net = DQN(input_dim, output_dim).to(device)
        self.target_net = DQN(input_dim, output_dim).to(device)


        self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.001)



        if network_file:
            self.policy_net.load_state_dict(torch.load(network_file, map_location=torch.device(device)))
            self.policy_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.learn_steps = 0
        self.best_reward = float('-inf')
        self.best_loss = float('inf')
        self.best_model_path = "weights_0/best_model.pth"
        self.best_weights_path = f'weights_0/best_weights_{name}.pth'

        # Add a variable to store the last action
        self.last_action = None

        # === 手动向经验池插入极端高/低奖励的样本，提升奖励多样性 ===
        # 只在经验池初始化时插入，避免影响后续真实采样分布
        # 这里假设state和next_state为全0或全1，action为0或1，reward为极大/极小值
        for _ in range(10):  # 每种极端样本各插入10条
            # 极端高奖励样本
            state_high = np.zeros((input_dim,), dtype=np.float32)
            next_state_high = np.ones((input_dim,), dtype=np.float32)
            action_high = 0
            reward_high = 10.0  # 极端高奖励
            td_error_high = reward_high
            self.replay.add(state_high, next_state_high, action_high, reward_high, td_error_high)

            # 极端低奖励样本
            state_low = np.ones((input_dim,), dtype=np.float32) * 5
            next_state_low = np.zeros((input_dim,), dtype=np.float32)
            action_low = 1
            reward_low = -10.0  # 极端低奖励
            td_error_low = reward_low
            self.replay.add(state_low, next_state_low, action_low, reward_low, td_error_low)
        # === 结束极端样本插入 ===

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

        if self.model == 'train' and random.random() <= eps_threshold:
            with torch.no_grad():
                act = self.get_action_randomly()
        else:
            with torch.no_grad():
                act = self.get_optim_action(state)
        return act

    def save_best_model(self):
        torch.save(self.model.state_dict(), self.best_model_path)
        print(f"Saved best model parameters to {self.best_model_path}")

    def load_best_model(self):
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path))
            print(f"Loaded best model parameters from {self.best_model_path}")
        else:
            print(f"No best model found at {self.best_model_path}")

    def update_best_reward(self, current_reward):
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.save_best_model()

    def save_model(self, filename='best_model.pth'):
        torch.save(self.policy_net.state_dict(), filename)

    def set_train(self):
        self.train = True
        self.model.train()

    def set_eval(self):
        self.train = False
        self.model.eval()

    def learn(self, experiences):
        if self.model == 'train':
            loss_fn = nn.MSELoss()
            if self.replay.steps_done <= 10:
                print(f"Exp Pool number:{self.replay.steps_done}")
                return
            print("DQN learn called")
            batch,_,_ = self.replay.sample(self.batch_size)


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

            loss = loss_fn(expected_state_action_values,state_action_values)

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
                print(f"[✔] learn_step={self.learn_steps}, loss={loss.item():.4f} → 保存模型")
                self.best_loss = loss.item()
                torch.save(self.policy_net.state_dict(), self.best_model_path)

            # === 1. Q值变化写入csv ===
            # 记录每次训练的Q值均值和方差，便于后续分析DQN学习情况
            q_csv_path = "data/dqn_learn/q_values.csv"
            os.makedirs(os.path.dirname(q_csv_path), exist_ok=True)
            if not os.path.exists(q_csv_path):
                with open(q_csv_path, "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "state_action_mean", "state_action_std", "expected_q_mean", "expected_q_std"])
            q_step = getattr(self, 'q_step', 0)
            with open(q_csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    q_step,
                    state_action_values.mean().item(),
                    state_action_values.std().item(),
                    expected_state_action_values.mean().item(),
                    expected_state_action_values.std().item()
                ])
            self.q_step = q_step + 1

            # === 2. 动作分布写入csv ===
            # 统计当前batch动作分布，便于分析策略是否多样
            action_csv_path = "data/dqn_learn/action_distribution.csv"
            os.makedirs(os.path.dirname(action_csv_path), exist_ok=True)
            if not os.path.exists(action_csv_path):
                with open(action_csv_path, "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "action_0_count", "action_1_count"])
            actions = action_batch.cpu().numpy().flatten()
            action_0 = np.sum(actions == 0)
            action_1 = np.sum(actions == 1)
            action_step = getattr(self, 'action_step', 0)
            with open(action_csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([action_step, action_0, action_1])
            self.action_step = action_step + 1

            # === 3. 经验回放池多样性写入csv ===
            # 定期采样经验池，统计状态和奖励的均值与方差，分析经验多样性
            replay_csv_path = "data/dqn_learn/replay_diversity.csv"
            os.makedirs(os.path.dirname(replay_csv_path), exist_ok=True)
            if not os.path.exists(replay_csv_path):
                with open(replay_csv_path, "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "state_mean", "state_std", "reward_mean", "reward_std"])
            # 每100步记录一次
            replay_step = getattr(self, 'replay_step', 0)
            if replay_step % 100 == 0 and hasattr(self.replay, 'sample'):
                try:
                    batch_sample, _, _ = self.replay.sample(100)
                    state_batch_sample = torch.cat(batch_sample.state).cpu().numpy()
                    reward_batch_sample = torch.cat(batch_sample.reward).cpu().numpy()
                    with open(replay_csv_path, "a", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            replay_step,
                            np.mean(state_batch_sample),
                            np.std(state_batch_sample),
                            np.mean(reward_batch_sample),
                            np.std(reward_batch_sample)
                        ])
                except Exception as e:
                    pass  # 防止经验池样本不足时报错
            self.replay_step = replay_step + 1

    def act_last(self, state, tlID):
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
        if current_q_values.dim() == 1:  # 1D case
            current_q = current_q_values[self.last_action]
        else:  # 2D case
            current_q = current_q_values.gather(1, torch.tensor([[self.last_action]], dtype=torch.int64).to(
                device)).squeeze(1)

        # Get Q-values for next state from target network
        next_q_values = self.target_net(next_state)
        next_max_q = torch.max(next_q_values).item()

        # Calculate TD error
        td_error = reward + self.gamma * next_max_q - current_q.item()

        # Add experience to replay buffer
        self.replay.add(state, next_state, self.last_action, reward, td_error)

        # Sample from replay buffer if it's large enough
        if len(self.replay) > self.batch_size:
            experiences = self.replay.sample(self.batch_size)
            self.learn(experiences)




