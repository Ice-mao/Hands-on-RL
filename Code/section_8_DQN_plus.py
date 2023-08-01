import random
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils


class DQN:
    """DQN算法"""

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device, dqn_type='VanillaDQN'):
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)  # 生成Q网络
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)  # 生成target_Q网络
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.dqn_type = dqn_type

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        return self.net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions'])).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.net(states).gather(1, actions)  # Q值

        # net的输出是两个动作的价值(Q(s,a))，max(1)是在第一维度上求最大值，也就是每个状态的最大动作价值
        # 其中max(1)[0]是价值，max(1)[1]是对应的动作(0:left;1:right)
        if self.dqn_type == 'DoubleDQN':
            # DDQN计算方法：用net选择action，用target_net来评估action value
            max_action = self.net(next_states).max(1)[1].view(-1, 1)
            max_next_q_value = self.target_net(next_states).gather(1, max_action)
        else:  # DQN的情况
            # DQN计算方法：用target_net来评估最大的action value
            max_next_q_value = self.target_net(next_states).max(1)[0].view(-1, 1)

        q_targets = rewards + self.gamma * max_next_q_value * (1 - dones)  # TD error

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_net.load_state_dict(
                self.net.state_dict())  # 更新目标网络
        self.count += 1


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size,
              batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                [state, _] = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    action_continuous = dis_to_con(action, env,
                                                   agent.action_dim)
                    [next_state, reward, done_1, done_2, _] = env.step([action_continuous])
                    if done_1 == True or done_2 == True:
                        done = True
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list


def dis_to_con(discrete_action, env, action_dim):
    """离散动作转回连续的函数"""
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值 -2.0
    action_upbound = env.action_space.high[0]  # 连续动作的最大值 2.0
    return action_lowbound + (discrete_action /
                              (action_dim - 1)) * (action_upbound -
                                                   action_lowbound)


lr = 1e-2
num_episodes = 200
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 50
buffer_size = 5000
minimal_size = 1000
batch_size = 64
device = torch.device("cuda")

env_name = 'Pendulum-v1'
env = gym.make(env_name, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = 11  # 将连续动作分成11个离散动作

# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# replay_buffer = rl_utils.ReplayBuffer(buffer_size)
# agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
#             target_update, device)
# return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
#                                           replay_buffer, minimal_size,
#                                           batch_size)
#
# episodes_list = list(range(len(return_list)))
# mv_return = rl_utils.moving_average(return_list, 5)
# plt.plot(episodes_list, mv_return)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('DQN on {}'.format(env_name))
# plt.show()
#
# frames_list = list(range(len(max_q_value_list)))
# plt.plot(frames_list, max_q_value_list)
# plt.axhline(0, c='orange', ls='--')
# plt.axhline(10, c='red', ls='--')
# plt.xlabel('Frames')
# plt.ylabel('Q value')
# plt.title('DQN on {}'.format(env_name))
# plt.show()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device, dqn_type='DoubleDQN')
return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
                                          replay_buffer, minimal_size,
                                          batch_size)

episodes_list = list(range(len(return_list)))
mv_return = rl_utils.moving_average(return_list, 5)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Double DQN on {}'.format(env_name))
plt.show()

frames_list = list(range(len(max_q_value_list)))
plt.plot(frames_list, max_q_value_list)
plt.axhline(0, c='orange', ls='--')
plt.axhline(10, c='red', ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.title('Double DQN on {}'.format(env_name))
plt.show()

