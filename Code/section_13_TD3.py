import random
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils


class PolicyNet(torch.nn.Module):
    """
    策略网络输出使用tanh激活函数限定取值范围
    :param action_bound:放缩输出范围
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    """拼接state和action，表达state_action value"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class TD3:
    """
    实现TD3算法(Twin Delayer DDPG)
    :param tau：更新target network的参数
    :param gamma：更新折旧率
    :param sigma：采样动作时添加的高斯噪声的标准差（均值始终为0）
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, target_noise, actor_lr, critic_lr,
                 tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标策略网络和动作状态价值网络和并设置相同的参数
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        # 设置优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.sigma = sigma  # 采样时的高斯噪声的标准差,均值直接设为0
        self.target_noise = target_noise  # 平滑目标策略动作时的高斯噪声的标准差
        self.noise_clip = 0.5
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.action_bound = action_bound  # 动作空间范围
        self.device = device
        # 用于延迟迭代策略网络计算
        self.policy_delay = 2
        self.count = 0

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        # add noise
        noise = 0 + self.sigma * np.random.randn(self.action_dim)
        action = action + noise
        return [action.item()]

    def soft_update(self, net, target_net):
        """
        更新target网络中的参数
        :param net:原网络
        :param target_net:target网络
        :return:None
        """
        for param, param_target in zip(net.parameters(), target_net.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1.0 - self.tau))

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),
                             dtype=torch.float).view(-1, 1).to(self.device)
        # Compute target actions
        target_actions = torch.clamp((self.actor(next_states).clone().to(self.device) +
                                      torch.clip(torch.tensor(self.target_noise * np.random.randn(self.action_dim)).to(
                                          self.device),
                                                 torch.tensor(-self.noise_clip).to(self.device),
                                                 torch.tensor(self.noise_clip).to(self.device)).to(self.device)),
                                     torch.tensor(-self.action_bound).to(self.device),
                                     torch.tensor(self.action_bound).to(self.device))
        # Compute targets
        next_q_values_1 = self.target_critic1(next_states, self.target_actor(next_states))
        next_q_values_2 = self.target_critic2(next_states, self.target_actor(next_states))
        next_q_values = torch.min(torch.cat((next_q_values_1, next_q_values_2), dim=1), 1, keepdim=True)[0]
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        # 更新状态动作价值(critic)网络
        critic_loss1 = torch.mean(F.mse_loss(q_targets.detach(), self.critic1(states, actions)))
        critic_loss2 = torch.mean(F.mse_loss(q_targets.detach(), self.critic2(states, actions)))
        self.critic1_optimizer.zero_grad()
        critic_loss1.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2_optimizer.step()
        # 更新策略(actor)网络
        self.count += 1
        if self.count == self.policy_delay:
            self.count = 0  # 清零计数器
            actor_loss = -torch.mean(self.critic1(states, self.actor(states)))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # 软更新策略网络和状态动作价值网络
            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic1, self.target_critic1)
            self.soft_update(self.critic2, self.target_critic2)


actor_lr = 3e-4
critic_lr = 3e-3
num_episodes = 200
hidden_dim = 64
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 10000
minimal_size = 1000
batch_size = 64
sigma = 0.1  # 高斯噪声标准差
target_noise = 0.2  # Stddev for smoothing noise added to target policy.
device = torch.device("cuda")

env_name = 'Pendulum-v1'
env = gym.make(env_name, render_mode="human")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
agent = TD3(state_dim, hidden_dim, action_dim, action_bound, sigma, target_noise, actor_lr, critic_lr, tau, gamma,
            device)

return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()
