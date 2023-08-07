import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
import rl_utils
import copy

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    """
    PPO算法，采用PPO截断算法

    :param gamma: 折旧因子
    :param lmbda：估计优势函数的lamma因子
    :param epochs: 一条序列的数据用来训练轮数
    :param eps: PPO中截断范围的参数
    :param type: discrete:离散环境 continous:连续环境

    """
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, type="discrete"):
        if type == "continuous":
            self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        else:
            self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.type = type

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        if self.type == "continuous":
            mu, std = self.actor(state)
            action_dist = torch.distributions.Normal(mu, std)
            action = action_dist.sample()
            return [action.item()]
        else:
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return action.item()

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
        if self.type == "continuous":
            rewards = (rewards + 8.0) / 8.0  #对奖励进行修改方便训练
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - self.critic(states)
            advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                                   td_delta.cpu()).to(self.device)
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu.detach(), std.detach())
            old_log_prob = action_dists.log_prob(actions)

            for _ in range(self.epochs):
                mu, std = self.actor(states)
                action_dists = torch.distributions.Normal(mu, std)
                log_prob = action_dists.log_prob(actions)
                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 公式中的截断项

                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        else:
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - self.critic(states)
            advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                                   td_delta.cpu()).to(self.device)
            old_log_prob = torch.log(self.actor(states).gather(1, actions)).detach()

            for _ in range(self.epochs):
                log_prob = torch.log(self.actor(states).gather(1, actions))
                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage #公式中的截断项

                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

# actor_lr = 1e-3
# critic_lr = 1e-2
# num_episodes = 500
# hidden_dim = 128
# gamma = 0.98
# lmbda = 0.95
# epochs = 10
# eps = 0.2
# device = torch.device("cuda")

# env_name = 'CartPole-v1'
# env = gym.make(env_name, render_mode="human")
# torch.manual_seed(0)
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
#             epochs, eps, gamma, device)

actor_lr = 1e-4
critic_lr = 5e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2
device = torch.device("cuda")

env_name = 'Pendulum-v1'
env = gym.make(env_name, render_mode="human")
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device, type="continuous")

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()