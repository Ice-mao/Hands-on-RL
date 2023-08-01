import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rl_utils


class PolicyNet(torch.nn.Module):
    """
    定义成策略网络，输入一个state，输出返回各个动作的概率分布
    网络结构为两个全连接层，中间用relu进行非线性化，softmax输出分类概率
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)

        # def init_xavier(m):
        #     # if type(m) == nn.Linear or type(m) == nn.Conv2d:
        #     if type(m) == torch.nn.Linear:
        #         torch.nn.init.xavier_normal_(m.weight)
        #
        # self.policy_net.apply(init_xavier)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        """根据输入状态得到的动作分布概率进行采样"""
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.policy_net(state)  # 得到动作概率
        action_dist = torch.distributions.Categorical(probs)  # 创建分布概率
        action = action_dist.sample()  # 采样
        return action.item()

    def update(self, trajectory_dict):
        # 根据采样得到的trajectory
        reward_list = trajectory_dict['rewards']
        state_list = trajectory_dict['states']
        action_list = trajectory_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor(np.array([state_list[i]]), dtype=torch.float).to(self.device)
            action = torch.tensor(np.array([action_list[i]])).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))  # 得到log(pai_theta(s_t,a_t))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 得到loss function
            loss.backward()
        self.optimizer.step()


learning_rate = 0.001
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda")

env = gym.make('CartPole-v1')  # 导入倒立摆环境
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]  # Cart Position, Pole Cart Velocity, Angular Pole, Angle Velocity
action_dim = env.action_space.n  # 0:left, 1:right
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes/10)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            [state, _] = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                [next_state, reward, done_1, done_2, _] = env.step(action)
                if done_1 == True or done_2 == True:
                    done = True
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format('CartPole-v1'))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format('CartPole-v1'))
plt.show()
