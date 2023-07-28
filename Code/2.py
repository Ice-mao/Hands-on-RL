import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    """create a BernoulliBandit, K是拉杆的个数"""

    def __init__(self, K):
        self.probs = np.random.uniform(size=K)  # 表示拉动每个连杆获奖的概率
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K  # 生成类内部的参数

    def step(self, k):
        """k是玩家选定拉杆的序号，返回是否获奖"""
        if np.random.rand() < self.probs[k]:  # 获奖
            return 1
        else:
            return 0


class Solver:
    """ 多臂老虎机算法基本框架 """

    def __init__(self, bandit):
        """bandit:老虎机模型"""
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数
        self.regret = 0  # 当前步的累积懊悔
        self.action = []
        self.regrets = []

    def update_regret(self, k):
        """选择拉杆k，更新累积regret"""
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        """返回由不同策略实现得到的aciton，此处返回拉杆编号"""
        raise NotImplementedError

    def run(self, num_step):
        """运行一定step"""
        for i in range(num_step):
            k = self.run_one_step()
            self.counts[k] += 1 #更新拉杆次数记录
            self.action.append(k) #更新动作记录
            self.update_regret(k) #更新regret

class EpsilonGreedy(Solver):
    """贪婪算法，继承Solver类"""
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon #初始化噪声系数
        self.estimates = np.array([init_prob] * self.bandit.K) #初始化所有拉杆的期望奖励

    def run_one_step(self):
        if np.random.random() < self.epsilon:#噪声随机
            k = np.random.randint(0, self.bandit.K) #随机选择一根拉杆
        else:#贪心算法
            k = np.argmax(self.estimates) # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k) # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k]) #更新对应编号拉杆的期望估计
        return k

class DecayingEpsilonGreedy(Solver):
    """随机比例随时间衰减的贪婪算法，继承Solver类"""
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)  # 初始化所有拉杆的期望奖励

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 /self.total_count: #epsilon随时间衰减
            k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆
        else:  # 贪心算法
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆

        r = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])  # 更新对应编号拉杆的期望估计
        return k

class UCB(Solver):
    """UCB算法，继承Solver类"""
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB,self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)  # 初始化所有拉杆的期望奖励
        self.coef = coef #常数c控制不确定性的比重

    def run_one_step(self):
        self.total_count += 1
        Uta = np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))#U_t(a)公式
        ucb = self.estimates + self.coef * Uta # 计算上置信界
        k = np.argmax(ucb) # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class ThompsonSampling(Solver):
    """汤普森采样算法"""
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.zeros(self.bandit.K)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.zeros(self.bandit.K)  # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a + 1, self._b + 1)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k

def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(114514)  # 设定随机种子,使实验具有可重复性
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print("随机生成了一个%d臂伯努利老虎机" % K)
    print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
          (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

    # epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    # epsilon_greedy_solver.run(5000)
    # print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
    # plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

    # decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
    # decaying_epsilon_greedy_solver.run(5000)
    # print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
    # plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

    # coef = 0.15
    # UCB_solver = UCB(bandit_10_arm, coef)
    # UCB_solver.run(5000)
    # print('上置信界算法的累积懊悔为：', UCB_solver.regret)
    # plot_results([UCB_solver], ["UCB"])

    thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompson_sampling_solver.run(5000)
    print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])


