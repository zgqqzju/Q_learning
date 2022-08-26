"""
Author：zhuge qinqin
Date：2022/08/26
Purpose：仿写一个走房间的Q-learning
"""

import networkx as nx
import numpy as np
import pandas as pd
import time

# 设置随机种子
np.random.seed(2)

"""  状态：共6个状态（房间）
    动作：通向哪个房间
"""


class Main_algorithm:
    def __init__(self):
        self.n_states = 6  # 共6个房间
        self.actions = list(range(self.n_states))  # 可以通向哪个房间
        self.epsilon = 0.9  # epsilon-greedy parameter：90%的情况下选择最优的动作，10%的情况下选择随机的动作
        self.discount = 0.8  # 折扣
        self.learning_rate = 1  # 学习率
        self.max_episodes = 100

    @staticmethod
    def define_environment():
        """定义一个环境：确定哪些房间相连，以及奖赏"""
        G = nx.DiGraph()  # 创建有向图
        G.add_edge(0, 4, weight=0)
        G.add_edge(1, 3, weight=0)
        G.add_edge(1, 5, weight=100)
        G.add_edge(2, 3, weight=0)
        G.add_edge(3, 1, weight=0)
        G.add_edge(3, 2, weight=0)
        G.add_edge(3, 4, weight=0)
        G.add_edge(4, 0, weight=0)
        G.add_edge(4, 3, weight=0)
        G.add_edge(4, 5, weight=100)
        G.add_edge(5, 1, weight=0)
        G.add_edge(5, 4, weight=0)
        G.add_edge(5, 5, weight=100)

        return G

    def build_Q_table(self):
        """
        创建一个初始的Q表（DataFrame类型）
        :return:
        """
        table = pd.DataFrame(
            np.zeros((self.n_states, len(self.actions))),  # 初始化
            columns=self.actions,
        )
        return table

    def choose_action(self, state, q_table, environment):
        """ 选动作（根据当前的状态和Q表）
                +++++++ 厄普西隆-greedy 策略 +++++++
        """
        optional_actions = [k for k in environment[state]]  # 可选的动作
        if np.random.rand() < self.epsilon:  # 选最好的
            state_actions = q_table.loc[state, :][optional_actions]  # 挑出来可选的 状态-动作
            # 打乱索引顺序，防止两者恰好相等时选不到第二个的情况
            state_actions = state_actions.reindex(np.random.permutation(state_actions.index))
            action = state_actions.idxmax()  # 选Q值最大的动作
        else:  # 选随机动作
            action = np.random.choice(optional_actions)
        return action

    @staticmethod
    def get_env_feedback(S, A, environment):
        """ 在某个状态S采取行动A后进入的下一个状态S_和获得的奖励R"""
        # 下一个状态就是房间号（即动作A的编号），奖赏就是两个房间之间连边的权值
        return A, environment.get_edge_data(S, A)['weight']

    def use_q_table(self, S, q_table, environment):
        """
        完成训练后，用Q表选动作
        :param environment:
        :param S:
        :param q_table:
        :return:
        """
        actions = [S]  # 表示走过的房间
        is_terminated = False
        while not is_terminated:
            state_actions = q_table.iloc[S, :]  # 子表：该状态下 各动作 对应的q值
            A = state_actions.idxmax()
            S_, R = self.get_env_feedback(S, A, environment)  # 进入下一个状态、获得状态转移带来的环境奖励
            actions.append(S_)
            if S_ == 5:
                is_terminated = True
            S = S_
        return actions


if __name__ == '__main__':
    print('学习阶段')
    """强化学习主循环
    """
    test = Main_algorithm()
    G = test.define_environment()
    Q_table = test.build_Q_table()
    for episode in range(test.max_episodes):
        print('%d episodes:' % episode)
        step_counter = 0
        S = np.random.choice(list(range(test.n_states)))
        is_terminated = False
        while not is_terminated:
            A = test.choose_action(S, Q_table, G)
            S_, R = test.get_env_feedback(S, A, G)  # 进入下一个状态、获得状态转移带来的环境奖励
            q_current = Q_table.iloc[S, A]  # Q(s,a)的当前值
            q_target = R + test.discount * Q_table.iloc[S_, :].max()  # S1到S2获得的奖赏+S2状态下采取某动作可获得的最大Q
            if S_ == 5:  # 目标房间的编号为5
                is_terminated = True
            Q_table.iloc[S, A] += test.learning_rate * (q_target - q_current)  # 更新：旧+学习率*差异
            S = S_  # 进入到下一个状态
            step_counter += 1
    """应用阶段"""
    print('最短路径：')
    for s in list(range(test.n_states)):
        actions = test.use_q_table(s, Q_table, G)
        print(actions)
