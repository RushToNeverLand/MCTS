from game2048 import *
from collections import defaultdict
import numpy as np
import json


class MCTS:
    def __init__(self, c=100, max_depth=10, gamma=0.9, loop_num=30):
        self.env = None

        self.Q_table = defaultdict()
        self.visit_times = defaultdict()

        self.c = c
        self.max_depth = max_depth
        self.gamma = gamma
        self.loop_num = loop_num

    def reset(self):
        self.Q_table = defaultdict()
        self.visit_times = defaultdict()

    def hash_state(self, obs):
        key = ''
        for row in obs:
            for it in row:
                key += str(it) + '_'
        return key
        
    def rollout(self, test_env, state, depth):
        if depth == 0:
            return 0
        action = random.randint(0, 3)
        new_state, reward, done, info = test_env.step(action)
        return reward + self.gamma * self.rollout(test_env, new_state, depth - 1)

    def select_action(self, env, obs):
        for i in range(self.loop_num):
            self.env = copy.deepcopy(env)
            self.simulate(obs, self.max_depth)
        state = self.hash_state(obs)
        return np.argmax(self.Q_table[state])

    def simulate(self, state, depth):
        if depth == 0:
            return 0

        if isinstance(state, list):
            state = self.hash_state(state)

        # expansion
        if state not in self.Q_table.keys():
            self.Q_table[state] = [0 for i in range(4)]
            self.visit_times[state] = [1 for i in range(4)]

            test_env = copy.deepcopy(self.env)
            return self.rollout(test_env, state, depth)
        
        # selection
        q_value = self.Q_table[state]
        n_table = self.visit_times[state]
        estimated = q_value + self.c * np.sqrt(np.log2(sum(n_table)) / n_table)
        best_action = np.argmax(estimated)
        new_state, reward, done, info = self.env.step(best_action)

        # back propagate
        q = reward + self.gamma * self.simulate(new_state, depth-1)
        self.visit_times[state][best_action] += 1
        self.Q_table[state][best_action] += (q - self.Q_table[state][best_action]) / self.visit_times[state][best_action]

        return q


if __name__ == '__main__':
    env = Game2048Env(True)
    # you can fix the seed for debugging, but your agent SHOULD NOT overfit to the env of a certain seed
    # env.seed(0)
    env.setRender(False)
    # render is automatically set to False for copied envs
    # remember to call reset() before calling step()
    obs = env.reset()
    done = False
    mcts = MCTS()

    experiment_data = []
    episodes = 1
    for i in range(episodes):

        while not done:
            action = mcts.select_action(env, obs)
            obs, rew, done, info = env.step(action)
            print('episode', i, rew, done, info)

        experiment_data.append(env.info)
        obs = env.reset()
        done = False
        # mcts.reset()
    # remember to close the env, but you can always let resources leak on your own computer :|
    env.close()

    with open('data.json', 'w') as fout:
        json.dump(experiment_data , fout)
