import gym
import numpy as np
from CliffWalking import CliffWalkingEnv
from collections import defaultdict


class QLearning:
    def __init__(self, learning_rate, state_dim, action_dim):
        self.learning_rate = learning_rate
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.Q_table = np.zeros((state_dim, action_dim))
        self.epsilon = 0.1
        self.gamma = 0.9

    def sample(self, state):
        probs = np.ones(self.action_dim) * self.epsilon / self.action_dim
        pos = np.argmax(self.Q_table[state])
        probs[pos] += 1 - self.epsilon
        return np.random.choice(self.action_dim, p=probs)

    def learn(self, state, action, reward, next_state, done):
        best_action = np.argmax(self.Q_table[next_state])
        error = reward + self.gamma * self.Q_table[next_state][best_action] - self.Q_table[state][action]
        self.Q_table[state][action] += self.learning_rate * error

    def play(self, env, episodes=5000):
        for i in range(episodes):
            steps = 0
            total_rewards = 0
            state, done = env.reset(), False

            while not done:
                steps += 1
                action = self.sample(state)
                next_state, reward, done, _ = env.step(action)
                total_rewards += reward
                self.learn(state, action, reward, next_state, done)
                state = next_state
            print('after iteration {} and the steps and rewards are {} and {}'.format(i, steps, total_rewards))


class SARSA:
    def __init__(self, n, learning_rate, state_dim, action_dim):
        self.n = n
        self.learning_rate = learning_rate
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.Q_table = np.zeros((state_dim, action_dim))
        self.epsilon = 0.1
        self.gamma = 0.9

    def sample(self, state):
        probs = np.ones(self.action_dim) * self.epsilon / self.action_dim
        pos = np.argmax(self.Q_table[state])
        probs[pos] += 1 - self.epsilon
        return np.random.choice(self.action_dim, p=probs)

    def play(self, env, episodes=5000):
        for i in range(episodes):
            steps = 0
            total_rewards = 0
            state, done = env.reset(), False
            action = self.sample(state)

            states, actions, rewards = [state], [action], [0]
            t, T = 0, np.inf

            while True:
                if t < T:
                    next_state, reward, done, _ = env.step(action)
                    
                    steps += 1
                    total_rewards += reward

                    states.append(next_state)
                    rewards.append(reward)

                    if done:
                        T = t + 1
                    else:
                        action = self.sample(next_state)
                        actions.append(action)
                tau = t - self.n + 1
                
                if tau >= 0:
                    G = 0
                    for j in range(tau + 1, min(tau + self.n + 1, T + 1)):
                        G += np.power(self.gamma, j - tau - 1) * rewards[j]
                    if tau + self.n < T:
                        state_action = (states[tau + self.n], actions[tau + self.n])
                        G += np.power(self.gamma, self.n) * self.Q_table[state_action[0]][state_action[1]]
                    # update Q values
                    state_action = (states[tau], actions[tau])
                    self.Q_table[state_action[0]][state_action[1]] += self.learning_rate * (
                                G - self.Q_table[state_action[0]][state_action[1]])
                
                if tau == T - 1:
                    break
                t += 1

            print('after iteration {} and the steps and rewards are {} and {}'.format(i, steps, total_rewards))


class SARSALambda:
    def __init__(self, learning_rate, state_dim, action_dim, lamda=0.1):
        self.learning_rate = learning_rate
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.Q_table = np.zeros((state_dim, action_dim))
        self.epsilon = 0.1
        self.gamma = 0.9
        self.lamda = lamda
        self.eligibility_trace = self.Q_table.copy()

    def sample(self, state):
        probs = np.ones(self.action_dim) * self.epsilon / self.action_dim
        pos = np.argmax(self.Q_table[state])
        probs[pos] += 1 - self.epsilon
        return np.random.choice(self.action_dim, p=probs)

    def learn(self, state, action, reward, next_state, done):
        best_action = np.argmax(self.Q_table[next_state])
        error = reward + self.gamma * self.Q_table[next_state][best_action] - self.Q_table[state][action]
        self.eligibility_trace[state][action] += 1
        self.Q_table = self.Q_table + self.learning_rate * error * self.eligibility_trace
        self.eligibility_trace = self.gamma * self.lamda * self.eligibility_trace
        return best_action

    def play(self, env, episodes=5000):
        for i in range(episodes):
            steps = 0
            total_rewards = 0
            state, done = env.reset(), False
            action = self.sample(state)

            while not done:
                steps += 1
                next_state, reward, done, _ = env.step(action)
                total_rewards += reward
                best_action = self.learn(state, action, reward, next_state, done)
                state = next_state
                action = best_action

            print('after iteration {} and the steps and rewards are {} and {}'.format(i, steps, total_rewards))


if __name__ == '__main__':
    env = CliffWalkingEnv()

    n = [1, 3, 5]
    learning_rate = [0, 0.5, 1.]
    state_dim, action_dim = env.observation_space.n, env.action_space.n

    # q_agent = QLearning(0.1, state_dim, action_dim)
    # q_agent.play(env, episodes=2000)

    # T = [1, 3, 5]
    sarsa_agent = SARSA(1, 0.1, state_dim, action_dim)
    sarsa_agent.play(env, episodes=2000)

    # lamda = [0, 0.5, 1]
    # sarsa_lamda = SARSALambda(0.1, state_dim, action_dim, lamda=0.5)
    # sarsa_lamda.play(env)
