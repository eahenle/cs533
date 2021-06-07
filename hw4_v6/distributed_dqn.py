#!/usr/bin/env python

import gym
import torch
import time
import os
import ray
import numpy as np
from tqdm import tqdm
from random import uniform, randint
import io
import base64
import matplotlib.pyplot as plt
import sys

from custom_cartpole import CartPoleEnv
from memory_remote import ReplayBuffer_remote
from dqn_model import _DQNModel
from dqn_model import DQNModel

FloatTensor = torch.FloatTensor

# Environment
ENV_NAME = 'CartPole_distributed'
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}
result_folder = ENV_NAME + "_distributed"
result_file = ENV_NAME + "/results.txt"


# Plot results.
def plot_result(total_rewards, learning_num, legend, nb_agents, nb_evaluators):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)
        
    plt.figure(num = 1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.savefig("{}agents_{}evaluators".format(nb_agents, nb_evaluators))


@ray.remote
class DQN_agent_remote(object):
    def __init__(self, env, hyper_params, eval_model, action_space):
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']
        self.episode = 0
        self.steps = 0
        self.best_reward = 0
        self.learning = True
        self.action_space = action_space
        self.eval_model = eval_model
        self.update_steps = hyper_params['update_steps']
    
    def explore_or_exploit_policy(self, state):
        p = uniform(0, 1)
        # Get decreased epsilon
        epsilon = self.linear_decrease(self.initial_epsilon, 
                               self.final_epsilon,
                               self.steps,
                               self.epsilon_decay_steps)
        if p < epsilon:
            #return action
            return randint(0, self.action_space - 1)
        else:
            #return action
            return self.greedy_policy(state)
        
    def greedy_policy(self, state):
        return self.eval_model.predict(state)

    def learn(self, test_interval):
        for _ in tqdm(range(test_interval), desc="Training"):
            state = self.env.reset()
            done = False
            steps = 0
            while steps < self.max_episode_steps and not done: # INSERTED MY CODE HERE
                # add experience from explore-exploit policy to memory
                action = self.explore_or_exploit_policy(state)
                state_next, reward, done, _ = self.env.step(action)
                self.memory.add(state, action, reward, state_next, done)
                state = state_next
                steps += 1
                


@ray.remote
class EvalWorker():
    def __init__(self, eval_model, env, max_episode_steps):
        self.eval_model = eval_model
        self.env = env
        self.max_episode_steps = max_episode_steps
    
    def greedy_policy(self, state):
        return self.eval_model.predict(state)

    def evaluate(self, trials = 30):
        total_reward = 0
        for _ in tqdm(range(trials), desc="Evaluating"):
            state = self.env.reset()
            done = False
            steps = 0
            while steps < self.max_episode_steps and not done:
                steps += 1
                action = self.greedy_policy(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
        avg_reward = total_reward / trials
        print(avg_reward)
        return avg_reward


class ModelServer():
    def __init__(self, hyper_params, memory_server, nb_agents, nb_evaluators, action_space=len(ACTION_DICT)):

        self.hyper_params = hyper_params
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']
        self.action_space = action_space
        self.batch_size = hyper_params['batch_size']
        self.memory_server = memory_server
        self.nb_agents = nb_agents
        self.nb_evaluators = nb_evaluators
        env = CartPoleEnv()
        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate = hyper_params['learning_rate'])
        self.target_model = DQNModel(input_len, output_len)

        self.agents = [DQN_agent_remote.remote(CartPoleEnv(), hyper_params, self.eval_model, action_space) for i in range(nb_agents)]
        self.evaluators = [EvalWorker.remote(self.eval_model, CartPoleEnv(), hyper_params['max_episode_steps']) for i in range(nb_evaluators)]

    # Linear decrease function for epsilon
    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def update_batch(self):
        if len(self.memory) < self.batch_size or self.steps % self.update_steps != 0:
            return
        batch = self.memory.sample(self.batch_size)
        (states, actions, reward, next_states, is_terminal) = batch
        states = states
        next_states = next_states
        nonterminal_x_beta = FloatTensor([0 if t else self.beta for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size, dtype=torch.long)
        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]        
        # Calculate target
        if self.use_target_model:
            actions, q_next = self.target_model.predict_batch(next_states)
        else:
            actions, q_next = self.eval_model.predict_batch(next_states)
        q_targets = reward + nonterminal_x_beta * torch.max(q_next, 1).values
        # update model
        self.eval_model.fit(q_values, q_targets)
    
    def learn_and_evaluate(self, training_episodes, test_interval):
        test_number = training_episodes // test_interval
        all_results = []
        for i in range(test_number):
            # send eval model to collectors, have them collect experience
            self.learn(test_interval)
            # sample experience from memory server, perform batch update on eval model
            if i % self.update_steps == 0:
                self.update_batch()
            # replace target model
            if i % self.model_replace_freq == 0:
                self.target_model.replace(self.eval_model)
            # send eval model to evaluators, record results
            avg_reward = self.evaluate()
            all_results.append(avg_reward)
        return all_results


def main():
    if len(sys.argv) == 3:
        nb_agents = sys.argv[1]
        nb_evaluators = sys.argv[2]
    else:
        nb_agents = 4
        nb_evaluators = 2

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    torch.set_num_threads(12)

    ray.shutdown()
    ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

    hps = {
            'epsilon_decay_steps' : 100000, 
            'final_epsilon' : 0.1,
            'batch_size' : 32, 
            'update_steps' : 10, 
            'memory_size' : 2000, 
            'beta' : 0.99, 
            'model_replace_freq' : 2000,
            'learning_rate' : 0.0003,
            'use_target_model': True,
            'max_episode_steps' : 500
    }

    training_episodes, test_interval = 10000, 50

    ddqn = ModelServer(hps, ReplayBuffer_remote.remote(hps['memory_size']), nb_agents, nb_evaluators)
    result = ddqn.learn_and_evaluate.remote(training_episodes, test_interval)
    plot_result(result, test_interval, nb_agents, nb_evaluators)


if __name__ == "__main__":
    main()
