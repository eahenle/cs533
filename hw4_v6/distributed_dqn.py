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
from time import sleep, time

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
def plot_result(total_rewards, learning_num, nb_agents, nb_evaluators, runtime):
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)
        
    plt.figure(num = 1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('{} agents, {} evaluators. {} seconds}'.format(nb_agents, nb_evaluators, runtime))
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.savefig("{}agents_{}evaluators".format(nb_agents, nb_evaluators))


@ray.remote
class DQN_agent_remote(object):
    def __init__(self, env, memory_server, hyper_params, action_space, agent_id):
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
        self.update_steps = hyper_params['update_steps']
        self.agent_id = agent_id
        self.memory_server = memory_server
    
    def explore_or_exploit_policy(self, state, epsilon):
        p = uniform(0, 1)
        if p < epsilon:
            #return action
            return randint(0, self.action_space - 1)
        else:
            #return action
            return self.greedy_policy(state)
        
    def greedy_policy(self, state):
        return self.eval_model.predict(state)

    def collect(self, eval_model, test_interval, epsilon):
        self.eval_model = eval_model
        for _ in tqdm(range(test_interval), desc="Training"):
            state = self.env.reset()
            done = False
            steps = 0
            while steps < self.max_episode_steps and not done: # INSERTED MY CODE HERE
                # add experience from explore-exploit policy to memory
                action = self.explore_or_exploit_policy(state, epsilon)
                state_next, reward, done, _ = self.env.step(action)
                self.memory_server.add.remote(state, action, reward, state_next, done)
                state = state_next
                steps += 1

    def pingback(self):
        return self.agent_id
                


@ray.remote
class EvalWorker():
    def __init__(self, eval_model, env, max_episode_steps, eval_trials, evaluator_id):
        self.eval_model = eval_model
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.evaluator_id = evaluator_id
        self.trials = eval_trials
    
    def greedy_policy(self, state):
        return self.eval_model.predict(state)

    def evaluate(self):
        total_reward = 0
        for _ in tqdm(range(self.trials), desc="Evaluating"):
            state = self.env.reset()
            done = False
            steps = 0
            while steps < self.max_episode_steps and not done:
                steps += 1
                action = self.greedy_policy(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
        avg_reward = total_reward / self.trials
        print(avg_reward)
        return avg_reward

    def pingback(self):
        return self.evaluator_id


class ModelServer():
    def __init__(self, hyper_params, memory_server, nb_agents, nb_evaluators, action_space=len(ACTION_DICT)):
        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']
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

        self.agents = [DQN_agent_remote.remote(CartPoleEnv(), memory_server, hyper_params, action_space, i) for i in range(nb_agents)]
        self.evaluators = [EvalWorker.remote(
            self.eval_model, CartPoleEnv(), hyper_params['max_episode_steps'], hyper_params['eval_trials'], i) for i in range(nb_evaluators)]

    # Linear decrease function for epsilon
    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def update_batch(self):
        batch = self.memory_server.sample.remote(self.batch_size)
        (states, actions, reward, next_states, is_terminal) = ray.get(batch)
        if len(states) < self.batch_size:
            return
        nonterminal_x_beta = FloatTensor([0 if t else self.beta for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size, dtype=torch.long)
        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]        
        # Calculate target
        actions, q_next = self.target_model.predict_batch(next_states)
        q_targets = reward + nonterminal_x_beta * torch.max(q_next, 1).values
        # update model
        self.eval_model.fit(q_values, q_targets)

    def learn(self, test_interval, epsilon):
        # determine which collectors are idle
        ready_ids, _ = ray.wait([agent.pingback.remote() for agent in self.agents], num_returns=1)
        ready_agents = ray.get(ready_ids)
        # send eval model to idle collectors, initiate collection
        for agent_id in ready_agents:
            self.agents[agent_id].collect.remote(self.eval_model, test_interval, epsilon)

    def evaluate(self, all_results):
        # determine which evaluators are idle
        ready_ids, _ = ray.wait([evaluator.pingback.remote() for evaluator in self.evaluators], num_returns=1)
        ready_evaluators = ray.get(ready_ids)
        # send eval model to idle evaluators, get results
        for evaluator_id in ready_evaluators:
            avg_reward = ray.get(self.evaluators[evaluator_id].evaluate.remote())
            all_results.append(avg_reward)
    
    def learn_and_evaluate(self, training_episodes, test_interval):
        test_number = training_episodes // test_interval
        all_results = []
        for i in range(test_number):
            self.steps = i * test_interval
            # Get decreased epsilon
            epsilon = self.linear_decrease(self.initial_epsilon, 
                               self.final_epsilon,
                               self.steps,
                               self.epsilon_decay_steps)
            # send eval model to collectors, have them collect experience
            self.learn(test_interval, epsilon)
            # sample experience from memory server, perform batch update on eval model
            if self.steps  % self.update_steps == 0:
                self.update_batch()
            # replace target model
            if self.steps  % self.model_replace_freq == 0:
                self.target_model.replace(self.eval_model)
            # send eval model to evaluators, record results
            self.evaluate(all_results)
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
            'batch_size' : 100, 
            'update_steps' : 10, 
            'memory_size' : 2000, 
            'beta' : 0.99, 
            'model_replace_freq' : 2000,
            'learning_rate' : 0.0003,
            'use_target_model': True,
            'max_episode_steps' : 500,
            'eval_trials' : 1
    }

    training_episodes, test_interval = 10000, 50

    print("\n\n\tDISTRIBUTED DQN\n\nHyper-parameters:\n{}\n\nTraining episodes: {}\nTest interval: {}\n# agents: {}\n# evaluators: {}\n".format(
        hps, training_episodes, test_interval, nb_agents, nb_evaluators
    ))

    print("Instantiating...")
    ddqn = ModelServer(hps, ReplayBuffer_remote.remote(hps['memory_size']), nb_agents, nb_evaluators)
    sleep(1)

    print("\nRunning...")
    start_time = time()
    result = ddqn.learn_and_evaluate(training_episodes, test_interval)
    runtime = time() - start_time
    sleep(1)
    
    print("Saving results...")
    plot_result(result, test_interval, nb_agents, nb_evaluators, runtime)
    
    print("Done! Ran in {} seconds.".format(runtime))


if __name__ == "__main__":
    main()
