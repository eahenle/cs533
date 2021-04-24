#!/usr/bin/env python3
import ray
from copy import deepcopy
from random import randint, choice
import sys
from contextlib import closing
import numpy as np
from six import StringIO, b
from gym import utils
from gym.envs.toy_text import discrete


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

TransitionProb = [0.7, 0.1, 0.1, 0.1]


class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4",is_slippery=True):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        rew_hole = -1000
        rew_goal = 1000
        rew_step = -1

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        self.TransitProb = np.zeros((nA, nS + 1, nS + 1))
        self.TransitReward = np.zeros((nS + 1, nA))

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'H':
                        li.append((1.0, s, 0, True))
                        self.TransitProb[a, s, nS] = 1.0
                        self.TransitReward[s, a] = rew_hole
                    elif letter in b'G':
                        li.append((1.0, s, 0, True))
                        self.TransitProb[a, s, nS] = 1.0
                        self.TransitReward[s, a] = rew_goal
                    else:
                        if is_slippery:
                            #for b in [(a-1)%4, a, (a+1)%4]:
                            for b, p in zip([a, (a+1)%4, (a+2)%4, (a+3)%4], TransitionProb):
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                #rew = float(newletter == b'G')
                                #li.append((1.0/10.0, newstate, rew, done))
                                if newletter == b'G':
                                    rew = rew_goal
                                elif newletter == b'H':
                                    rew = rew_hole
                                else:
                                    rew = rew_step
                                li.append((p, newstate, rew, done))
                                self.TransitProb[a, s, newstate] += p
                                self.TransitReward[s, a] = rew_step
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def GetSuccessors(self, s, a):
        next_states = np.nonzero(self.TransitProb[a, s, :])
        probs = self.TransitProb[a, s, next_states]
        return [(s,p) for s,p in zip(next_states[0], probs[0])]

    def GetTransitionProb(self, s, a, ns):
        return self.TransitProb[a, s, ns]

    def GetReward(self, s, a):
        return self.TransitReward[s, a]

    def GetStateSpace(self):
        return self.TransitProb.shape[1]

    def GetActionSpace(self):
        return self.TransitProb.shape[0]


@ray.remote
class VI_server_v2(object): # same as v1
    def __init__(self,size):
        self.v_current = [0] * size
        self.pi = [0] * size
        self.v_new = [0] * size

    def get_value_and_policy(self):
        return self.v_current, self.pi

    def update(self, update_index, update_v, update_pi):
        self.v_new[update_index] = update_v
        self.pi[update_index] = update_pi

    def get_error_and_update(self):
        max_error = 0
        for i in range(len(self.v_current)):
            error = abs(self.v_new[i] - self.v_current[i])
            if error > max_error:
                max_error = error
            self.v_current[i] = self.v_new[i]

        return max_error


@ray.remote
def VI_worker_v2(VI_server, data, start_state, end_state): # operates over contiguous chunk of state-space
        env, workers_num, beta, epsilon = data
        A = env.GetActionSpace()
        S = env.GetStateSpace()

        # get the infos
        V, _ = ray.get(VI_server.get_value_and_policy.remote())

        # iterate over states in batch
        for state in range(start_state, end_state):
            # bellman backup
            max_v = float('-inf')
            max_a = 0
            for action in range(A):
                sum_TV = 0
                for SP in env.GetSuccessors(state, action):
                    sum_TV += SP[1] * V[SP[0]]
                v_a = env.GetReward(state, action) + beta * sum_TV
                if v_a > max_v:
                    max_v = v_a
                    max_a = action
            VI_server.update.remote(state, max_v, max_a) # update value and policy

        # return ith worker
        return start_state, end_state


def fast_value_iteration(env, beta = 0.999, epsilon = 0.01, workers_num = 4):
    S = env.GetStateSpace()
    VI_server = VI_server_v2.remote(S)
    data_id = ray.put((env, workers_num, beta, epsilon))

    # generate batches
    batch_size = int((S - 1) / workers_num)
    start_states = range(0, S, batch_size)

    # start workers on batches and store IDs in workers_list
    workers_list = [VI_worker_v2.remote(VI_server, data_id, start_states[i], start_states[i] + batch_size) for i in range(workers_num)]

    error = float('inf')
    k = 0
    while error > epsilon:
        # wait for the workers to return and update the error
        finished_worker_ids = ray.wait(workers_list, num_returns=workers_num, timeout=None)[0][0]
        finished_worker_states = ray.get(finished_worker_ids)
        error = ray.get(VI_server.get_error_and_update.remote())
        # restart the workers
        workers_list = [VI_worker_v2.remote(VI_server, data_id, start_states[i], start_states[i] + batch_size) for i in range(workers_num)]
        k += 1

    return ray.get(VI_server.get_value_and_policy.remote())


if __name__ == "__main__":
    MAPS = {"4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]}
    map_4 = (MAPS["4x4"], 4)
    MAP = map_4
    map_size = MAP[1]

    ray.shutdown()
    ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

    v, pi = fast_value_iteration(FrozenLakeEnv(desc = MAP[0], is_slippery = True))

    v_np, pi_np  = np.array(v), np.array(pi)
    print("\nState Value:\n")
    print(np.array(v_np[:-1]).reshape((map_size,map_size)))
    print("\nPolicy:\n")
    print(np.array(pi_np[:-1]).reshape((map_size,map_size)))
