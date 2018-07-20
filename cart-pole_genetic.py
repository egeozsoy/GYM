import gym
import math
import numpy as np
from random import uniform
import time
from multiprocessing import Pool
import os

env = gym.make('CartPole-v0')
# defaul limit is 200!!!
env._max_episode_steps = 5000
env.reset()

epoch = 100

def f(obs, params):
    tmp = params[0] + params[1]*obs[0] + params[2]*obs[1] + params[3]*obs[2] + params[4]*obs[3]
    return 1 if tmp > 0.5 else 0

def str_player(p):
    return '{} {} {} {} {} \n'.format(p[0],p[1],p[2],p[3],p[4])

def play_100_games(player):
    total_reward = 0
    for i in range(100):
        observation, reward, done, _ = env.step(env.action_space.sample())
        while not done:
            # env.render()
            action = f(observation, player)
            observation, reward, done, _ = env.step(action)
            total_reward += reward

        env.reset()

    avg_reward = total_reward/100
    return (avg_reward, player)


def init():
    players = []
    #load save
    if os.path.isfile('function.npy'):
        players = np.load('function.npy').tolist()
    #new random
    else:
        for i in range (100):
            toAdd = [uniform(-10,10),uniform(-10,10),uniform(-10,10),uniform(-10,10),uniform(-10,10)]
            players.append(toAdd)
    return players

def get_best(res):
    best = max(res)
    return best[0]

def make_avg_survivor(players):
    player = []
    for i in range(5):
        a = 0
        for p in players:
            a += p[1][i]
        a /= 10
        player.append(a)

    return player

def add_varianz(player):
    return [player[0] + uniform(-2,2),player[1] + uniform(-2,2),player[2] + uniform(-2,2),player[3] + uniform(-2,2),player[4] + uniform(-2,2)]


def new_generation(players):
    survivors = sorted(players,reverse=True)[0:10]
    new_gen = []

    #new players as varianz from average
    avg_survivor = make_avg_survivor(survivors)
    for i in range(30):
        new_player = add_varianz(avg_survivor)
        new_gen.append(new_player)

    #new players as average from single survivors
    for p in survivors:
        new_gen.append(p[1])
        for i in range(4):
            new_player = add_varianz(p[1])
            new_gen.append(new_player)

    #new random players
    for i in range (20):
        new_player = [uniform(-10,10),uniform(-10,10),uniform(-10,10),uniform(-10,10),uniform(-10,10)]
        new_gen.append(new_player)

    return new_gen


players = init()
# players = [[0.8538134387693717, 0.8782469014832657, 7.744378202348017, 9.02427854572888, 7.925810882115055]]

for i in range(epoch):

    with Pool(8) as p:
        res = p.map(play_100_games,players)

    print("best in generation {}: {}".format(i, get_best(res)))
    players = new_generation(res)

    np.save('function',np.array(players))

# best in generation 0: 1491.52
# best in generation 1: 1095.73
# best in generation 2: 2855.95
# best in generation 3: 4999.0
# best in generation 4: 4999.0
# best in generation 5: 4999.0
