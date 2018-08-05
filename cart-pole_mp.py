import gym
import tensorflow as tf
import numpy as np
from multiprocessing import Pool , cpu_count
import os
import random
from random import uniform


class Model():

    def __init__(self):
        self.model = tf.keras.Sequential()
        # input layer a 2D input with shape `(batch_size, input_dim)`
        self.model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)))
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dense(2))
        self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.003),
                           loss=tf.keras.losses.mean_squared_error)  # sgd lr=0.003 was good

#make the whole think multiprocess
def play_and_train(player):
    # hyper params
    model_name_prefix = 'cart-pole_multiprocess'
    model_name_suffix = '.h5'
    epsilon = 0.1
    n_epochs = 500000
    discount = 0.9
    model = Model()
    pid = os.getpid()
    was_random = True
    model_name = '{}pid_{}{}'.format(model_name_prefix, pid, model_name_suffix)
    #loads the base model which was the best model from the last run
    #there is a chance no model will be loaded, and this pid will start from scratch(like mutating)
    if os.path.isfile(model_name_prefix + model_name_suffix) and uniform(0, 1) > 0.1:
        was_random = False
        model.model = tf.keras.models.load_model(model_name_prefix + model_name_suffix)

    total_reward = 0
    env = gym.make('CartPole-v0')
    # defaul limit is 200!!!
    env._max_episode_steps = 5000
    env.reset()
    n_epoch = 1000
    #playing
    for i in range(n_epoch):
        moves = []
        observation, reward, done, _ = env.step(env.action_space.sample())
        while not done:
            # only for visualizing
            # env.render()
            prediction = model.model.predict(np.array([observation]))[0]
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                # SEARCH MAX
                action = np.argmax(prediction)

            observation_old = observation
            observation, reward, done, _ = env.step(action)
            # training data
            moves.append((observation_old, observation, prediction, action, reward, done))

            total_reward += reward
        env.reset()

        #training (after every game)
        targets = []
        observations = []
        for idx, move in enumerate(moves):
            observation_old, observation_new, prediction, action, reward, done = move
            observations.append(observation_old)
            if done:
                prediction[action] = 0
            else:
                # use what the model would predict for the next one as basis for our reward
                Q = np.max(model.model.predict(np.array([observation_new]))[0])
                prediction[action] = reward + discount * Q

            targets.append(prediction)

        observations = np.array(observations)
        targets = np.array(targets)
        model.model.fit(observations, targets, verbose=0)

    # print('AVG: {} , {}'.format(total_reward / n_epoch, model_name))
    model.model.save(model_name)

    return (total_reward / n_epoch, model_name,was_random)


if __name__ == '__main__':
    cpus = cpu_count()
    pool = Pool(processes=cpus)
    model_name_prefix = 'cart-pole_multiprocess'
    model_name_suffix = '.h5'
    model_name = model_name_prefix + model_name_suffix
    for i in range(10):
        #create as many players as there are cpus
        players = [range(cpus)]
        results = pool.map(play_and_train, players)
        # select the best of 8
        avg, best_model,was_random = sorted(results, reverse=True)[0]
        #remove the old general model
        if os.path.isfile(model_name):
            os.remove(model_name)
        #change best temp model to general model
        print('Model {} selected. Was_random: {}'.format(best_model,was_random))
        os.rename(best_model, model_name)

        #remove all unnecessary files
        for file in os.listdir():
            if file[-3:] == '.h5' and file != model_name and 'multiprocess' in file:
                os.remove(file)

        print('Max Avg {} in epoch {}'.format(avg, i))

# reached just after 3 epochs
# Model cart-pole_multiprocesspid_93133.h5 selected. Was_random: False
# Max Avg 3351.15 in epoch 2