import gym
import tensorflow as tf
import numpy as np
import tensorboard
import random


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


env = gym.make('CartPole-v0')
# defaul limit is 200!!!
env._max_episode_steps = 5000
env.reset()
# if saved model then load it

# https://gym.openai.com/docs/

# hyper params
total_reward = 0
very_total_reward = 0
very_game_count = 0
epsilon = 0.1
n_epochs = 500000
discount = 0.9
model_name = 'cart-pole_dotsandboxes_lr0.1_discount_{}_epsilon{}_64_2.h5'.format(discount, epsilon)
model = Model()
model.model = tf.keras.models.load_model(model_name)
while True:
    moves = []
    observation, reward, done, _ = env.step(env.action_space.sample())
    while not done:
        # only for visualizing
        env.render()
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

    very_total_reward += total_reward
    very_game_count += 1
    total_reward = 0
    #
    targets = []
    observations = []
    for idx, move in enumerate(moves):
        observation_old, observation_new, prediction, action, reward, done = move
        observations.append(observation_old)
        # like dotsandboxes
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
    if very_game_count % 10000 == 0:
        print('AVG: {} , Epoch {} , {}'.format(very_total_reward / 10000, very_game_count, model_name))
        very_total_reward = 0
        model.model.save(model_name)
        if very_game_count == n_epochs:
            break

    env.reset()
