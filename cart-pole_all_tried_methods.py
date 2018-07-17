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
        self.model.compile(optimizer=tf.keras.optimizers.SGD(),
                           loss=tf.keras.losses.mean_squared_error)


env = gym.make('CartPole-v0')
#defaul limit is 200!!!
env._max_episode_steps = 5000
env.reset()
# https://gym.openai.com/docs/

#
total_reward = 0
very_total_reward = 0
very_game_count = 0
max_reward = 1
epsilon = 0.1
n_epochs = 100000
discount = 0.9
model_name = 'cart-pole_successrate_lr0.1_discount_{}_epsilon{}.h5'.format(discount,epsilon)
# model = tf.keras.models.load_model(model_name)
model = Model()
while True:
    moves = []
    observation, reward, done, _ = env.step(env.action_space.sample())
    while not done:
        env.render()
        prediction = model.model.predict(np.array([observation]))[0]
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            # SEARCH MAX
            action = np.argmax(prediction)

        observation_old = observation
        observation, reward, done, _ = env.step(action)
        moves.append((observation_old, observation, prediction, action, reward, done))

        total_reward += reward

    success_rate = (total_reward / max_reward) - 1
    max_reward = max(total_reward, max_reward)
    very_total_reward += total_reward
    very_game_count += 1
    total_reward = 0

    # FIRST CONVERT PREDICTION USING ACTION AND REWARD TO A TARGET ARRAY
    # THEN for each element in moves, train the model using the observation and target array

    targets = []
    observations = []
    for idx, move in enumerate(moves):
        # SUCCESRATE IS THE NEW TARGET
        observation_old, observation_new, prediction, action, reward, done = move
        observations.append(observation_old)
        # 1 Success Rate as Reward (not learning)
        prediction[action] = success_rate

        # 2 The last 10 moves can always save the game, so we reward them with 0, else 1 not working
        # if idx < len(moves) -10:
        #     prediction[action] = 1
        # else:
        #     prediction[action] = 0

        # 3 like dotsandboxes
        # if done:
        #     prediction[action] = 0
        # else:
        #     #use what the model would predict for the next one as basis for our reward
        #     Q = np.max(model.model.predict(np.array([observation_new]))[0])
        #     prediction[action] = reward + discount * Q

        targets.append(prediction)

    observations = np.array(observations)
    targets = np.array(targets)
    # THIS IS STILL KERAS
    model.model.fit(observations, targets, verbose=0)
    if very_game_count % 1000 == 0:
        print('AVG: {} , Epoch {} , {}'.format(very_total_reward / 1000, very_game_count, model_name))
        very_total_reward = 0
        model.model.save(model_name)
        if very_game_count == n_epochs:
            break

    env.reset()
