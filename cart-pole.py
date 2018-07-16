import gym
import tensorflow as tf
import numpy as np
import tensorboard

class Model():
    def __init__(self):
        self.model = tf.keras.Sequential()
        #input layer a 2D input with shape `(batch_size, input_dim)`
        self.model.add(tf.keras.layers.Dense(32,activation='relu',input_shape=(4,)))
        self.model.add(tf.keras.layers.Dense(64,activation='relu'))
        self.model.add(tf.keras.layers.Dense(32,activation='relu'))
        self.model.add(tf.keras.layers.Dense(2))
        self.model.compile(optimizer=tf.keras.optimizers.SGD(),
                      loss=tf.keras.losses.mean_squared_error)

env = gym.make('CartPole-v0')
env.reset()

# TODO FIRST ONLY EVALUATE AT THE END
# https://gym.openai.com/docs/

total_reward = 0
max_reward = 1
model = Model()
while True:
    moves = []
    observation, reward, done, _ = env.step(env.action_space.sample())
    while not done:
        env.render()
        prediction = model.model.predict(np.array([observation]))[0]
        #SEARCH MAX
        action = np.argmax(prediction)

        moves.append((observation, prediction, action))

        _, reward, done, _ = env.step(action)

        total_reward += reward


    succes_rate = (total_reward/max_reward)-1
    max_reward = max(total_reward, max_reward)
    print(total_reward)
    total_reward = 0

    #TODO TRAIN
    #FIRST CONVERT PREDICTION USING ACTION AND REWARD TO A TARGET ARRAY
    #THEN for each element in moves, train the model using the observation and target array

    targets = []
    observations = []
    for move in moves:

        #SUCCESRATE IS THE NEW TARGET
        observation, prediction, action = move
        observations.append(observation)
        prediction[action] = succes_rate
        targets.append(prediction)

    observations = np.array(observations)
    targets = np.array(targets)
    model.model.fit(observations, targets,verbose=0)


    env.reset()