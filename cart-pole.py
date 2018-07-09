import gym
import tensorflow as tf
import numpy as np

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
model = Model()
while True:
    done = False
    moves = []
    observation, reward, done, info = env.step(env.action_space.sample())
    while not done:
        env.render()
        prediction = model.model.predict(np.array([observation]))[0]
        print(prediction)
        #SEARCH MAX
        action = np.argmax(prediction)

        moves.append((observation, prediction,action))

        observation, reward, done, info = env.step(action)

        total_reward += reward

    #TODO TRAIN
    #FIRST CONVERT PREDICTION USING ACTION AND REWARD TO A TARGET ARRAY
    #THEN for each element in moves, train the model using the observation and target array
