import gym
import tensorflow as tf

class Model():
    def __init__(self):
        self.model = tf.keras.Sequential()

env = gym.make('CartPole-v0')
env.reset()

# TODO FIRST ONLY EVALUATE AT THE END
# https://gym.openai.com/docs/
while True:
    env.render()
    env.step(env.action_space.sample())
