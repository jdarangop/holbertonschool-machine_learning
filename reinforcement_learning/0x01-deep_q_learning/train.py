#!/usr/bin/env python3
""" Train """
import keras as K
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
import numpy as np
import rl
import gym
from PIL import Image


class AtariProcessor(Processor):
    """ AtariProcessor """

    def process_observation(self, observation):
        """ Processor observation """
        img = Image.fromarray(observation)
        img = img.resize((84, 84)).convert('L')
        proc_observation = np.array(img)
        """
        processed = tf.image.rgb_to_grayscale(observation)
        processed = tf.image.crop_to_bounding_box(processed,
                                                  34,
                                                  0, 160,
                                                  160)
        proc_observation = tf.image.resize_images(processed,
                                                  [84, 84],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        """

        return proc_observation.astype('uint8')

    def process_state_batch(self, batch):
        """ process state batch. """
        proc_batch = batch.astype('float32') / 255
        return proc_batch

    def process_reward(self, reward):
        """ process reward. """
        return np.clip(reward, -1, 1)


def build_model(window_length, nb_actions):
    """ build the model based in the deepmind paper.
        Args:
            input_shape: (tuple) the shape of the
                         processed images, and the windows.
            nb_actions: (int) number of actions.
        Returns:
            (K.Model) the model.
    """
    input_shape = (window_length, ) + (84, 84)
    input_layer = K.layers.Input(shape=input_shape)
    permute = K.layers.Permute((2, 3, 1))(input_layer)
    conv_1 = K.layers.Conv2D(16, (8, 8), strides=4, activation='relu')(permute)
    conv_2 = K.layers.Conv2D(32, (4, 4), strides=2, activation='relu')(conv_1)
    flattened = K.layers.Flatten()(conv_2)
    hidden = K.layers.Dense(256, activation='relu')(flattened)

    output = K.layers.Dense(nb_actions, activation='linear')(hidden)

    model = K.models.Model(input_layer, output)

    # print(model.summary())
    return model


if __name__ == '__main__':

    env = gym.make('Breakout-v0')
    np.random.seed(12345)
    env.seed(12345)

    window_length = 4
    nb_actions = env.action_space.n

    model = build_model(window_length, nb_actions)

    processor = AtariProcessor()

    memory = rl.memory.SequentialMemory(limit=1000000,
                                        window_length=window_length)

    policy = rl.policy.LinearAnnealedPolicy(rl.policy.EpsGreedyQPolicy(),
                                            attr='eps',
                                            value_max=1,
                                            value_min=0.1,
                                            value_test=0.05,
                                            nb_steps=1000000)

    dqn = rl.agents.dqn.DQNAgent(model=model,
                                 nb_actions=nb_actions,
                                 policy=policy,
                                 memory=memory,
                                 processor=processor,
                                 nb_steps_warmup=50000,
                                 gamma=0.99,
                                 train_interval=4,
                                 delta_clip=1)

    # optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    dqn.compile(optimizer=K.optimizers.Adam(lr=0.00025), metrics=['mse'])

    dqn.fit(env, nb_steps=17, visualize=False)

    dqn.save_weights('policy.h5')
