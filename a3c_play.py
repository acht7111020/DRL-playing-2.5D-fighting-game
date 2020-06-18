import os
import threading

import numpy as np
import tensorflow as tf

from a3c_network import *


# Sample code from here
def main(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    import lf2gym
    from lf2gym import WebDriver, Character
    env = lf2gym.make(
                startServer=True,
                driverType=WebDriver.Chrome,
                characters=[Character.Firen, Character[args.player]],
                versusPlayer=True)
    env.start_recording()

    num_actions = env.action_space.n
    global_network = Worker(-1, num_actions, 1e-4, sess, AIchar=env.characters[0])

    saver = tf.train.Saver()
    latest_ckpt = tf.train.latest_checkpoint(args.loadpath)
    saver.restore(sess, latest_ckpt)
    print("restore from", latest_ckpt)

    global_network.test(env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loadpath', type=str, default='/path/to/model/', help="Path to load model.")
    parser.add_argument('--player', type=str, default='Bandit', help="Player that you want to play.")
    parser.add_argument('--comment', type=str, default='', help="Comment.")
    args = parser.parse_args()
    main(args)

# Terminal:
# For train: python a3c.py --train --savepath 'model_a3c/'
# For test: python a3c.py --test --loadpath 'model_a3c/'
