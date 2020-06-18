#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Hsin-Yu Chang <acht7111020@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT
import argparse
import threading
import os

import tensorflow as tf
import numpy as np

from a3c_network import *


# Sample code from here
# https://github.com/papoudakis/a3c-tensorflow
def main(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 8
    config.inter_op_parallelism_threads = 8
    sess = tf.InteractiveSession(config=config)

    if args.train:
        workers = []
        import lf2gym
        from lf2gym import Difficulty

        # url = 'http://127.0.0.1:' + str(args.port)
        envs = [lf2gym.make(
                    startServer=True,
                    difficulty=Difficulty.Challengar,
                    port=args.port+i) for i in range(args.thread)]
        # [env.reduce_action_space(13) for env in envs]
        try:
            num_actions = envs[0].action_space.n
            global_network = Worker(-1, num_actions, args.learning_rate, sess, AIchar=envs[0].characters[0])

            for i in range(args.thread):
                workers.append(Worker(i, num_actions, args.learning_rate, sess, glob_net=global_network, save_path=args.savepath, AIchar=envs[i].characters[0]))
            saver = tf.train.Saver(max_to_keep=35)

            if args.retrain:
                latest_ckpt = tf.train.latest_checkpoint(args.loadpath)
                saver.restore(sess, latest_ckpt)
                print("restore from", latest_ckpt)
            else:
                sess.run(tf.global_variables_initializer())

            if not os.path.exists(args.savepath):
                os.mkdir(args.savepath)

            path = args.savepath + 'output_thread.log'
            t_writer = open(path, 'w')
            t_writer.write(args.comment)
            t_writer.write('\nstart training a3c, thread 1 log...\n')
            actor_learner_threads = [threading.Thread(target=workers[i].train, args=(envs[i], args.checkpoint_interval, saver, t_writer)) for i in range(args.thread)]
            for t in actor_learner_threads:
                t.start()
            for t in actor_learner_threads:
                t.join()

        except KeyboardInterrupt:
            print("W: interrupt received, stopping...")

        finally:
            [env.close() for env in envs]

    if args.test:
        import lf2gym # difficulty='CRUSHER 1.0'
        from lf2gym import Character, Difficulty
        # env = lf2gym.make(startServer=False, autoStart=True, rewardList=['hp'], difficulty=Difficulty.Challengar, wrap='skip4' )
        env = lf2gym.make(
                startServer=True,
                difficulty=Difficulty.Challengar,
                port=args.port,
                action_options=['Basic', 'AJD', 'No Combos'])
        env.start_recording()
        num_actions = env.action_space.n
        global_network = Worker(-1, num_actions, args.learning_rate, sess, AIchar=env.characters[0])

        saver = tf.train.Saver()

        latest_ckpt = tf.train.latest_checkpoint(args.loadpath)
        saver.restore(sess, latest_ckpt)
        print("restore from", latest_ckpt)

        global_network.test(env, render=False)

    if args.test_chrome:
        import lf2gym
        from lf2gym import WebDriver, Character, Difficulty
        env = lf2gym.make(
                startServer=False,
                driverType=WebDriver.Chrome,
                canvasSize=(600, 800),
                characters=[Character.Firen, Character.Freeze],
                versusPlayer=False,
                difficulty=Difficulty.Crusher)
        # env.reduce_action_space(13)

        num_actions = env.action_space.n
        global_network = Worker(-1, num_actions, args.learning_rate, sess, AIchar=env.characters[0])

        saver = tf.train.Saver()
        latest_ckpt = tf.train.latest_checkpoint(args.loadpath)
        saver.restore(sess, latest_ckpt)
        print("restore from", latest_ckpt)

        global_network.test(env, render=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test_chrome", action="store_true")
    parser.add_argument("--test", action="store_true")

    parser.add_argument('--thread', type=int, default=8, help='Number of threads to use during training.')
    parser.add_argument('--tmax', type=int, default=80000000, help='Number of training timesteps.')
    parser.add_argument('--port', type=int, default=8000, help='Server Port.')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Reward discount rate.')

    parser.add_argument('--savepath', type=str, default='/path/to/model/', help="Path to save model.")
    parser.add_argument('--loadpath', type=str, default='/path/to/model/', help="Path to load model.")
    parser.add_argument('--checkpoint_interval', type=int, default=50000, help='Save the model every n steps.')
    parser.add_argument('--comment', type=str, default='', help="Comment.")
    args = parser.parse_args()
    main(args)

# Terminal:
# For train: python a3c.py --train --savepath 'model_a3c/'
# For test: python a3c.py --test --loadpath 'model_a3c/'
