#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Hsin-Yu Chang <acht7111020@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT
import random
import time

import numpy as np
import tensorflow as tf
import scipy.signal
from skimage.transform import resize as imresize

T = 0
slim = tf.contrib.slim


def sample_policy_action(probs):
    probs = probs - np.finfo(np.float32).epsneg
    histogram = np.random.multinomial(1, probs)
    action_index = int(np.nonzero(histogram)[0])
    return action_index


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


class Worker:
    def __init__(self, thread_id, output_size, learning_rate, sess, tmax=20, glob_net=None, save_path=None, AIchar='Firen'):
        self.thread_id = thread_id
        self.scope = "net" + str(thread_id)

        self.state_size = (80, 190, 4)
        self.output_size = output_size

        self.input_state = tf.placeholder("float", [None, 80, 190, 4])
        self.input_pos = tf.placeholder("float", [None, 7])
        self.input_hps = tf.placeholder("float", [None, 2])
        self.pre_actions = tf.placeholder("float", [None, self.output_size])

        self.learning_rate = learning_rate
        self.character = AIchar
        self.sess = sess
        self.tmax = tmax
        self.TMAX = 80000000

        if save_path != None:
            self.savepath = save_path

        global T

        with tf.variable_scope(self.scope) as scope:
            self._create_network()
            self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

        if glob_net != None:
            self.glob_net = glob_net
            self.sync_op = self._create_sync_operation()
            self._create_loss()
            self._create_optimizer()


    def _create_network(self):
        self.conv1 = lrelu(slim.conv2d(self.input_state, 32, activation_fn=None, kernel_size=[8,8], stride=[4,4], padding='SAME')) # (?, 20, 48, 32)
        self.conv2 = lrelu(slim.conv2d(self.conv1, 64, activation_fn=None, kernel_size=[4,4], stride=[2,2], padding='SAME')) # (?, 10, 24, 64)
        self.conv3 = lrelu(slim.conv2d(self.conv2, 64, activation_fn=None, kernel_size=[3,3], stride=[1,1], padding='SAME')) # (?, 10, 24, 64)
        flat = slim.flatten(self.conv3)
        hidden = tf.concat([flat, self.input_pos, self.pre_actions], 1)

        #Recurrent network for temporal dependencies
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(512, state_is_tuple=True) # 263
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        self.state_in = (c_in, h_in)
        step_size = tf.shape(self.input_state)[:1]
        state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm_cell, tf.expand_dims(hidden, [0]), initial_state=state_in, sequence_length=step_size,
            time_major=False)

        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        rnn_out = tf.reshape(lstm_outputs, [-1, 512])

        #Output layers for policy and value estimations
        self.policy = slim.fully_connected(rnn_out, self.output_size,
            activation_fn=tf.nn.softmax,
            weights_initializer=normalized_columns_initializer(0.01),
            biases_initializer=None)
        self.value = slim.fully_connected(rnn_out, 1,
            activation_fn=None,
            weights_initializer=normalized_columns_initializer(1.0),
            biases_initializer=None)


    def _create_loss(self):
        self.advantage = tf.placeholder("float", [None])
        self.targets = tf.placeholder("float", [None])
        self.actions = tf.placeholder("float", [None, self.output_size])
        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions, [1])

        #Loss functions
        self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.targets - tf.reshape(self.value, [-1])))
        self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
        self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantage)
        # In order to explore more, we want the entropy be bigger, then the loss is smaller
        self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01


    def _create_optimizer(self):
        # add norm
        trainer = tf.train.RMSPropOptimizer(self.learning_rate, epsilon=0.1, decay=0.99)
        self.gradients = tf.gradients(self.loss, self.train_vars)
        self.var_norms = tf.global_norm(self.train_vars)
        grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

        self.opt = trainer.apply_gradients(zip(grads, self.glob_net.train_vars))


    def _create_sync_operation(self):
        sync = [self.train_vars[j].assign(self.glob_net.train_vars[j]) for j in range(len(self.train_vars))]
        return tf.group(*sync)


    def train(self, env, checkpoint_interval, saver, t_writer, gamma=0.99):
        global T

        self.saver = saver
        time.sleep(3 * self.thread_id)
        action_size = env.action_space.n
        print('Starting thread ' + str(self.thread_id))

        state = env.reset()
        env.env.set_ai_epsilon(0)
        state = preproc(state, self.state_size)

        logg = env.get_detail()
        if logg[0]['name'] == str(self.character): # if is RL agent
            my_index = 0
        else:
            my_index = 1

        episode_reward, total_cost, counter = 0, 0, 0
        terminal = False

        pre_action_list = np.zeros([self.output_size])

        rnn_state = self.state_init
        self.batch_rnn_state = rnn_state
        while T < self.TMAX:

            states, actions, prev_reward, state_values, state_pos, state_pre_actions = [], [], [], [], [], []
            prev_ep_reward = 0

            t = 0
            self.sess.run(self.sync_op)

            # init rnn batch state
            while not (terminal or (t == self.tmax)):
                logg = env.get_detail()
                my_pos = get_position(logg, my_index)

                probs, v, rnn_state = self.sess.run(
                                            [self.policy, self.value, self.state_out], # rnn_state, self.state_out
                                            feed_dict = {self.input_state : [state],
                                                         self.input_pos: [my_pos],
                                                         self.pre_actions: [pre_action_list],
                                                         self.state_in[0] : rnn_state[0],
                                                         self.state_in[1] : rnn_state[1]})

                probs = probs[0]
                v = v[0][0]

                action_list = np.zeros([self.output_size])
                action_index = sample_policy_action(probs)
                action_list[action_index] = 1

                actions.append(action_list)
                states.append(state)
                state_values.append(v)
                state_pre_actions.append(pre_action_list)
                state_pos.append(my_pos)

                new_state, reward, terminal, info = env.step(action_index)
                new_state = preproc(new_state, self.state_size)

                if info == False:
                    state = new_state
                    print('Thread [%d] has no info, stop update model ...' % (self.thread_id))
                    continue

                episode_reward += reward

                # For debug
                if self.thread_id == 1 and t % 50 == 0:
                    print(self.thread_id, 'action: ', action_index, ', reward: ', reward, ', hps:', env.env.hps)

                prev_reward.append(reward)
                prev_ep_reward = reward

                state = new_state
                pre_action_list = action_list
                T += 1
                t += 1
                counter += 1

                if T % checkpoint_interval == 0:
                    self.saver.save(self.sess, self.savepath+"/model.ckpt" )

                if t % 100 == 0:
                    print("Thread : %d, ep reward: %2f, hps: %s" % (self.thread_id, episode_reward, env.env.hps))
                    t_writer.write("Thread : %d, ep reward: %2f, hps: %s\n" % (self.thread_id, episode_reward, env.env.hps))
                    # self.writer.write("Thread %d: reward: %2f, hps: %s\n" %(reward, env.env.hps))

            if terminal:
                R_t = 0
            else:
                R_t = self.sess.run(self.value,
                                    feed_dict = {self.input_state : [state],
                                                 self.input_pos: [my_pos],
                                                 self.pre_actions: [pre_action_list],
                                                 self.state_in[0] : rnn_state[0],
                                                 self.state_in[1] : rnn_state[1]})
                R_t = R_t[0][0]

            state_values.append(R_t)
            targets = np.zeros((t))

            for i in range(t - 1, -1, -1):
                R_t = prev_reward[i] + gamma * R_t
                targets[i] = R_t

            delta = np.array(prev_reward) + gamma * np.array(state_values[1:]) - np.array(state_values[:-1])
            advantage = scipy.signal.lfilter([1], [1, -gamma], delta[::-1], axis=0)[::-1]

            # every T step or terminal
            cost, _, self.batch_rnn_state  = self.sess.run(
                                                    [self.loss, self.opt, self.state_out],  #  self.batch_rnn_state , self.state_out
                                                    feed_dict = {self.input_state: states,
                                                                 self.input_pos: state_pos,
                                                                 self.actions: actions,
                                                                 self.targets: targets,
                                                                 self.advantage: advantage,
                                                                 self.pre_actions: state_pre_actions,
                                                                 self.state_in[0]:self.batch_rnn_state[0],
                                                                 self.state_in[1]:self.batch_rnn_state[1]})
            total_cost += cost

            if terminal:
                terminal = False
                print( "Thread: %d, Time: %10d, Reward: %2.4f, Loss: %3.3f" %(self.thread_id, T, episode_reward, total_cost/counter))
                t_writer.write("Finish. Thread %d: Time: %10d, Reward: %2.4f, hps: %s, loss: %3.3f\n" %(self.thread_id, T, episode_reward, env.env.hps, total_cost/counter))

                episode_reward, total_cost, counter = 0, 0, 0

                rnn_state = self.state_init
                self.batch_rnn_state = rnn_state

                # save for debug
                env.debug(self.thread_id)

                state = env.reset()
                state = preproc(state, self.state_size)
                env.env.set_ai_epsilon(0)
                pre_action_list = np.zeros([self.output_size])


    def test(self, env, render=False):
        terminal = False
        state = env.reset()
        state, _, _, _ = env.step(0)
        state = preproc(state, self.state_size)
        logg = env.get_detail()
        if logg[0]['name'] == self.character:
            my_index = 0
        else:
            my_index = 1

        episode_reward, t = 0, 0
        rnn_state = self.state_init
        self.batch_rnn_state = rnn_state
        for record in range(10):
            pre_action_list = np.zeros([self.output_size])
            print('start', record)
            count = 0

            while not terminal:
                logg = env.get_detail()
                my_pos = get_position(logg, my_index)

                probs, v, rnn_state = self.sess.run(
                                            [self.policy, self.value, self.state_out],
                                             feed_dict = {self.input_state : [state],
                                                          self.input_pos: [my_pos],
                                                          self.pre_actions: [pre_action_list],
                                                          self.state_in[0] : rnn_state[0],
                                                          self.state_in[1] : rnn_state[1]})
                probs = probs[0]
                v = v[0][0]

                action_list = np.zeros([self.output_size])
                action_index = sample_policy_action(probs)
                action_list[action_index] = 1

                if random.random() < 0.05:
                    action_index = 4 # move to right
                else:
                    probs = probs - np.finfo(np.float32).epsneg
                    action_index = np.argmax(probs)

                new_state, reward, terminal, _ = env.step(action_index)
                new_state = preproc(new_state, self.state_size)
                if render:
                    env.render()
                state = new_state
                pre_action_list = action_list
                episode_reward += reward

                print("Ep count: %3d, reward: %2f, hps: %s            " % (count, reward, env.env.hps), end='\r')
                count += 1

                if count > 0 and count % 1000 == 0:
                    name = str(record) + '-' + str(count) + '.mp4'
                    env.save_recording(name)

            if terminal:
                terminal = False
                print("Finish. Thread %d: Time: %10d, Reward: %2.4f, hps: %s\n" %(self.thread_id, T, episode_reward, env.env.hps))

                name = str(record) + '.mp4'
                env.save_recording(name)
                episode_reward, count = 0, 0
                rnn_state = self.state_init
                self.batch_rnn_state = rnn_state
                state = env.reset()
                state = preproc(state, self.state_size)


def get_position(logg, my_index):
    my_pos = np.zeros((7))
    if logg is not None:
        my_pos[0] = logg[my_index]['x'] / 800.
        my_pos[1] = logg[my_index]['y'] / 20.
        my_pos[2] = logg[my_index]['z'] / 200.
        my_pos[3] = logg[1-my_index]['x'] / 800.
        my_pos[4] = logg[1-my_index]['y'] / 20.
        my_pos[5] = logg[1-my_index]['z'] / 200.
        my_pos[6] = (my_pos[0]-my_pos[3])**2 + (my_pos[1]-my_pos[4])**2 + (my_pos[2]-my_pos[5])**2
    else:
        print('no pos')
    return my_pos


def preproc(obs, res_size=(80, 190, 4)):
    resize = imresize(obs, (res_size))
    return resize.astype(np.float32) / 255.0
