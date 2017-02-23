from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.contrib.legacy_seq2seq import sequence_loss

import ntm_cell

import os
from utils import progress

def softmax_loss_function(labels, inputs):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=inputs)

def sp_loss_function(labels, inputs):

    inputs_a, inputs_b = tf.split(inputs, 2, 0)
    labels_a, labels_b = tf.split(labels, 2, 0)

    loss_a = tf.nn.softmax_cross_entropy_with_logits(labels=labels_a, logits=inputs_a)
    loss_b = tf.nn.softmax_cross_entropy_with_logits(labels=labels_b, logits=inputs_b)

    return loss_a + loss_b



class NTM(object):
    def __init__(self, cell, sess,
                 min_length, max_length,
                 test_max_length=120,
                 min_grad=-10, max_grad=+10,
                 lr=1e-4, momentum=0.9, decay=0.95,
                 scope="NTM", forward_only=False):
        """Create a neural turing machine specified by NTMCell "cell".

        Args:
            cell: An instantce of NTMCell.
            sess: A TensorFlow session.
            min_length: Minimum length of input sequence.
            max_length: Maximum length of input sequence for training.
            test_max_length: Maximum length of input sequence for testing.
            min_grad: (optional) Minimum gradient for gradient clipping [-10].
            max_grad: (optional) Maximum gradient for gradient clipping [+10].
            lr: (optional) Learning rate [1e-4].
            momentum: (optional) Momentum of RMSProp [0.9].
            decay: (optional) Decay rate of RMSProp [0.95].
        """
        if not isinstance(cell, ntm_cell.NTMCell):
            raise TypeError("cell must be an instance of NTMCell")

        self.cell = cell
        self.sess = sess
        self.scope = scope

        self.lr = lr
        self.momentum = momentum
        self.decay = decay

        self.min_grad = min_grad
        self.max_grad = max_grad
        self.min_length = min_length
        self.max_length = max_length
        self._max_length = max_length

        if forward_only:
            self.max_length = test_max_length

        self.inputs = []
        self.outputs = []
        self.output_logits = []
        self.true_outputs = []
        self.answer = []


        self.prev_states = {}
        self.input_states = defaultdict(list)
        self.output_states = defaultdict(list)

        self.start_symbol = tf.placeholder(tf.float32, [self.cell.input_dim],
                                           name='start_symbol')
        self.end_symbol = tf.placeholder(tf.float32, [self.cell.input_dim],
                                         name='end_symbol')

        self.losses = 0
        self.optim = 0
        self.grads = []

        self.saver = None
        self.params = None

        with tf.variable_scope(self.scope):
            self.global_step = tf.Variable(0, trainable=False)

        self.build_model(forward_only)

    def build_model(self, forward_only, is_copy=False):
        print(" [*] Building a NTM model")

        with tf.variable_scope(self.scope):
            # present start symbol
            _, _, prev_state = self.cell(self.start_symbol, state=None)
            self.save_state(prev_state, 0, self.max_length)

            #zeros = np.zeros(self.cell.input_dim, dtype=np.float32)

            tf.get_variable_scope().reuse_variables()
            for seq_length in range(1, self.max_length + 1):
                progress(seq_length / float(self.max_length))

                input_ = tf.placeholder(tf.float32, [self.cell.input_dim],
                                        name='input_%s' % seq_length)
                true_output = tf.placeholder(tf.float32, [self.cell.output_dim],
                                             name='true_output_%s' % seq_length)

                self.inputs.append(input_)
                self.true_outputs.append(true_output)

                # present inputs
                output, output_logit, prev_state = self.cell(input_, prev_state)
                self.save_state(prev_state, seq_length, self.max_length)

                self.output_logits.append(output_logit)
                self.outputs.append(output)
                self.prev_states[seq_length] = prev_state

            print(" [*] Constructing mask")
            out_stacked = tf.stack(self.output_logits)
            true_stacked = tf.stack(self.true_outputs)

            self.mask = tf.sign(tf.reduce_max(tf.abs(true_stacked), reduction_indices=1))
            # So *very* hacky, but it works
            self.mask_full = tf.transpose(tf.reshape(tf.tile(self.mask, tf.constant([20])), [20, 3]))
            answer_stacked = tf.multiply(out_stacked, self.mask_full)
            self.answer = tf.unstack(answer_stacked)
    
            print(" [*] Building a loss model for max seq_length %s" % seq_length)


            self.loss = sequence_loss(
                logits=self.answer,
                targets=self.true_outputs,
                weights=[1] * self.max_length,#seq_length,
                average_across_timesteps=False,
                average_across_batch=False,
                softmax_loss_function=sp_loss_function)


            if not self.params:
                self.params = tf.trainable_variables()

            # grads, norm = tf.clip_by_global_norm(
            #                  tf.gradients(loss, self.params), 5)

            print("  [*] Generating gradients")
            grads = []
            for grad in tf.gradients(self.loss, self.params):
                if grad is not None:
                    grads.append(tf.clip_by_value(grad,
                                                  self.min_grad,
                                                  self.max_grad))
                else:
                    grads.append(grad)

            print("  [*] Building optimiser")
            self.grads = grads
            opt = tf.train.RMSPropOptimizer(self.lr,
                                            decay=self.decay,
                                            momentum=self.momentum)

            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                self.optim = opt.apply_gradients(
                    zip(grads, self.params),
                    global_step=self.global_step)

        model_vars = \
            [v for v in tf.global_variables() if v.name.startswith(self.scope)]
        self.saver = tf.train.Saver(model_vars)
        print(" [*] Build a NTM model finished")

    def get_output_logits(self):
        return self.output_logits

    #def get_prediction(self, seq_length):
     #   labels_a, labels_b = tf.self.output_logits(labels, 2, 0)

    def get_loss(self):
        """
        if not seq_length in self.outputs:
            self.get_outputs(seq_length)

        """
        return self.loss

    def get_output_states(self, seq_length):
        zeros = np.zeros(self.cell.input_dim, dtype=np.float32)

        if not seq_length in self.output_states:
            with tf.variable_scope(self.scope):
                tf.get_variable_scope().reuse_variables()

                outputs, output_logits = [], []
                state = self.prev_states[seq_length]

                for _ in range(seq_length):
                    output, output_logit, state = self.cell(zeros, state)
                    self.save_state(state, seq_length, is_output=True)
                    outputs.append(output)
                    output_logits.append(output_logit)
                self.outputs[seq_length] = outputs
                self.output_logits[seq_length] = output_logits
        return self.output_states[seq_length]

    #@property
    #def loss(self):
    #    return self.losses[self.cell.depth]

    #@property
    #def optim(self):
    #    return self.optims[self.cell.depth]

    def save_state(self, state, from_, to=None, is_output=False):
        if is_output:
            state_to_add = self.output_states
        else:
            state_to_add = self.input_states

        if to:
            for idx in range(from_, to + 1):
                state_to_add[idx].append(state)
        else:
            state_to_add[from_].append(state)

    def save(self, checkpoint_dir, task_name, step):
        task_dir = os.path.join(checkpoint_dir, "%s_%s" % (task_name, self.max_length))
        file_name = "%s_%s.model" % (self.scope, task_name)

        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

        self.saver.save(
            self.sess,
            os.path.join(task_dir, file_name),
            global_step=step.astype(int))

    def load(self, checkpoint_dir, task_name, strict=True):
        print(" [*] Reading checkpoints...")

        task_dir = "%s_%s" % (task_name, self._max_length)
        checkpoint_dir = os.path.join(checkpoint_dir, task_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        else:
            if strict:
                raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)
            else:
                print(' [!] Training, but previous training data %s not found' % checkpoint_dir)
