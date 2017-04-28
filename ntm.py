from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.contrib.legacy_seq2seq import sequence_loss

import ntm_cell

import os
from utils import progress

def lazy_property(function):
    """
    Used to annotate functions that we only want computer once. Cached results
    are then used on subsequent calls.
    """
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

def softmax_loss_function(labels, inputs):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=inputs)

def sp_loss_function(labels, logits):

    inputs_a, inputs_b = tf.split(logits, 2, 0)
    labels_a, labels_b = tf.split(labels, 2, 0)

    loss_a = tf.nn.softmax_cross_entropy_with_logits(labels=labels_a, logits=inputs_a)
    loss_b = tf.nn.softmax_cross_entropy_with_logits(labels=labels_b, logits=inputs_b)

    return loss_a + loss_b



class NTM(object):
    def __init__(self, cell, sess,
                 min_length, max_length, min_size, max_size,
                 min_grad=-10, max_grad=+10,
                 lr=1e-4, momentum=0.9, decay=0.95, beta=0.01,
                 scope="NTM", forward_only=False):
        """Create a neural turing machine specified by NTMCell "cell".

        Args:
            cell: An instantce of NTMCell.
            sess: A TensorFlow session.
            min_length: Minimum length of input sequence.
            max_length: Maximum length of input sequence for training.
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
        self.beta = beta

        self.min_grad = min_grad
        self.max_grad = max_grad
        self.min_length = min_length
        self.max_length = max_length
        self._max_length = max_length
        self.min_size = min_size
        self.max_size = max_size

        self.inputs = []
        self.outputs_train = []
        self.true_outputs = []
        self.answer = []

        self.outputs_test = []
        self.answer_test = []


        self.prev_states = {}
        self.input_states = defaultdict(list)
        self.output_states = defaultdict(list)
        self.test_states = []
        self.train_states = []

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

    def build_model(self, forward_only, is_structured=True):
        print(" [*] Building a NTM model")

        with tf.variable_scope(self.scope):
            # present start symbol
            with tf.name_scope("start_step"):
                _, prev_state = self.cell(self.start_symbol, state=None)

            self.test_states.append(prev_state)
            self.train_states.append(prev_state)

            start_answer = self.max_length - self.max_size + 1
            prefix = tf.constant([1, 1], np.float32)

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
                if is_structured:
                    if seq_length > start_answer:
                        with tf.name_scope("step_answer"):
                            if seq_length == start_answer + 1:
                                prev_state_train = prev_state
                                prev_state_test = prev_state

                            with tf.name_scope("train"):
                                # For training, use target
                                s_input = tf.concat([prefix, self.true_outputs[-2]], 0)

                                with tf.name_scope("step"):
                                    output_train, prev_state_train = self.cell(s_input,
                                                                               prev_state_train)
                                    self.outputs_train.append(output_train)
                                    self.train_states.append(prev_state_train)

                            with tf.name_scope("test"):
                                with tf.name_scope("converter"):
                                    # For testing, use previous
                                    # TODO CHECK THIS ACTUALLY WORKS AS INTENDED
                                    out_a, out_b = tf.split(self.outputs_test[-1], 2)
                                    pred_a = tf.arg_max(tf.nn.softmax(out_a, name="test_soft_a"),
                                                        0,
                                                        "test_argm_a")
                                    pred_b = tf.arg_max(tf.nn.softmax(out_b, name="test_soft_b"),
                                                        0,
                                                        "test_argm_b")
                                    in_a = tf.one_hot(9 - pred_a, 10, name="one_hot_a")
                                    in_b = tf.one_hot(9 - pred_b, 10, name="one_hot_b")
                                    s_input = tf.concat([prefix, in_a, in_b], 0, name="input_next")

                                #self.test_predictions.append([pred_a, pred_b])
                                with tf.name_scope("step"):
                                    output_test, prev_state_test = self.cell(s_input,
                                                                             prev_state_test)
                                    self.outputs_test.append(output_test)
                                    self.test_states.append(prev_state_test)


                    else:
                        with tf.name_scope("step"):
                            # Everything before the answer phase is the same for both
                            output, prev_state = self.cell(input_, prev_state)
                            self.outputs_train.append(output)
                            self.outputs_test.append(output)
                            self.test_states.append(prev_state)
                            self.train_states.append(prev_state)

                else:
                    with tf.name_scope("step"):
                        # Without any structured learning
                        output, prev_state = self.cell(input_, prev_state)
                        self.outputs_train.append(output)
                        self.train_states.append(prev_state)


            print(" [*] Constructing mask")
            with tf.name_scope("answer_filter"):
                # So *very* hacky, but it works
                true_stacked = tf.stack(self.true_outputs)
                mask = tf.sign(tf.reduce_max(tf.abs(true_stacked), reduction_indices=1))
                self.mask_full = tf.transpose(tf.reshape(tf.tile(mask, tf.constant([20])),
                                                         [20, self.max_length]))

                # Train answer
                self.out_stacked = tf.stack(self.outputs_train)
                answer_stacked = tf.multiply(self.out_stacked, self.mask_full)
                self.answer = tf.unstack(answer_stacked)

                # Test answer
                out_test_stacked = tf.stack(self.outputs_test)
                answer_test_stacked = tf.multiply(out_test_stacked, self.mask_full)
                self.answer_test = tf.unstack(answer_test_stacked)

            print(" [*] Building a loss model for max seq_length %s" % seq_length)


            self.loss = sequence_loss(
                logits=self.answer_test,
                targets=self.true_outputs,
                weights=[1] * self.max_length,#seq_length,
                average_across_timesteps=False,
                average_across_batch=False,
                softmax_loss_function=sp_loss_function)

            with tf.variable_scope("Linear"):
                output_w = tf.get_variable("output_w")

            regulariser = tf.nn.l2_loss(output_w)
            self.loss = tf.reduce_mean(self.loss + self.beta * regulariser)

            tf.summary.scalar('loss', self.loss)

            if not self.params:
                self.params = tf.trainable_variables()

            # grads, norm = tf.clip_by_global_norm(
            #                  tf.gradients(loss, self.params), 5)

            print(" [*] Generating gradients (slow)")
            with tf.name_scope("gradients"):
                grads = []
                for grad in tf.gradients(self.loss, self.params):
                    if grad is not None:
                        grads.append(tf.clip_by_value(grad,
                                                    self.min_grad,
                                                    self.max_grad))
                    else:
                        grads.append(grad)

            print(" [*] Building optimiser")
            self.grads = grads
            opt = tf.train.RMSPropOptimizer(self.lr,
                                            decay=self.decay,
                                            momentum=self.momentum)

            # Reuse false because I removed the loop from the original code,
            # therefore there's never anything to reuse
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                self.optim = opt.apply_gradients(
                    zip(grads, self.params),
                    global_step=self.global_step)


        model_vars = \
            [v for v in tf.global_variables() if v.name.startswith(self.scope)]
        self.saver = tf.train.Saver(model_vars)
        self.merged = tf.summary.scalar('loss', self.loss)

        print(" [*] Build a NTM model finished")


    @lazy_property
    def answer_train(self):
        output_a, output_b = tf.split(self.answer, 2, 1)

        pred_a = tf.argmax(tf.nn.softmax(output_a), 1)
        pred_b = tf.argmax(tf.nn.softmax(output_b), 1)

        return pred_a, pred_b

    @lazy_property
    def error(self):
        pred_a, pred_b = tf.split(self.answer_test, 2, 1)
        target_a, target_b = tf.split(self.true_outputs, 2, 1)

        self.pred_argmax_a = tf.argmax(tf.nn.softmax(pred_a), 1)
        self.pred_argmax_b = tf.argmax(tf.nn.softmax(pred_b), 1)

        self.target_argmax_a = tf.argmax(target_a, 1)
        self.target_argmax_b = tf.argmax(target_b, 1)

        mistake_a = tf.not_equal(self.pred_argmax_a, self.target_argmax_a)
        mistake_b = tf.not_equal(self.pred_argmax_b, self.target_argmax_b)

        mistake = tf.logical_or(mistake_a, mistake_b)
        error_rate = tf.reduce_sum(tf.cast(mistake, tf.float32))
        error_rate /= tf.reduce_sum(tf.reduce_max(self.mask_full, reduction_indices=1))
        #tf.summary.histogram("error", error_rate)

        #return tf.summary.histogram("error", error_rate)
        return error_rate


    def get_loss(self):
        """
        if not seq_length in self.outputs:
            self.get_outputs(seq_length)

        """
        return self.loss

    def save(self, checkpoint_dir, task_name, step):
        file_name = "%s_%s.model" % (self.scope, task_name)

        self.saver.save(
            self.sess,
            os.path.join(checkpoint_dir, file_name),
            global_step=step.astype(int))

    def load(self, checkpoint_dir, task_name, strict=True):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        else:
            if strict:
                raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)
            else:
                print(' [!] Training, but previous training data %s not found' % checkpoint_dir)

        