from __future__ import absolute_import

import importlib
import tensorflow as tf
from ntm_cell import NTMCell
from ntm import NTM

from utils import pp

flags = tf.app.flags
flags.DEFINE_string("task", "shortest_path", "Task to run [copy, recall, shortest_path]")
flags.DEFINE_integer("epoch", 20000, "Epoch to train [100000]")
flags.DEFINE_integer("input_dim", 22, "Dimension of input [10]")
flags.DEFINE_integer("output_dim", 20, "Dimension of output [10]")
#flags.DEFINE_integer("min_length", 1, "Minimum length of input sequence [1]")
#flags.DEFINE_integer("max_length", 10, "Maximum length of output sequence [10]")
flags.DEFINE_integer("controller_layer_size", 1, "The size of LSTM controller [1]")
flags.DEFINE_integer("controller_dim", 100, "Dimension of LSTM controller [100]")
flags.DEFINE_integer("write_head_size", 1, "The number of write head [1]")
flags.DEFINE_integer("read_head_size", 1, "The number of read head [1]")
flags.DEFINE_integer("test_max_length", 120, "Maximum length of output sequence [120]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("summary_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("continue_train", None, "True to continue training from saved checkpoint. False for restarting. None for automatic [None]")

# My config vars
flags.DEFINE_integer("min_size", 4, "Minimum graph size")
flags.DEFINE_integer("max_size", 4, "Maximum graph size")
flags.DEFINE_integer("plan_length", 10, "Length of planning phase")
flags.DEFINE_integer("test_set_size", 2000, "Number of runs to perform when testing accuracy")
FLAGS = flags.FLAGS


def create_ntm(config, sess, **ntm_args):
    cell = NTMCell(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        controller_layer_size=config.controller_layer_size,
        controller_dim=config.controller_dim,
        write_head_size=config.write_head_size,
        read_head_size=config.read_head_size)
    scope = ntm_args.pop('scope', 'NTM-%s' % config.task)

    # Description + query + plan + answer
    min_length = (config.min_size - 1) + 1 + config.plan_length + (config.min_size - 1)
    max_length = int(((config.max_size * (config.max_size - 1)/2) +
                  1 + config.plan_length + (config.max_size - 1)))
    ntm = NTM(
        cell, sess, min_length, max_length, config.min_size, config.max_size,
        test_max_length=config.test_max_length, scope=scope, **ntm_args)
    return cell, ntm


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    #with tf.device('/gpu:0'), tf.Session() as sess:
    with tf.Session() as sess:
        try:
            task = importlib.import_module('tasks.%s' % FLAGS.task)
        except ImportError:
            print("task '%s' does not have implementation" % FLAGS.task)
            raise

        if FLAGS.is_train:
            cell, ntm = create_ntm(FLAGS, sess)
            task.train(ntm, FLAGS, sess)
        else:
            cell, ntm = create_ntm(FLAGS, sess, forward_only=True)

        #ntm.load(FLAGS.checkpoint_dir, FLAGS.task)

        if FLAGS.task == 'copy':
            task.run(ntm, int(FLAGS.test_max_length * 1 / 3), sess)
            print
            task.run(ntm, int(FLAGS.test_max_length * 2 / 3), sess)
            print
            task.run(ntm, int(FLAGS.test_max_length * 3 / 3), sess)
        #else:
         #   task.run(ntm, int(FLAGS.test_max_length), sess)


if __name__ == '__main__':
    tf.app.run()
