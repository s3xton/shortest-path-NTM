from __future__ import absolute_import

import importlib
import tensorflow as tf
import numpy as np
import random
import csv
import os
from ntm_cell import NTMCell
from ntm import NTM

from utils import pp
from utils import sample_loguniform

flags = tf.app.flags
flags.DEFINE_string("task", "shortest_path", "Task to run [copy, recall, shortest_path]")
flags.DEFINE_integer("epoch", 20000, "Epoch to train [100000]")
flags.DEFINE_integer("input_dim", 22, "Dimension of input [10]")
flags.DEFINE_integer("output_dim", 20, "Dimension of output [10]")
flags.DEFINE_integer("controller_layer_size", 1, "The size of LSTM controller [1]")
flags.DEFINE_integer("controller_dim", 100, "Dimension of LSTM controller [100]")
flags.DEFINE_integer("write_head_size", 1, "The number of write head [1]")
flags.DEFINE_integer("read_head_size", 1, "The number of read head [1]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("summary_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_final_test", False, "True for final test, False for validation testing [False]")
flags.DEFINE_boolean("is_test", False, "True for validation testing, False for training [False]")
flags.DEFINE_boolean("continue_train", None, "True to continue training from saved checkpoint. False for restarting. None for automatic [None]")

# My config vars
flags.DEFINE_integer("min_size", 10, "Minimum graph size")
flags.DEFINE_integer("max_size", 10, "Maximum graph size")
flags.DEFINE_integer("graph_size", 6, "The number of nodes in the graphs being input")
flags.DEFINE_integer("plan_length", 10, "Length of planning phase")
flags.DEFINE_integer("train_set_size", 1000, "Number of runs to perform when training")
flags.DEFINE_integer("val_set_size", 1000, "Number of runs to perform when training")
flags.DEFINE_integer("test_set_size", 0, "Number of runs to perform when testing accuracy")
flags.DEFINE_boolean("is_LSTM_mode", False, "Toggle for using LSTM mode (memory off)")
flags.DEFINE_string("dataset_dir", "dataset_files", "Directory from which to load the dataset")
flags.DEFINE_float("beta", 0.01, "The beta value used in L2 regularisation")
flags.DEFINE_boolean("rand_hyper", False, "Toggle for hyperparameter randomisation")
FLAGS = flags.FLAGS

def generate_hyperparams(config):
    l2_beta = random.choice([0, sample_loguniform(0.0000001, 0.0001)])
    lr = sample_loguniform(0.00001, 0.001)
    momentum = sample_loguniform(0.001, 1)
    decay = sample_loguniform(0.001, 1)
    controller_dim = np.around(sample_loguniform(100, 600))
    controller_layers = random.choice([1, 2, 3])
    mem_size = random.choice([128, 256])

    # Crash protection
    if controller_layers == 3:
        mem_size = 128

    hyper_params = {'l2':l2_beta,
                    'lr':lr,
                    'momentum':momentum,
                    'decay':decay,
                    'c_dim':controller_dim,
                    'c_layer':controller_layers,
                    'mem_size': mem_size}

    if not os.path.isdir(config.checkpoint_dir):
        print(" [!] Directory %s not found. Creating." % config.checkpoint_dir)
        os.makedirs(config.checkpoint_dir)

    with open(config.checkpoint_dir + '/config.csv', 'w', newline='') as csvfile:
        fieldnames = ['l2', 'lr', 'momentum', 'decay', 'c_dim', 'c_layer', 'mem_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow(hyper_params)

    return hyper_params

def load_hyperparamters(config):
    read_list = []
    with open(config.checkpoint_dir + '/config.csv', newline='') as file: 
        reader = csv.reader(file, delimiter=",", quotechar="|") 
        for row in reader:
            for elem in row:
                read_list.append(float(elem))

    hyper_params = {'l2':read_list[0],
                    'lr':read_list[1],
                    'momentum':read_list[2],
                    'decay':read_list[3],
                    'c_dim':int(read_list[4]),
                    'c_layer':int(read_list[5]),
                    'mem_size': int(read_list[6])}
    return hyper_params


def create_ntm(config, sess, **ntm_args):
    if config.rand_hyper:
        hyper_params = {}
        if config.is_test:
            hyper_params = load_hyperparamters(config)
        else:
            hyper_params = generate_hyperparams(config)
        print(" [*] Hyperparameters: {}".format(hyper_params))
        cell = NTMCell(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            controller_layer_size=hyper_params["c_layer"],
            controller_dim=hyper_params["c_dim"],
            mem_size=hyper_params["mem_size"],
            write_head_size=config.write_head_size,
            read_head_size=config.read_head_size,
            is_LSTM_mode=config.is_LSTM_mode)
        scope = ntm_args.pop('scope', 'NTM-%s' % config.task)

        # Description + query + plan + answer
        min_length = (config.min_size - 1) + 1 + config.plan_length + (config.min_size - 1)
        max_length = int(((config.max_size * (config.max_size - 1)/2) +
                    1 + config.plan_length + (config.max_size - 1)))
        ntm = NTM(
            cell, sess, min_length, max_length, config.min_size, config.max_size,
            scope=scope, **ntm_args,
            lr=hyper_params["lr"], momentum=hyper_params["momentum"],
            decay=hyper_params["decay"], beta=hyper_params["l2"])

    else:
        cell = NTMCell(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            controller_layer_size=config.controller_layer_size,
            controller_dim=config.controller_dim,
            write_head_size=config.write_head_size,
            read_head_size=config.read_head_size,
            is_LSTM_mode=config.is_LSTM_mode)
        scope = ntm_args.pop('scope', 'NTM-%s' % config.task)

        # Description + query + plan + answer
        min_length = (config.min_size - 1) + 1 + config.plan_length + (config.min_size - 1)
        max_length = int(((config.max_size * (config.max_size - 1)/2) +
                    1 + config.plan_length + (config.max_size - 1)))
        ntm = NTM(
            cell, sess, min_length, max_length, config.min_size, config.max_size,
            scope=scope, **ntm_args)
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

        cell, ntm = create_ntm(FLAGS, sess)
        if FLAGS.is_train:
            task.train(ntm, FLAGS, sess)

        if FLAGS.is_test:
            ntm.load(FLAGS.checkpoint_dir, FLAGS.task)
            task.run(ntm, FLAGS, sess)


if __name__ == '__main__':
    tf.app.run()
