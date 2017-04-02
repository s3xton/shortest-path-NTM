import os
import time
import dataset
import numpy as np
import tensorflow as tf
import random
import csv
import utils

from utils import pprint

print_interval = 5

def train(ntm, config, sess):
    # Check all relevant directories are present
    if not os.path.isdir(config.checkpoint_dir):
        raise Exception(" [!] Directory %s not found" % config.checkpoint_dir)

    if not os.path.isdir("dataset_files"):
        raise Exception(" [!] Directory dataset_files not found")

    task_dir = "%s_%s_%s" % (config.task, config.min_size, config.max_size)
    summary_dir = os.path.join(config.summary_dir, task_dir)

    # Delimiter flag for start and end
    start_symbol = np.ones([config.input_dim], dtype=np.float32)
    end_symbol = np.ones([config.input_dim], dtype=np.float32)
    end_symbol[0] = 0

    print(" [*] Initialize all variables")
    tf.global_variables_initializer().run()
    print(" [*] Initialization finished")

    # Load the checkpoint file if necessary
    if config.continue_train is True:
        ntm.load(config.checkpoint_dir, config.task, strict=config.continue_train is True)
        print(" [*] Loading summaries...")
        train_writer = tf.summary.FileWriterCache.get(summary_dir)
    else:
        train_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Load the dataset from the file and get training sets
    dset = dataset.Dataset(config.dataset_filename)
    input_set, target_set = dset.get_training_data(config.train_set_size)

    # Start training
    start_time = time.time()
    idx = 0
    while idx < config.epoch:
        # Get inputs
        inp_seq = input_set[idx]
        target_seq = target_set[idx]

        # Generate feed dictionary
        feed_dict = {input_: vec for vec, input_ in zip(inp_seq, ntm.inputs)}
        feed_dict.update(
            {true_output: vec for vec, true_output in zip(target_seq, ntm.true_outputs)}
        )
        feed_dict.update({
            ntm.start_symbol: start_symbol,
            ntm.end_symbol: end_symbol
        })

        # Run the NTM
        _, cost, step, summary, states = sess.run([ntm.optim,
                                                   ntm.get_loss(),
                                                   ntm.global_step,
                                                   ntm.merged,
                                                   ntm.train_states], feed_dict=feed_dict)

        # Save stuff, print stuff
        if idx % 1000 == 0:
            ntm.save(config.checkpoint_dir, config.task, step)
        if idx % print_interval == 0:
            print(
                "[%5d] %.10f (%.1fs)"
                % (step, cost, time.time() - start_time))
            #utils.pprint(states[-1]['M'])
            train_writer.add_summary(summary, step)

    # Cleanup
    train_writer.close
    train_writer.flush

#TODO fix up this run section
def run(ntm, config, sess, graph_size):

    if not os.path.isdir("dataset_files"):
        raise Exception(" [!] Directory dataset_files not found")

    task_dir = "%s_%s_%s" % (config.task, config.min_size, config.max_size)
    summary_dir = os.path.join(config.summary_dir, task_dir)

    # Delimiter flag for start and end
    start_symbol = np.ones([config.input_dim], dtype=np.float32)
    end_symbol = np.ones([config.input_dim], dtype=np.float32)
    end_symbol[0] = 0

    # Load the dataset from the file and get training sets
    dset = dataset.Dataset(config.dataset_filename)
    input_set, target_set = dset.get_validation_data(config.train_set_size)

    error_sum = 0.0
    for idx in range(config.val_set_size):
        # Get inputs
        inp_seq = input_set[idx]
        target_seq = target_set[idx]

        feed_dict = {input_: vec for vec, input_ in zip(inp_seq, ntm.inputs)}
        feed_dict.update(
            {true_output: vec for vec, true_output in zip(target_seq, ntm.true_outputs)}
        )

        feed_dict.update({
            ntm.start_symbol: start_symbol,
            ntm.end_symbol: end_symbol
        })

        error = sess.run(ntm.error, feed_dict=feed_dict)

        error_sum += error

        if idx % print_interval == 0:
            print(
                "[%5d] %.5f"
                % (idx, error_sum/(idx +1)))
            print(error)


        #train_writer.add_summary(error, idx)
    final_error = error_sum/config.test_set_size
    print("Final error rate: %.5f" % final_error)

    with open(summary_dir + '/error.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow("%.5f" % final_error)
