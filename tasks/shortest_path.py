import os
import time
import dataset
import numpy as np
import tensorflow as tf
import random
import csv
import utils
import pickle

from utils import pprint

print_interval = 1

def train(ntm, config, sess):
    np.set_printoptions(threshold=np.nan)
    # Check all relevant directories are present
    if not os.path.isdir(config.checkpoint_dir):
        print(" [!] Directory %s not found. Creating." % config.checkpoint_dir)
        os.makedirs(config.checkpoint_dir)

    if not os.path.isdir(config.dataset_dir):
        raise Exception(" [!] Directory dataset_files not found")

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
        train_writer = tf.summary.FileWriterCache.get(config.checkpoint_dir)
    else:
        train_writer = tf.summary.FileWriter(config.checkpoint_dir)

    print(" [*] Loading dataset...")
    # Load the dataset from the file and get training sets
    dset = dataset.Dataset(config.graph_size, config.dataset_dir)
    input_set, target_set, lengths, dist, unencoded = dset.get_training_data(config.train_set_size)
    print(dist)

    # Start training
    print(" [*] Starting training")
    start_time = time.time()
    for idx, _ in enumerate(input_set):
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
        _, cost, step, summary, answer = sess.run([ntm.optim,
                                           ntm.get_loss(),
                                           ntm.global_step,
                                           ntm.merged,
                                           ntm.answer_train], feed_dict=feed_dict)

        # Save stuff, print stuff
        if (idx+1) % 10000 == 0:
            print(" [*] Saving checkpoint")
            ntm.save(config.checkpoint_dir, config.task, step)

        if idx % print_interval == 0:
            print(
                "[%d:%d] %.10f (%.1fs)"
                % (idx, lengths[idx], cost, time.time() - start_time))
            #utils.pprint(states[-1]['M'])
            train_writer.add_summary(summary, step)

        if cost < 0.01:
            print("True:")
            print(np.array(unencoded[idx][3]))
            print("Answer:")
            print(np.array(answer))


    print(dist)
    # Cleanup
    train_writer.close
    train_writer.flush

#TODO fix up this run section
def run(ntm, config, sess):
    np.set_printoptions(threshold=np.nan)
    if not os.path.isdir(config.dataset_dir):
        raise Exception(" [!] Directory dataset_files not found")

    # Delimiter flag for start and end
    start_symbol = np.ones([config.input_dim], dtype=np.float32)
    end_symbol = np.ones([config.input_dim], dtype=np.float32)
    end_symbol[0] = 0

    # Load the dataset from the file and get training sets
    dset = dataset.Dataset(config.graph_size, config.dataset_dir)
    input_set, target_set, lengths, dist, unencoded = dset.get_validation_data(config.val_set_size)
    print(dist)

    # 1) Completely wrong
    abs_error = []#[0] * config.graph_size - 1
    # 2) Where they were wrong
    pos_error = []
    # 3) The actual paths
    paths_output = []

    lengths_count = [0] * (config.graph_size - 1)
    test_results = [abs_error, pos_error, paths_output]


    # Run the actual test
    for idx, _ in enumerate(input_set):
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

        errors, step = sess.run([ntm.error, ntm.global_step], feed_dict=feed_dict)

        error_count_edges, stripped_mistake, final_output = errors

        length = lengths[idx]
        lengths_count[length-1] += 1

        # Record stuff
        if error_count_edges == 0:
            abs_error.append(0)
        else:
            abs_error.append(1)

        final_output = list(map(list, list(final_output)))
        paths_output.append(final_output)
        pos_error.append(stripped_mistake)

        print("[{}:{}] error:{}".format(idx, length, error_count_edges))
        #print("    {}".format(unencoded[idx][0].edge_list))


    with open('{}/results.pkl'.format(config.checkpoint_dir), 'wb') as output:
        pickle.dump(test_results, output, pickle.HIGHEST_PROTOCOL)

    print(np.sum(pos_error, 0))

    sum_pos = np.sum(pos_error, 0)
    percent_pos = []
    running_sum = 0
    for i, lcount in enumerate(lengths_count):
        percent_pos.append(sum_pos[i] / (config.val_set_size - running_sum))
        running_sum += lcount

    print(percent_pos)

    with open(config.checkpoint_dir + '/error.csv', 'w', newline='') as csvfile:
        fieldnames = ['complete', 'first', 'second', 'third', 'fourth', 'fifth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({'complete':np.sum(abs_error)/config.val_set_size,
                         'first':percent_pos[0],
                         'second':percent_pos[1],
                         'third':percent_pos[2],
                         'fourth':percent_pos[3],
                         'fifth':percent_pos[4]})
