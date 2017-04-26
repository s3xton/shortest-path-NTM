import os
import time
import dataset
import numpy as np
import tensorflow as tf
import random
import csv
import utils

from utils import pprint

print_interval = 1

def train(ntm, config, sess):
    # Check all relevant directories are present
    if not os.path.isdir(config.checkpoint_dir):
        print(" [!] Directory %s not found. Creating." % config.checkpoint_dir)
        os.makedirs(config.checkpoint_dir)

    if not os.path.isdir("dataset_files"):
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
        train_writer = tf.summary.FileWriter(config.checkpoint_dir, sess.graph)

    print(" [*] Loading dataset...")
    # Load the dataset from the file and get training sets
    dset = dataset.Dataset(config.graph_size, config.dataset_dir)
    input_set, target_set, lengths, dist, _ = dset.get_training_data(config.train_set_size)
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
        _, cost, step, summary = sess.run([ntm.optim,
                                           ntm.get_loss(),
                                           ntm.global_step,
                                           ntm.merged], feed_dict=feed_dict)

        # Save stuff, print stuff
        if (idx+1) % 1000 == 0:
            print(" [*] Saving checkpoint")
            ntm.save(config.checkpoint_dir, config.task, step)

        if idx % print_interval == 0:
            print(
                "[%d:%d] %.10f (%.1fs)"
                % (idx, lengths[idx], cost, time.time() - start_time))
            #utils.pprint(states[-1]['M'])
            train_writer.add_summary(summary, step)
    print(dist)
    # Cleanup
    train_writer.close
    train_writer.flush

#TODO fix up this run section
def run(ntm, config, sess):
    np.set_printoptions(threshold=np.nan)
    if not os.path.isdir("dataset_files"):
        raise Exception(" [!] Directory dataset_files not found")

    # Delimiter flag for start and end
    start_symbol = np.ones([config.input_dim], dtype=np.float32)
    end_symbol = np.ones([config.input_dim], dtype=np.float32)
    end_symbol[0] = 0

    # Load the dataset from the file and get training sets
    dset = dataset.Dataset(config.graph_size, config.dataset_dir)
    input_set, target_set, lengths, dist, unencoded = dset.get_validation_data(config.val_set_size)
    print(dist)
    error_sum = 0.0
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

        error, step, p_a, p_b, t_a, t_b, true_out = sess.run([ntm.error, ntm.global_step,
                                                    ntm.pred_argmax_a,
                                                    ntm.pred_argmax_b,
                                                    ntm.target_argmax_a,
                                                    ntm.target_argmax_b,
                                                    ntm.true_outputs], feed_dict=feed_dict)

        error_sum += error

        #if idx % print_interval == 0:
        print(
            "[%d:%d] %.5f %d"
            % (idx, lengths[idx], error_sum/(idx +1), step))
        print(error)
        
        print("pred:")
        print(np.array(p_a))
        print(np.array(p_b))
        print("target:")
        print(np.array(t_a))
        print(np.array(t_b))
        '''
        print("original:")
        print(unencoded[idx][0].edge_list)
        print(unencoded[idx][3])
        print("encoded target:")
        print(np.array(target_seq))
        print("true outputs:")
        print(np.array(true_out))
        '''

    final_error = error_sum/config.test_set_size
    print("Final error rate: %.5f" % final_error)
    print(dist)
    with open(config.checkpoint_dir + '/error.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow("%.5f" % final_error)
