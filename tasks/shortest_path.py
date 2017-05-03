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
    abs_error = {}
    # 2) Edges wrong in the path
    edges_error = {}
    # 3) Nodes wrong in the path
    nodes_error = {}
    # 4) Valid paths given the graph
    valid_paths = {}
    # 5) Valid edges given the graph

    test_results = [abs_error, edges_error, nodes_error, valid_paths]

    for i in range(1, config.graph_size+1):
        abs_error[i] = []
        edges_error[i] = []
        nodes_error[i] = []
        valid_paths[i] = []

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

        error_count_edges, error_count_nodes, final_output = errors

        length = lengths[idx]

        # Record stuff
        if error_count_edges == 0:
            abs_error[length].append(0)
        else:
            abs_error[length].append(1)
        edges_error[length].append(error_count_edges)
        nodes_error[length].append(error_count_nodes)

        print("[{}:{}] edge:{} nodes:{} output:{}".format(idx, length, error_count_edges, error_count_nodes, final_output))


    with open('{}/results.pkl'.format(config.checkpoint_dir), 'wb') as output:
        pickle.dump(test_results, output, pickle.HIGHEST_PROTOCOL)


    # STATISTICS
    mean = {"abs":[], "edge":[], "node":[]}
    sd = {"abs":[], "edge":[], "node":[]}
    se = {"abs":[], "edge":[], "node":[]}

    abs_overall = []
    edge_overall = []
    node_overall = []

    for i in range(1, config.graph_size+1):
        # Mean
        mean["abs"].append(np.mean(np.array(abs_error[i])))
        mean["edge"].append(np.mean(np.array(edges_error[i])))
        mean["node"].append(np.mean(np.array(nodes_error[i])))

        # Standard deviation
        sd["abs"].append(np.std(np.array(abs_error[i])))
        sd["edge"].append(np.std(np.array(edges_error[i])))
        sd["node"].append(np.std(np.array(nodes_error[i])))

        # Standard error

        # Appending stuff together
        abs_overall.append(abs_error[i])
        edge_overall.append(edges_error[i])
        node_overall.append(nodes_error[i])


    mean_abs_overall = np.mean(abs_overall)
    sd_abs_overall = np.std(abs_overall)

    mean_edge_overall = np.mean(edge_overall)
    sd_edge_overall = np.std(edge_overall)

    mean_node_overal = np.mean(node_overall)
    sd_node_overall = np.std(node_overall)

    with open(config.checkpoint_dir + '/error_abs.csv', 'w', newline='') as csvfile:
        fieldnames = ['mean', 'SD', 'SE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for i in range(config.graph_size):
            writer.writerow({'mean':mean['abs'][i], 'SD':sd['abs'], 'SE':0})

        writer.writerow({'mean':mean_abs_overall, 'SD':sd_abs_overall, 'SE':0})

