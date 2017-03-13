import os
import time
import numpy as np
import tensorflow as tf
import random
import graph_util
import csv

from utils import pprint

print_interval = 5

def train(ntm, config, sess):
    if not os.path.isdir(config.checkpoint_dir):
        raise Exception(" [!] Directory %s not found" % config.checkpoint_dir)

    task_dir = "%s_%s_%s" % (config.task, config.min_size, config.max_size)
    summary_dir = os.path.join(config.summary_dir, task_dir)


    # delimiter flag for start and end
    start_symbol = np.zeros([config.input_dim], dtype=np.float32)
    start_symbol[0] = 1
    end_symbol = np.zeros([config.input_dim], dtype=np.float32)
    end_symbol[1] = 1

    print(" [*] Initialize all variables")
    tf.global_variables_initializer().run()
    print(" [*] Initialization finished")

    if config.continue_train is True:
        ntm.load(config.checkpoint_dir, config.task, strict=config.continue_train is True)
        print(" [*] Loading summaries...")
        train_writer = tf.summary.FileWriterCache.get(summary_dir)
    else:
        train_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    start_time = time.time()
    idx = 0
    while idx < config.epoch:
        graph_size = random.randint(config.min_size, config.max_size)

        inp_seq, target_seq, edges = graph_util.gen_single(graph_size,
                                                           config.plan_length,
                                                           config.max_size)


        feed_dict = {input_: vec for vec, input_ in zip(inp_seq, ntm.inputs)}
        feed_dict.update(
            {true_output: vec for vec, true_output in zip(target_seq, ntm.true_outputs)}
        )

        feed_dict.update({
            ntm.start_symbol: start_symbol,
            ntm.end_symbol: end_symbol
        })

        _, cost, step, summary = sess.run([ntm.optim,
                                           ntm.get_loss(),
                                           ntm.global_step,
                                           ntm.merged], feed_dict=feed_dict)

        idx = step

        if idx % 100 == 0:
            ntm.save(config.checkpoint_dir, config.task, step)

        if idx % print_interval == 0:
            print(
                "[%5d] %2d: %.10f (%.1fs)"
                % (step, edges, cost, time.time() - start_time))

            train_writer.add_summary(summary, step)



    error_sum = 0.0
    
    for idx in range(config.test_set_size):
        graph_size = random.randint(config.min_size, config.max_size)

        inp_seq, target_seq, edges = graph_util.gen_single(graph_size, config.plan_length, config.max_size)

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
    final_error = 0/config.test_set_size
    print("Final error rate: %.5f" % final_error)

    with open(summary_dir + '/error.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow("%.5f" % final_error)

    train_writer.close
    train_writer.flush



    