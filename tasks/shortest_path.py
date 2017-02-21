import os
import time
import numpy as np
import tensorflow as tf
import random
import graph_util

from utils import pprint

print_interval = 5

def train(ntm, config, sess):
    if not os.path.isdir(config.checkpoint_dir):
        raise Exception(" [!] Directory %s not found" % config.checkpoint_dir)

    # delimiter flag for start and end
    start_symbol = np.zeros([config.input_dim], dtype=np.float32)
    start_symbol[0] = 1
    end_symbol = np.zeros([config.input_dim], dtype=np.float32)
    end_symbol[1] = 1

    print(" [*] Initialize all variables")
    tf.global_variables_initializer().run()
    print(" [*] Initialization finished")

    if config.continue_train is not False:
        ntm.load(config.checkpoint_dir, config.task, strict=config.continue_train is True)

    start_time = time.time()
    for idx in range(config.epoch):
        graph_size = random.randint(config.min_size, config.max_size)

        inp_seq, target_seq = graph_util.gen_single(graph_size)
        seq_length = len(inp_seq)

        feed_dict = {input_: vec for vec, input_ in zip(inp_seq, ntm.inputs)}
        feed_dict.update(
            {true_output: vec for vec, true_output in zip(target_seq, ntm.true_outputs)}
        )

        feed_dict.update({
            ntm.start_symbol: start_symbol,
            ntm.end_symbol: end_symbol
        })

        _, cost, _, outputs = sess.run([ntm.optims[seq_length],
                               ntm.get_loss(seq_length),
                               ntm.global_step, ntm.get_output_logits(seq_length)], feed_dict=feed_dict)

        if idx % print_interval == 0:
            print(
                "[%5d] %2d: %.10f (%.1fs)"
                % (idx, seq_length, cost, time.time() - start_time))
            #print(outputs)



    