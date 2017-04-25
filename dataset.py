import random
import math
import pickle
import graph as gr


class Dataset:
    #SET -> BINS -> [GRAPH | START | END | PATH | LENGTH]
    def __init__(self, graph_size, dataset_dir):
        self.train_set_bins = []
        self.val_set_bins = []
        self.test_set_bins = []

        for i in range(1, graph_size):
            with open("{}/train/bin_{}.pkl".format(dataset_dir, i), 'rb') as pickled_dataset:
                self.train_set_bins.append(pickle.load(pickled_dataset))
            with open("{}/val/bin_{}.pkl".format(dataset_dir, i), 'rb') as pickled_dataset:
                self.val_set_bins.append(pickle.load(pickled_dataset))
            with open("{}/test/bin_{}.pkl".format(dataset_dir, i), 'rb') as pickled_dataset:
                self.test_set_bins.append(pickle.load(pickled_dataset))


    def get_validation_data(self, val_size=0):
        return self.get_encoded_data(self.val_set_bins, val_size)

    def get_training_data(self, train_size=0):
        return self.get_encoded_data(self.train_set_bins, train_size)

    def get_encoded_data(self, draw_set, draw_size):
        total_size = 0
        for path_bin in draw_set:
            total_size += len(path_bin)

        if draw_size < total_size:
            if draw_size != 0:
                max_per_bin = int(draw_size/(len(draw_set)))
            input_set = []
            target_set = []
            lengths = []
            dist = {}
            input_set_unencoded = []
            for i in range(0, len(draw_set)):
                path_bin = draw_set[i]
                if draw_size == 0:
                    max_per_bin = len(path_bin)
                for j in range(0, max_per_bin):
                    graph = path_bin[j][0]
                    start = path_bin[j][1]
                    end = path_bin[j][2]
                    path = path_bin[j][3]
                    inp, target = self.encode_graph_data(graph, start, end, path)
                    input_set.append(inp)
                    target_set.append(target)
                    lengths.append(len(path))
                    input_set_unencoded.append(path_bin[j])
                dist[i+1] = max_per_bin
            return input_set, target_set, lengths, dist, input_set_unencoded
        else:
            print("[!] There are only %d training examples. You asked for %d"
                  %(total_size, draw_size))


    def encode_graph_data(self, graph, start, end, path):
        plan_length = 10
        max_graph_size = 10
        # Construct description phase
        max_desc_length = max_graph_size * (max_graph_size - 1) / 2
        encoded_graph = []
        for edge in graph.edge_list:
            node_a = self.__decimal_to_onehot(edge[0])
            node_b = self.__decimal_to_onehot(edge[1])
            encoded_edge = [0, 0] + node_a + node_b
            encoded_graph.append(encoded_edge)

        pad_length = max_desc_length - len(encoded_graph)
        desc_phase = encoded_graph + ([[0] * 22] * int(pad_length))

        # Encode the query phase
        query_phase = [[0, 1] + self.__decimal_to_onehot(start) + self.__decimal_to_onehot(end)]

        # Construct the plan phase
        plan_phase = [[1, 0] + [0] * 20] * plan_length

        # Encode the answer phase
        answer_phase = [[1, 1] + [0] * 20] * len(path)

        # Construct the final input with padding
        _input = desc_phase + query_phase + plan_phase + answer_phase
        max_seq_length = max_desc_length + 1 + plan_length + (max_graph_size - 1)
        pad_length = max_seq_length - len(_input)
        _input += [[0] * 22] * int(pad_length)

        # Construct the target
        encoded_path = []
        for edge in path:
            node_a = self.__decimal_to_onehot(edge[0])
            node_b = self.__decimal_to_onehot(edge[1])
            encoded_path.append(node_a + node_b)

        target = [[0] * 20] * (len(desc_phase) + 1 + plan_length) + encoded_path + ([[0] * 20] * int(pad_length))

        return _input, target

    def __decimal_to_onehot(self, decimal_digit):
        one_hot = [0] * 10
        one_hot[-decimal_digit-1] = 1
        return one_hot
