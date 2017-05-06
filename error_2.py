import pickle
import numpy as np
import dataset

def calc_percent_nodes(result_dir, dset):
    with open("{}/results.pkl".format(result_dir), 'rb') as pickled_dataset:
        results = pickle.load(pickled_dataset)

    overall_sum = 0
    for i in range(0, 5):
        sum_bucket = 0
        for j in range(8000*i, 8000*(i+1)):
            sum_path = 0
            for k, edge in enumerate(results[2][j]):
                edge_tar = dset[j][3][k]
                for l, node in enumerate(edge):
                    # graph|path|edge|node
                    node_tar = edge_tar[l]
                    if node == node_tar:
                        sum_path += 1
            sum_bucket = sum_path/(i+1)
        print("{}: {}".format(i+1, (1 - sum_bucket/8000)*100))  
        overall_sum += (sum_bucket/8000)

    print((1 - overall_sum/5)*100)



def calce_percent_edge(result_dir):
    with open("{}/results.pkl".format(result_dir), 'rb') as pickled_dataset:
        results = pickle.load(pickled_dataset)

    overall_sum = 0
    for i in range(0, 5):
        sum = 0
        for j in range(8000*i, 8000*(i+1)):
            sum += (np.sum(results[1][j]))/(i+1)

        print("{}: {}".format(i+1, (sum/8000)*100))
        overall_sum += sum/8000

    print((overall_sum/5)*100)

def calc_percent_consec(result_dir):
    with open("{}/results.pkl".format(result_dir), 'rb') as pickled_dataset:
        results = pickle.load(pickled_dataset)

    overall_sum = 0
    for i in range(0, 5):
        sum = 0
        for j in range(8000*i, 8000*(i+1)):
            count = 0
            k = 0
            current = results[1][j][k]
            while current != 1 and k < (i+1):
                count += 1
                k += 1
                current = results[1][j][k]
            sum += count/(i+1)

        print("{}: {}".format(i+1, (1 - sum/8000)*100))
        overall_sum += sum/8000

    print((1 - overall_sum/5)*100)


