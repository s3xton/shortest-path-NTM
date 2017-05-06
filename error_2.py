import pickle
import numpy as np
import dataset

def calc_percent_nodes(result_dir, dset):
    with open("{}/results.pkl".format(result_dir), 'rb') as pickled_dataset:
        results = pickle.load(pickled_dataset)

    for i in range(0, 5):
        bucket = []
        for j in range(8000*i, 8000*(i+1)):
            
            sum_path = 0
            for k, edge in enumerate(results[2][j]):
                edge_tar = dset[j][3][k]
                for l, node in enumerate(edge):
                    # graph|path|edge|node
                    node_tar = int(edge_tar[l])
                    node = int(node)
                    if node != node_tar:
                        sum_path += 1

            bucket.append(sum_path)

        av_bucket = (np.sum(bucket)/8000)/((i+1)*2)
        std_bucket = np.std(bucket)
        conf_int = 1.96*(std_bucket/np.sqrt(40000))
        print("{}: {} +- {}".format(i+1, (av_bucket)*100, conf_int *100))
    



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

def compare_perf():
    result_pos_bins = []
    for i in range(1, 6):
        results = []
        with open("learn_curve/{}0k/results.pkl".format(i), 'rb') as pickled_dataset:
            results = pickle.load(pickled_dataset)
        result_pos_bins.append(results[1])

    scores_bins = []
    for i in range(1, 6):
        scores_bins.append(get_scores(result_pos_bins, i))

    print(np.array(scores_bins))

def get_scores(result_pos_bins, path_length):
    scores = []
    # For each training set size, 10-50k
    for i, set1 in enumerate(result_pos_bins):
        better_count = 0
        # Compare with each other training set size
        for j, set2 in enumerate(result_pos_bins):
            if i != j:
                if better(set1, set2, path_length):
                    better_count += 1
        # Record how many it was better than
        scores.append(better_count)

    return scores



def better(set1, set2, path_length):
    start = 8000 * (path_length-1)
    end = start + 8000
    slice1 = set1[start:end]
    slice2 = set2[start:end]

    m1 = np.mean(np.sum(slice1, 1))
    m2 = np.mean(np.sum(slice2, 1))
    s1 = np.std(np.sum(slice1, 1))
    s2 = np.std(np.sum(slice2, 1))
    n1 = 8000
    n2 = 8000
    print("path_length:{} mean: {}, std: {}".format(path_length, m1, s1))
    pooled = pooled_sd(n1, n2, s1, s1)

    tval = np.abs(m1-m2)/np.sqrt( ( (s1**2) /n1) + ( (s2**2) /n2) )

    if tval > 1.96 and m1 < m2:
        return True
    else:
        return False


def pooled_sd(n1, n2, s1, s2):
    numerator = (n1-1)*(s1**2) + (n2 -1)*(s2**2)
    denominator = n1+n2-2
    pooled = np.sqrt(numerator/denominator)
    return pooled
