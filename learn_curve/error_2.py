import pickle
import numpy as np

def calce_percent_edge(result_dir):
    with open("{}/results.pkl".format(result_dir), 'rb') as pickled_dataset:
        results = pickle.load(pickled_dataset)

    step = len(results[1])/5
    print(step)

    overall_sum = 0
    for i in range(0, 5):
        sum = 0
        for j in range(step*i, step*(i+1)):
            sum += (np.sum(results[1][j]))/(i+1)

        print("{}: {}".format(i+1, (sum/8000)*100))
        overall_sum += sum/8000

    print((overall_sum/5)*100)




