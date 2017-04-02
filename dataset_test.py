import unittest
import dataset
import dataset_util
import random
import numpy as np
np.set_printoptions(threshold=np.nan)
import graph as gr
import copy

class TestDatasetGeneration(unittest.TestCase):

    def test_collision(self):
        dataset_util.build_dataset_file(1000, 1000, 100, 4)
        dset = dataset.Dataset("4_1000")
        print(dset.train_set_bins[1][0])

        inp, target = dset.get_training_data()
        print(np.array(target[0]))

        v_inp, vtarget = dset.get_validation_data()

if __name__ == '__main__':
    unittest.main()