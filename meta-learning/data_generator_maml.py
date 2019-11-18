# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Code for loading data. """
import numpy as np
import os
import random


class DataGenerator:
    def __init__(self, num_classes, num_shots, num_queries, shape_x, dataset, batch_size):
        """
        Create episode function
            Args:
                num_classes (int): number of support classes, generally called n_way in one-shot literatuer
                num_shots (int): number of shots per class
                num_queries (int): numper of queries per class.
                shape_x (tuple): dimension of the input image.
                dataset(nd_array): nd_array of (class, sample, shape)
                batch_size (int): number of tasks sampled per meta-update
        """
        self.num_classes = num_classes
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.shape_x = shape_x
        self.dataset = dataset
        self.batch_size = batch_size
        self.rng = np.random.RandomState(706)

    def next(self):
        """
        Create episode function
            Returns:
                inputa (~nnabla.Variable): support images for inner train
                inputb (~nnabla.Variable): query images for meta train
                labela (~nnabla.Variable): support class_id in support sets
                labelb (~nnabla.Variable): query class_id in query sets
        """
        # parameters
        num_classes = self.num_classes
        num_shots = self.num_shots
        num_queries = self.num_queries
        shape_x = self.shape_x
        batch_size = self.batch_size

        # dataset selection
        dataset = self.dataset

        # memory allocations
        inputa = np.zeros((num_classes * num_shots, ) + shape_x)
        inputb = np.zeros((num_classes * num_queries, ) + shape_x)
        labela = np.zeros((num_classes * num_shots, 1), dtype=np.int)
        labelb = np.zeros((num_classes * num_queries, 1), dtype=np.int)

        for bs in range(batch_size):
            # episode classes
            support_class_ids = np.random.choice(
                dataset.shape[0], num_classes, False)

            # operate for each support class
            for i, support_class_id in enumerate(support_class_ids):

                # select support class
                xi = dataset[support_class_id]
                n_sample = xi.shape[0]

                if n_sample < num_shots + num_queries:
                    print("Error: too few samples.")
                    exit()

                # sample indices for support and queries
                sample_ids = np.random.choice(
                    n_sample, num_shots + num_queries, False)
                s_sample_ids = sample_ids[:num_shots]
                q_sample_ids = sample_ids[num_shots:]

                # support set
                for j in range(num_shots):
                    si = i * num_shots + j
                    inputa[si] = xi[s_sample_ids[j]]
                    labela[si] = i

                # query set
                for j in range(num_queries):
                    qi = i * num_queries + j
                    inputb[qi] = xi[q_sample_ids[j]]
                    labelb[qi] = i

            yield inputa, inputb, labela, labelb
