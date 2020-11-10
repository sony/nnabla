# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

import os
import sys
import numpy as np 

cifar_data_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..','..', '..', 'cifar10-100-collection'))
sys.path.append(cifar_data_path)
from cifar10_data import data_iterator_cifar10

def cifar10_iterator(config, comm, train=True):

    data_iterator_ = data_iterator_cifar10(batch_size=config['train']['batch_size'], train=train, rng=np.random.RandomState(config['model']['rng']), 
                                      with_memory_cache=config['dataset']['with_memory_cache'], with_file_cache=config['dataset']['with_file_cache'])
    if comm.n_procs > 1:
        data_iterator_ = data_iterator_.slice(rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

    return data_iterator_