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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

from nnabla.utils.data_iterator import data_iterator_cache
from nnabla_ext.cuda.experimental import dali_iterator

import numpy as np


class DataPipeline(Pipeline):

    def __init__(self, image_dir, batch_size, num_threads, device_id, num_gpus=1, seed=1, train=True):
        super(DataPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=seed)
        self.input = ops.FileReader(file_root=image_dir,
                                    random_shuffle=True, num_shards=num_gpus, shard_id=device_id)
        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        self.rrc = ops.RandomResizedCrop(device='gpu', size=(128, 128))
        self.cmn = ops.CropMirrorNormalize(device='gpu',
                                           crop=(128, 128),
                                           image_type=types.RGB,
                                           mean=[0.5*256, 0.5*256, 0.5*256],
                                           std=[0.5*256, 0.5*256, 0.5*256],
                                           )
        self.coin = ops.CoinFlip(probability=0.5)
        self.res = ops.Resize(device="gpu", resize_shorter=256)
        self.train = train

    def define_graph(self):
        jpegs, labels = self.input(name='Reader')
        images = self.decode(jpegs)
        if self.train:
            images = self.rrc(images)
            images = self.cmn(images, mirror=self.coin())
        else:
            images = self.res(images)
            images = self.cmn(images)
        return images, labels


def imagenet_iterator(config, comm, train=True):
    if config['dataset']['dali']:
        if train:
            pipe = DataPipeline(config['dataset']['path'],
                                config['train']['batch_size'], config['dataset']['dali_threads'], comm.rank,
                                num_gpus=comm.n_procs, seed=1, train=train)
        else:
            pipe = DataPipeline(config['dataset']['val_path'],
                                config['train']['batch_size'], config['dataset']['dali_threads'], comm.rank,
                                num_gpus=comm.n_procs, seed=1, train=train)

        data_iterator_ = dali_iterator.DaliIterator(pipe)
        data_iterator_.size = np.ceil(pipe.epoch_size("Reader")/comm.n_procs)
        data_iterator_.batch_size = config['train']['batch_size']

        return data_iterator_
    else:
        return data_iterator_cache(config['dataset']['cache_dir'], config['train']['batch_size'],
                                   shuffle=True, normalize=True)
