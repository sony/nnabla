# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

from __future__ import absolute_import

from .dali_data_iterator import DALIClassificationIterator
from .dali_config import DaliConfig as conf

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

_norm_mean = [255 * 0.5] * 3
_norm_std = [255 * 0.5] * 3


class TrainPipeline(Pipeline):
    def __init__(self, batch_size, shard_id, image_dir, file_list,
                 seed=1, num_shards=1, channel_last=True, dtype="half"):
        super(TrainPipeline, self).__init__(
            batch_size, conf.num_threads, shard_id,
            seed=seed, prefetch_queue_depth=conf.prefetch_queue)

        # file_list: [[image_path, labelId_path, instanceId_path], ...]

        self.input_image = ops.ExternalSource()
        self.input_labelId = ops.ExternalSource()
        self.input_instId = ops.ExternalSource()

        # todo: currently all images supposed to be saved as png.
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB,
                                        device_memory_padding=conf.nvjpeg_padding,
                                        host_memory_padding=conf.nvjpeg_padding)

        self.rrc = ops.RandomResizedCrop(device="gpu", size=(224, 224))
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16 if dtype == "half" else types.FLOAT,
                                            output_layout=types.NHWC if channel_last else types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=_norm_mean,
                                            std=_norm_std)
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.rrc(images)
        images = self.cmnp(images, mirror=self.coin())
        return images, labels


def get_dali_data_iterators(args, comm):
    '''
    Creates and returns DALI data iterators for both datasets of training and
    validation.

    The datasets are partitioned in distributed training
    mode according to comm rank and number of processes.
    '''

    # Pipelines and Iterators for training
    train_pipes = [TrainPipeline(args.batch_size, args.dali_num_threads, comm.rank,
                                 args.train_dir,
                                 args.train_list, args.dali_nvjpeg_memory_padding,
                                 seed=comm.rank + 1,
                                 num_shards=comm.n_procs,
                                 channel_last=args.channel_last,
                                 dtype=args.type_config)]

    train_pipes[0].build()
    train_iterator = DALIClassificationIterator(train_pipes,
                                                train_pipes[0].epoch_size(
                                                    "Reader") // comm.n_procs,
                                                auto_reset=True,
                                                stop_at_epoch=False)

    # Pipelines and Iterators for validation
    val_pipes = [ValPipeline(args.batch_size, args.dali_num_threads, comm.rank,
                             args.val_dir, args.val_list, args.dali_nvjpeg_memory_padding,
                             seed=comm.rank + 1,
                             num_shards=comm.n_procs,
                             channel_last=args.channel_last,
                             dtype=args.type_config)]
    val_pipes[0].build()
    val_iterator = DALIClassificationIterator(val_pipes, val_pipes[0].epoch_size("Reader") // comm.n_procs,
                                              auto_reset=True,
                                              stop_at_epoch=False)

    return train_iterator, val_iterator
