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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nnabla_ext.cuda.experimental import dali_iterator

from normalize_config import (
    _pixel_mean, _pixel_std,
    get_normalize_config,
)


def int_div_ceil(a, b):
    '''
    returns int(ceil(a / b))
    '''
    return (a + b - 1) // b


class TrainPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, shard_id, image_dir, file_list, nvjpeg_padding,
                 prefetch_queue=3, seed=1, num_shards=1, channel_last=True,
                 spatial_size=(224, 224), dtype="half",
                 mean=_pixel_mean, std=_pixel_std, pad_output=True):
        super(TrainPipeline, self).__init__(
            batch_size, num_threads, shard_id, seed=seed, prefetch_queue_depth=prefetch_queue)
        self.input = ops.FileReader(file_root=image_dir, file_list=file_list,
                                    random_shuffle=True, num_shards=num_shards, shard_id=shard_id)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB,
                                       device_memory_padding=nvjpeg_padding,
                                       host_memory_padding=nvjpeg_padding)

        self.rrc = ops.RandomResizedCrop(device="gpu", size=spatial_size)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16 if dtype == "half" else types.FLOAT,
                                            output_layout=types.NHWC if channel_last else types.NCHW,
                                            crop=spatial_size,
                                            image_type=types.RGB,
                                            mean=mean,
                                            std=std,
                                            pad_output=pad_output)
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.rrc(images)
        images = self.cmnp(images, mirror=self.coin())
        return images, labels.gpu()


class ValPipeline(Pipeline):
    def __init__(
            self, batch_size, num_threads, shard_id, image_dir, file_list,
            nvjpeg_padding, seed=1, num_shards=1, channel_last=True,
            spatial_size=(224, 224), dtype='half',
            mean=_pixel_mean, std=_pixel_std, pad_output=True):
        super(ValPipeline, self).__init__(
            batch_size, num_threads, shard_id, seed=seed)
        self.input = ops.FileReader(file_root=image_dir, file_list=file_list,
                                    random_shuffle=False, num_shards=num_shards, shard_id=shard_id)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB,
                                       device_memory_padding=nvjpeg_padding,
                                       host_memory_padding=nvjpeg_padding)
        import args as A
        resize_shorter = A.resize_by_ratio(spatial_size[0])
        self.res = ops.Resize(device="gpu", resize_shorter=resize_shorter)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16 if dtype == "half" else types.FLOAT,
                                            output_layout=types.NHWC if channel_last else types.NCHW,
                                            crop=spatial_size,
                                            image_type=types.RGB,
                                            mean=mean,
                                            std=std,
                                            pad_output=pad_output)

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        images = self.cmnp(images)
        return images, labels.gpu()


def get_pad_output_by_channels(channels):
    if channels == 4:
        return True
    elif channels == 3:
        return False
    raise ValueError(f'channels must be 3 or 4. Given {channels}')


def get_train_data_iterator(args, comm, channels, spatial_size=(224, 224), norm_config='default'):
    # Pipelines and Iterators for training
    mean, std = get_normalize_config(norm_config)
    if std is None:
        std = [1., 1., 1.]
    pad_output = get_pad_output_by_channels(channels)
    train_pipe = TrainPipeline(args.batch_size, args.dali_num_threads, comm.rank,
                               args.train_dir,
                               args.train_list, args.dali_nvjpeg_memory_padding,
                               seed=comm.rank + 1,
                               num_shards=comm.n_procs,
                               channel_last=args.channel_last,
                               spatial_size=spatial_size,
                               dtype=args.type_config,
                               mean=list(mean), std=list(std),
                               pad_output=pad_output)

    data = dali_iterator.DaliIterator(train_pipe)
    data.size = int_div_ceil(
        train_pipe.epoch_size("Reader"), comm.n_procs)
    return data


def get_val_data_iterator(args, comm, channels, spatial_size=(224, 224), norm_config='default'):
    # Pipelines and Iterators for validation
    mean, std = get_normalize_config(norm_config)
    if std is None:
        std = [1., 1., 1.]
    pad_output = get_pad_output_by_channels(channels)
    val_pipe = ValPipeline(args.batch_size, args.dali_num_threads, comm.rank,
                           args.val_dir, args.val_list, args.dali_nvjpeg_memory_padding,
                           seed=comm.rank + 1,
                           num_shards=comm.n_procs,
                           channel_last=args.channel_last,
                           spatial_size=spatial_size,
                           dtype=args.type_config,
                           mean=list(mean), std=list(std),
                           pad_output=pad_output)
    vdata = dali_iterator.DaliIterator(val_pipe)
    vdata.size = int_div_ceil(
        val_pipe.epoch_size("Reader"), comm.n_procs)
    return vdata


def get_data_iterators(args, comm, channels, spatial_size=(224, 224), norm_config='default'):
    '''
    Creates and returns DALI data iterators for both datasets of training and
    validation.

    The datasets are partitioned in distributed training
    mode according to comm rank and number of processes.
    '''
    data = get_train_data_iterator(
        args, comm, channels, spatial_size, norm_config)
    vdata = get_val_data_iterator(
        args, comm, channels, spatial_size, norm_config)
    return data, vdata
