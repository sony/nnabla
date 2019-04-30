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

import numpy as np

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nnabla as nn
import nnabla.communicators as C
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

from args import get_args
import model_resnet
import os
from collections import namedtuple

from dali_data_iterator import DALIClassificationIterator


class ValPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, image_dir, file_list, seed=1, num_gpu=1):
        super(ValPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=seed)
        self.input = ops.FileReader(file_root=image_dir, file_list=file_list,
                                    random_shuffle=False, num_shards=num_gpu, shard_id=0)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=256)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 *
                                                  255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        images = self.cmnp(images)
        return images, labels


def get_model(args, num_classes, n_devices, accum_grad, test=False):
    """
    Create computation graph and variables.
    """
    nn_in_size = 224
    image = nn.Variable([args.batch_size, 3, nn_in_size, nn_in_size])
    label = nn.Variable([args.batch_size, 1])
    pred, hidden = model_resnet.resnet_imagenet(
        image, num_classes, args.num_layers, args.shortcut_type, test=test, tiny=False)
    loss = F.mean(F.softmax_cross_entropy(pred, label)) / \
        (n_devices * accum_grad)
    Model = namedtuple('Model', ['image', 'label', 'pred', 'loss', 'hidden'])
    return Model(image, label, pred, loss, hidden)


def valid():
    """
    Main script for validation.

    """

    args = get_args()
    n_valid_samples = 50000
    num_classes = 1000
    assert n_valid_samples % args.batch_size == 0, \
        "Set batch_size such that n_valid_samples (50000) can be devided by batch_size. \Batch size is now set as {}".format(
            args.batch_size)

    # Context
    from nnabla.ext_utils import get_extension_context
    extension_module = "cudnn"
    ctx = get_extension_context(
        extension_module, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Pipelines and Iterators for validation
    device_id = int(args.device_id)
    val_pipes = [ValPipeline(args.batch_size, args.num_threads, device_id,
                             args.val_cachefile_dir, args.val_list,
                             seed=device_id,
                             num_gpu=1)]
    val_pipes[0].build()
    vdata = DALIClassificationIterator(val_pipes, val_pipes[0].epoch_size("Reader"),
                                       auto_reset=True,
                                       stop_at_epoch=False)

    # Network for validation
    nn.load_parameters(args.model_load_path)
    v_model = get_model(args, num_classes, 1, args.accum_grad, test=True)
    v_e = F.mean(F.top_n_error(v_model.pred, v_model.label, n=args.top_n))

    # Monitors
    import nnabla.monitor as M
    monitor = M.Monitor(args.monitor_path)
    monitor_verr = M.MonitorSeries("Validation error", monitor, interval=1)
    monitor_vtime = M.MonitorTimeElapsed(
        "Validation time", monitor, interval=1)

    # Validation
    ve_local = 0.
    val_iter_local = n_valid_samples // args.batch_size
    for i in range(val_iter_local):
        nextImage, nextLabel = vdata.next()
        v_model.image.data.copy_from(nextImage)
        v_model.label.data.copy_from(nextLabel)
        v_model.image.data.cast(np.float, ctx)
        v_model.label.data.cast(np.int32, ctx)
        v_e.forward(clear_buffer=True)
        nn.logger.info(
            "validation error is {} at {}-th batch".format(v_e.d, i))
        ve_local += v_e.d.copy()
    ve_local /= val_iter_local

    monitor_verr.add(0, ve_local)
    monitor_vtime.add(0)


if __name__ == '__main__':
    """
    """
    valid()
