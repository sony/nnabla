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


class TrainPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, image_dir, file_list,
                 seed=1, num_gpu=1, random_area=[0.08, 1.0]):
        super(TrainPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=seed)
        self.input = ops.FileReader(file_root=image_dir, file_list=file_list,
                                    random_shuffle=True, num_shards=num_gpu, shard_id=device_id)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.rrc = ops.RandomResizedCrop(
            device="gpu", size=(224, 224), random_area=random_area)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 *
                                                  255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.rrc(images)
        images = self.cmnp(images, mirror=self.coin())
        return images, labels


class ValPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, image_dir, file_list, seed=1, num_gpu=1):
        super(ValPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=seed)
        self.input = ops.FileReader(file_root=image_dir, file_list=file_list,
                                    random_shuffle=False, num_shards=num_gpu, shard_id=device_id)
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


def train():
    """
    Main script.

    Naive Multi-Device Training

    NOTE: the communicator exposes low-level interfaces

    * Parse command line arguments.
    * Instantiate a communicator and set parameter variables.
    * Specify contexts for computation.
    * Initialize DataIterator.
    * Construct a computation graph for training and one for validation.
    * Initialize solver and set parameter variables to that.
    * Create monitor instances for saving and displaying training stats.
    * Training loop
      * Computate error rate for validation data (periodically)
      * Get a next minibatch.
      * Execute forwardprop
      * Set parameter gradients zero
      * Execute backprop.
      * Inplace allreduce (THIS IS THE MAIN difference from a single device training)
      * Solver updates parameters by using gradients computed by backprop.
      * Compute training error

    """

    args = get_args()
    n_train_samples = 1281167
    num_classes = 1000

    # Communicator and Context
    from nnabla.ext_utils import get_extension_context
    extension_module = "cudnn"
    ctx = get_extension_context(extension_module, type_config=args.type_config)
    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    device_id = mpi_rank
    ctx.device_id = str(device_id)
    nn.set_default_context(ctx)

    # Pipelines and Iterators for training
    train_pipes = [TrainPipeline(args.batch_size, args.num_threads, device_id,
                                 args.train_cachefile_dir, args.train_list,
                                 seed=device_id + 1,
                                 num_gpu=n_devices,
                                 random_area=args.random_area)]
    train_pipes[0].build()
    data = DALIClassificationIterator(train_pipes,
                                      train_pipes[0].epoch_size(
                                          "Reader") // n_devices,
                                      auto_reset=True,
                                      stop_at_epoch=False)
    # Pipelines and Iterators for validation
    val_pipes = [ValPipeline(args.batch_size, args.num_threads, device_id,
                             args.val_cachefile_dir, args.val_list,
                             seed=device_id + 1,
                             num_gpu=n_devices)]
    val_pipes[0].build()
    vdata = DALIClassificationIterator(val_pipes, val_pipes[0].epoch_size("Reader") // n_devices,
                                       auto_reset=True,
                                       stop_at_epoch=False)
    # Network for training
    t_model = get_model(args, num_classes, n_devices,
                        args.accum_grad, test=False)
    t_model.pred.persistent = True  # Not clearing buffer of pred in backward
    t_pred2 = t_model.pred.get_unlinked_variable(need_grad=False)
    t_e = F.mean(F.top_n_error(t_pred2, t_model.label))
    # Network for validation
    v_model = get_model(args, num_classes, n_devices,
                        args.accum_grad, test=True)
    v_model.pred.persistent = True  # Not clearing buffer of pred in forward
    v_pred2 = v_model.pred.get_unlinked_variable(need_grad=False)
    v_e = F.mean(F.top_n_error(v_pred2, v_model.label))

    # Solver
    solver = S.Momentum(args.learning_rate, 0.9)
    solver.set_learning_rate(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Monitors
    import nnabla.monitor as M
    monitor = M.Monitor(args.monitor_path)
    monitor_loss = M.MonitorSeries("Training loss", monitor, interval=10)
    monitor_err = M.MonitorSeries("Training error", monitor, interval=10)
    monitor_vloss = M.MonitorSeries("Validation loss", monitor, interval=1)
    monitor_verr = M.MonitorSeries("Validation error", monitor, interval=1)
    monitor_time = M.MonitorTimeElapsed("Training time", monitor, interval=10)
    monitor_vtime = M.MonitorTimeElapsed(
        "Validation time", monitor, interval=1)

    # Training loop
    vl = nn.Variable()
    ve = nn.Variable()
    for i in range(int(args.max_iter / n_devices)):
        # Save parameters
        if i % (args.model_save_interval // n_devices) == 0 and device_id == 0:
            nn.save_parameters(os.path.join(
                args.model_save_path, 'param_%06d.h5' % i))

        # Validation
        if i % (args.val_interval // n_devices) == 0 and i != 0:
            ve_local = 0.
            vl_local = 0.
            val_iter_local = args.val_iter // n_devices
            for j in range(val_iter_local):
                nextImage, nextLabel = vdata.next()
                v_model.image.data = nextImage
                v_model.label.data = nextLabel
                v_model.loss.forward(clear_buffer=True)
                v_e.forward(clear_buffer=True)
                vl_local += v_model.loss.d.copy()
                ve_local += v_e.d.copy()
            vl_local /= val_iter_local
            vl.d = vl_local
            comm.all_reduce(vl.data, division=True, inplace=True)
            ve_local /= val_iter_local
            ve.d = ve_local
            comm.all_reduce(ve.data, division=True, inplace=True)

            if device_id == 0:
                monitor_vloss.add(i * n_devices, vl.d.copy())
                monitor_verr.add(i * n_devices, ve.d.copy())
                monitor_vtime.add(i * n_devices)

        # Training
        l = 0.0
        e = 0.0
        solver.zero_grad()

        def accumulate_error(l, e, t_model, t_e):
            l += t_model.loss.d
            e += t_e.d
            return l, e

        # Gradient accumulation loop
        for j in range(args.accum_grad):
            nextImage, nextLabel = data.next()
            t_model.image.data = nextImage
            t_model.label.data = nextLabel
            t_model.loss.forward(clear_no_need_grad=True)
            t_model.loss.backward(clear_buffer=True)  # Accumulating gradients
            t_e.forward(clear_buffer=True)
            l, e = accumulate_error(l, e, t_model, t_e)

        # AllReduce
        params = [x.grad for x in nn.get_parameters().values()]
        comm.all_reduce(params, division=False, inplace=False)

        # Update
        solver.weight_decay(args.weight_decay)
        solver.update()

        if device_id == 0:
            monitor_loss.add(i * n_devices, l / args.accum_grad)
            monitor_err.add(i * n_devices, e / args.accum_grad)
            monitor_time.add(i * n_devices)

        # Learning rate decay at scheduled iter
        if i * n_devices in args.learning_rate_decay_at:
            solver.set_learning_rate(solver.learning_rate() * 0.1)

    if device_id == 0:
        nn.save_parameters(os.path.join(
            args.model_save_path,
            'param_%06d.h5' % (args.max_iter / n_devices)))


if __name__ == '__main__':
    """
    Call this script with `mpirun` or `mpiexec`
    $ mpirun -n 4 python dali_multi_device_multi_process_classification.py -b 32 -a 2 -L 50 -l 0.1 -i 2000000 -v 20004 -j 1563 -s 20004 -D 600000,1200000,1800000 -T "path of train data directory" -TL "train label file" -V "path of validation data directory" -VL "validation label file" -N 4
    """
    train()
