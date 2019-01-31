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

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.communicators as C

from args import get_args
from segmentation_data import data_iterator_segmentation
import model

import os
from collections import namedtuple


def get_model(args, test=False):
    """
    Create computation graph and variables.

    """
    nn_in_size = 513
    
    image = nn.Variable([args.batch_size, 3, nn_in_size, nn_in_size])
    label = nn.Variable([args.batch_size, 1, nn_in_size, nn_in_size])
    mask = nn.Variable([args.batch_size, 1, nn_in_size, nn_in_size])
    
    pred = model.deeplabv3plus_model(image, args.output_stride, args.num_class, test=test, fix_params=False)

    #Initializing moving variance by 1
    params = nn.get_parameters()
    for key,val in params.items():
        if 'bn/var' in key:
            val.d.fill(1)


    loss = F.sum(F.softmax_cross_entropy(pred, label,axis=1) * mask) / F.sum(mask)
    Model = namedtuple('Model', ['image', 'label', 'mask', 'pred', 'loss'])
    return Model(image, label, mask, pred, loss)



def one_hot_encode(label, num_class):
    # label is 2-d of shape (batch, pixels(513x513))
    one_hot_vector = np.zeros((label.shape[0],label.shape[1], num_class), dtype=np.int32)

    batch_id = np.arange(label.shape[0]).reshape(label.shape[0],1)
    pixel_id = np.tile(np.arange(label.shape[1]), (label.shape[0],1))

    one_hot_vector[batch_id, pixel_id, label.astype(np.int32)] = 1

    return one_hot_vector


def compute_miou(num_classes, gt, pred, mask):
    
    gt = np.squeeze(np.transpose(gt, (1,0,2,3)), axis=0)
    gt = gt.reshape(gt.shape[0],gt.shape[1]*gt.shape[2])
    gt = one_hot_encode(gt, num_classes)

    pred = pred.reshape(pred.shape[0],pred.shape[1]*pred.shape[2])
    pred = one_hot_encode(pred, num_classes)

    # Now, gt: [B, P, C] ; pred: [B, P, C]   
    mask = np.squeeze(np.transpose(mask, (1,0,2,3)), axis=0)  
    mask = mask.reshape(mask.shape[0],mask.shape[1]*mask.shape[2]) #mask: [B, P]
    mask = mask[..., None]

    numer = np.sum(np.logical_and(gt, pred) * mask, axis=1)  # gt: [B, P, C], mask: [B, P, 1]
    denom = np.sum(np.logical_or(gt, pred) * mask, axis=1)
    iou = numer / np.maximum(denom, 1)

    cat_mask = np.max(gt, axis=1) 
    iou_per_image = np.sum(iou * cat_mask, axis=1) / np.sum(cat_mask, axis=1)
    miou = np.mean(iou_per_image) 

    return miou.mean()

def train():
    """
    Main script.
    """

    args = get_args()

    _ = nn.load_parameters(args.pretrained_model_path)
    if args.fine_tune:
        nnabla.parameter.pop_parameter('decoder/logits/affine/conv/W')
        nnabla.parameter.pop_parameter('decoder/logits/affine/conv/b')


    n_train_samples = args.train_samples
    n_val_samples = args.val_samples
    distributed = args.distributed
    compute_acc = args.compute_acc

    if distributed:
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
    else:
        # Get context.
        from nnabla.ext_utils import get_extension_context
        extension_module = args.context
        if args.context is None:
            extension_module = 'cpu'
        logger.info("Running in %s" % extension_module)
        ctx = get_extension_context(
            extension_module, device_id=args.device_id, type_config=args.type_config)
        nn.set_default_context(ctx)
        n_devices = 1
        device_id=0

    #training data
    data = data_iterator_segmentation(
            args.train_samples, args.batch_size, args.train_dir, args.train_label_dir)
    #validation data
    vdata = data_iterator_segmentation(args.val_samples, args.batch_size, args.val_dir, args.val_label_dir)

    if distributed:
        data = data.slice(
            rng=None, num_of_slices=n_devices, slice_pos=device_id)
        vdata = vdata.slice(
            rng=None, num_of_slices=n_devices, slice_pos=device_id)
    num_classes = args.num_class

     

    # Workaround to start with the same initialized weights for all workers.
    np.random.seed(313)
    t_model = get_model(
        args, test=False)
    t_model.pred.persistent = True  # Not clearing buffer of pred in backward
    t_pred2 = t_model.pred.unlinked()
    t_e = F.sum(F.top_n_error(t_pred2, t_model.label, axis=1)* t_model.mask) / F.sum(t_model.mask)

    v_model = get_model(
        args, test=True)
    v_model.pred.persistent = True  # Not clearing buffer of pred in forward
    v_pred2 = v_model.pred.unlinked()
    v_e = F.sum(F.top_n_error(v_pred2, v_model.label, axis=1)* v_model.mask) / F.sum(t_model.mask)


    # Create Solver
    solver = S.Momentum(args.learning_rate, 0.9)
    solver.set_parameters(nn.get_parameters())

    # Setting warmup.
    base_lr = args.learning_rate / n_devices
    warmup_iter = int(1. * n_train_samples /
                      args.batch_size / args.accum_grad / n_devices) * args.warmup_epoch
    warmup_slope = base_lr * (n_devices - 1) / warmup_iter
    solver.set_learning_rate(base_lr)


    # Create monitor
    import nnabla.monitor as M
    monitor = M.Monitor(args.monitor_path)
    monitor_loss = M.MonitorSeries("Training loss", monitor, interval=10)
    monitor_err = M.MonitorSeries("Training error", monitor, interval=10)
    monitor_vloss = M.MonitorSeries("Validation loss", monitor, interval=1)
    monitor_verr = M.MonitorSeries("Validation error", monitor, interval=1)
    monitor_time = M.MonitorTimeElapsed("Training time", monitor, interval=10)
    monitor_miou = M.MonitorSeries("mean IOU", monitor, interval=10)
    monitor_vtime = M.MonitorTimeElapsed(
        "Validation time", monitor, interval=1)


    # Training loop
    for i in range(int(args.max_iter / n_devices)):
        # Save parameters
        if i % (args.model_save_interval // n_devices) == 0 and device_id == 0:
            nn.save_parameters(os.path.join(
                args.model_save_path, 'param_%06d.h5' % i))

        # Validation
        if i % (args.val_interval // n_devices) == 0 and i != 0:
            vmiou_local = 0.
            val_iter_local = n_val_samples // args.batch_size
            vl_local = nn.NdArray()
            vl_local.zero()
            ve_local = nn.NdArray()
            ve_local.zero()
            for j in range(val_iter_local):
                images, labels, masks = vdata.next()
                v_model.image.d = images
                v_model.label.d = labels
                v_model.mask.d = masks
                v_model.image.data.cast(np.float32, ctx)
                v_model.label.data.cast(np.int32, ctx)
                v_model.loss.forward(clear_buffer=True)
                v_e.forward(clear_buffer=True)
                vl_local += v_model.loss.data
                ve_local += v_e.data
                # Mean IOU computation
                if compute_acc:
                    vmiou_local += compute_miou(num_classes, labels, np.argmax(v_model.pred.d, axis=1), masks)

            vl_local /= val_iter_local
            ve_local /= val_iter_local
            if compute_acc:
                vmiou_local /= val_iter_local
                vmiou_ndarray = nn.NdArray.from_numpy_array(np.array(vmiou_local))
            if distributed:
                comm.all_reduce(vl_local, division=True, inplace=True)
                comm.all_reduce(ve_local, division=True, inplace=True)
                if compute_acc:
                    comm.all_reduce(vmiou_ndarray, division=True, inplace=True)


            if device_id == 0:
                monitor_vloss.add(i * n_devices, vl_local.data.copy())
                monitor_verr.add(i * n_devices, ve_local.data.copy())
                if compute_acc:
                    monitor_miou.add(i* n_devices, vmiou_local)
                monitor_vtime.add(i * n_devices)

        # Training
        l = 0.0
        e = 0.0
        solver.zero_grad()

        e_acc = nn.NdArray(t_e.shape)
        e_acc.zero()
        l_acc = nn.NdArray(t_model.loss.shape)
        l_acc.zero()
        # Gradient accumulation loop
        for j in range(args.accum_grad):
            images, labels, masks = data.next()
            t_model.image.d = images
            t_model.label.d = labels
            t_model.mask.d = masks
            t_model.image.data.cast(np.float32, ctx)
            t_model.label.data.cast(np.int32, ctx)
            t_model.loss.forward(clear_no_need_grad=True)
            t_model.loss.backward(clear_buffer=True)  # Accumulating gradients
            t_e.forward(clear_buffer=True)
            e_acc += t_e.data
            l_acc += t_model.loss.data

        # AllReduce
        if distributed:
            params = [x.grad for x in nn.get_parameters().values()]
            comm.all_reduce(params, division=False, inplace=False)
            comm.all_reduce(l_acc, division=True, inplace=True)
            comm.all_reduce(e_acc, division=True, inplace=True)
        solver.scale_grad(1./args.accum_grad)
        solver.weight_decay(args.weight_decay)
        solver.update()


        # Linear Warmup
        if i <= warmup_iter:
            lr = base_lr + warmup_slope * i
            solver.set_learning_rate(lr)

        if distributed:
            # Synchronize by averaging the weights over devices using allreduce
            if (i+1) % args.sync_weight_every_itr == 0:
                weights = [x.data for x in nn.get_parameters().values()]
                comm.all_reduce(weights, division=True, inplace=True)

        if device_id == 0:
            monitor_loss.add(i * n_devices, (l_acc / args.accum_grad).data.copy())
            monitor_err.add(i * n_devices, (e_acc / args.accum_grad).data.copy())
            monitor_time.add(i * n_devices)

        # Learning rate decay at scheduled iter --> changed to poly learning rate decay policy
        #if i in args.learning_rate_decay_at:
        solver.set_learning_rate(base_lr * ((1 - i / args.max_iter)**0.1) )

    if device_id == 0:
        nn.save_parameters(os.path.join(args.model_save_path,
                                    'param_%06d.h5' % args.max_iter))


if __name__ == '__main__':
    train()
