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


# This file was forked from https://github.com/marvis/pytorch-yolo2 ,
# licensed under the MIT License (see LICENSE.external for more details).


import dataset
from utils import *
import region_loss
import os

import nnabla as nn
import nnabla.solvers as S
from region_loss import create_network

def adjust_learning_rate(solver, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        if batch >= args.steps[i]:
            lr = lr * scale
            if batch == args.steps[i]:
                break
        else:
            break
    solver.set_learning_rate(lr/(batch_size*args.accum_times))
    return lr

def train(epoch):
    global processed_batches
    global seen
    global region_loss_seen

    t0 = time.time()

    def batch_iter(it, batch_size):
        def list2np(t):
            imgs, labels = zip(*t)
            retimgs = np.zeros((len(imgs),) + imgs[0].shape, dtype=np.float32)
            retlabels = np.zeros((len(labels),) + labels[0].shape, dtype=np.float32)
            for i, img in enumerate(imgs):
                retimgs[i,:,:,:] = img
            for i, label in enumerate(labels):
                retlabels[i,:] = label
            return retimgs, retlabels
        retlist = []
        for i, item in enumerate(it):
            retlist.append(item)
            if i % batch_size == batch_size - 1:
                ret = list2np(retlist)
                # Don't train for batches that contain no labels
                if not (np.sum(ret[1]) == 0):
                    yield ret
                retlist = []
        # Excess data is discarded
        if len(retlist) > 0:
            ret = list2np(retlist)
            # Don't train for batches that contain no labels
            if not (np.sum(ret[1]) == 0):
                yield ret

    train_loader_base = dataset.listDataset(args.train, args, shape=(init_width, init_height),
                   shuffle=True,
                   train=True,
                   seen=seen,
                   batch_size=batch_size,
                   num_workers=num_workers)
    train_loader = batch_iter(iter(train_loader_base), batch_size=batch_size)

    lr = adjust_learning_rate(solver_convweights, processed_batches)
    lr = adjust_learning_rate(solver_others, processed_batches)
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader_base), lr))

    yolo_x_nnabla, yolo_features_nnabla, yolo_vars, yolo_tvars, loss_nnabla = create_network(batch_size, init_height, init_width, args)

    t1 = time.time()
    step_called = False
    solver_convweights.zero_grad()
    solver_others.zero_grad()
    for batch_idx, (data_tensor, target_tensor) in enumerate(train_loader):
        dn = data_tensor
        tn = target_tensor
        if dn.shape != yolo_x_nnabla.shape:
            del(yolo_x_nnabla)
            del(yolo_features_nnabla)
            yolo_x_nnabla, yolo_features_nnabla, yolo_vars, yolo_tvars, loss_nnabla = create_network(dn.shape[0], dn.shape[2], dn.shape[3], args)
        yolo_x_nnabla.d = dn
        yolo_features_nnabla.forward()

        for v in yolo_vars:
            v.forward()

        region_loss_seen += data_tensor.shape[0]

        nGT, nCorrect, nProposals = region_loss.forward_nnabla(args, region_loss_seen, yolo_features_nnabla, target_tensor, yolo_vars, yolo_tvars)
        loss_nnabla.forward()
        loss_nnabla.backward()
        print('%d: nGT %d, recall %d, proposals %d, loss: total %f' % (region_loss_seen, nGT, nCorrect, nProposals, loss_nnabla.d))

        yolo_features_nnabla.backward(grad=None, clear_buffer=True)
        if batch_idx % args.accum_times == args.accum_times - 1:
            step_called = True
            solver_convweights.weight_decay(args.decay*batch_size)
            solver_convweights.update()
            solver_convweights.zero_grad()
            solver_others.update()
            solver_others.zero_grad()

            processed_batches = processed_batches + 1
            adjust_learning_rate(solver_convweights, processed_batches)
            adjust_learning_rate(solver_others, processed_batches)
        t1 = time.time()
    if not step_called:
        solver_convweights.weight_decay(args.decay*batch_size)
        solver_convweights.update()
        solver_convweights.zero_grad()
        solver_others.update()
        solver_others.zero_grad()
    print()
    t1 = time.time()
    logging('training with %f samples/s' % (len(train_loader_base)/(t1-t0)))
    if (epoch+1) % args.save_interval == 0:
        logging('save weights to %s/%06d.h5' % (args.output, epoch+1))
        seen = (epoch + 1) * len(train_loader_base)
        nn.save_parameters('%s/%06d.h5' % (args.output, epoch+1))

if __name__ == '__main__':
    # Training settings
    args = parse_args()

    nsamples      = file_lines(args.train)
    ngpus         = len(args.gpus.split(','))
    num_workers   = args.num_workers

    batch_size    = args.batch_size
    learning_rate = args.learning_rate

    # Training parameters
    max_epochs    = args.max_batches*batch_size*args.accum_times/nsamples+1
    use_cuda      = args.use_cuda
    seed          = args.seed

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    ###############

    seen = 0
    region_loss_seen  = 0
    processed_batches = 0/(batch_size*args.accum_times)

    init_width        = args.width
    init_height       = args.height
    init_epoch        = seen/nsamples


    yolo_x_nnabla, yolo_features_nnabla, yolo_vars, yolo_tvars, loss_nnabla = create_network(batch_size, init_height, init_width, args)

    from nnabla.contrib.context import extension_context
    ctx = extension_context("cudnn")
    nn.set_default_context(ctx)

    # Load parameters
    print("Load", args.weight, "...")
    nn.load_parameters(args.weight)
    print(nn.get_parameters())

    param_convweights = {k: v for k, v in nn.get_parameters().items() if k.endswith("conv/W")}
    param_others = {k: v for k, v in nn.get_parameters().items() if not k.endswith("conv/W")}

    solver_convweights = S.Momentum(learning_rate, args.momentum)
    solver_others = S.Momentum(learning_rate, args.momentum)
    solver_convweights.set_parameters(param_convweights)
    solver_others.set_parameters(param_others)
    print(init_epoch, max_epochs)

    for epoch in range(int(init_epoch), int(max_epochs)):
        train(epoch)
