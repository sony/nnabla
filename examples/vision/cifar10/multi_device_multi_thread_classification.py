import os
import time

from args import get_args
from cifar10_data import data_iterator_cifar10
import nnabla as nn
import nnabla.communicators as C
from nnabla.contrib.context import extension_context
import nnabla.functions as F
from nnabla.initializer import (
    calc_uniform_lim_glorot,
    UniformInitializer)
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np
from multiprocessing.pool import ThreadPool

def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()

def forward_backward(data_train, data, data_label, label, 
                     loss_train, solver):
    data_train.d = data
    data_label.d = label
    loss_train.forward()
    solver.zero_grad()
    loss_train.backward()
    
def cifar10_resnet23_prediction(image, 
                                ctx, device_scope_name, test=False):
    """
    Construct ResNet 23
    """    
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]

        with nn.parameter_scope(scope_name):
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv1"):
                w_init = UniformInitializer(
                    calc_uniform_lim_glorot(C, C/2, kernel=(1, 1)), 
                    rng=rng)
                h = PF.convolution(x, C/2, kernel=(1, 1), pad=(0, 0), 
                                   w_init=w_init, with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv2"):
                w_init = UniformInitializer(
                    calc_uniform_lim_glorot(C/2, C/2, kernel=(3, 3)),
                    rng=rng)
                h = PF.convolution(h, C/2, kernel=(3, 3), pad=(1, 1), 
                                   w_init=w_init, with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"): 
                w_init = UniformInitializer(
                    calc_uniform_lim_glorot(C/2, C, kernel=(1, 1)), 
                    rng=rng)
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0), 
                                   w_init=w_init, with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
        # Residual -> Relu
        h = F.relu(h + x)
        
        # Maxpooling
        if dn:
            h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))
        
        return h

    # Random generator for using the same init parameters in all devices
    rng = np.random.RandomState(0)
    nmaps = 64
    ncls = 10
    
    # Conv -> BN -> Relu
    with nn.context_scope(ctx):
        with nn.parameter_scope(device_scope_name):
            with nn.parameter_scope("conv1"):
                # Preprocess
                if not test:
                    
                    image = F.image_augmentation(image, contrast=1.0,
                                             angle=0.25,
                                             flip_lr=True)
                    image.need_grad = False

                w_init = UniformInitializer(
                    calc_uniform_lim_glorot(3, nmaps, kernel=(3, 3)), 
                    rng=rng)
                h = PF.convolution(image, nmaps, kernel=(3, 3), pad=(1, 1), 
                                   w_init=w_init, with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
        
            h = res_unit(h, "conv2", rng, False)    # -> 32x32
            h = res_unit(h, "conv3", rng, True)     # -> 16x16
            h = res_unit(h, "conv4", rng, False)    # -> 16x16
            h = res_unit(h, "conv5", rng, True)     # -> 8x8
            h = res_unit(h, "conv6", rng, False)    # -> 8x8
            h = res_unit(h, "conv7", rng, True)     # -> 4x4
            h = res_unit(h, "conv8", rng, False)    # -> 4x4
            h = F.average_pooling(h, kernel=(4, 4)) # -> 1x1
        
            w_init = UniformInitializer(
                calc_uniform_lim_glorot(int(np.prod(h.shape[1:])), 10, kernel=(1, 1)), rng=rng)
            pred = PF.affine(h, ncls, w_init=w_init)

    return pred
        
def cifar10_resnet32_loss(pred, label):
    loss = F.mean(F.softmax_cross_entropy(pred, label))
    return loss

def train():
    """
    Naive Multi-Device Training
    
    NOTE: the communicator exposes low-level interfaces for the 
    distributed training, thus it might be changed in the near future.
    
    * Parse command line arguments.
    * Specify contexts for computation.
    * Initialize DataIterator.
    * Construct computation graphs for training and one for validation.
    * Initialize solvers and set parameter variables to those.
    * Instantiate a communicator and set parameter variables.
    * Create monitor instances for saving and displaying training stats.
    * Training loop
      * Computate error rate for validation data (periodically)
      * Get a next minibatch.
      * Execute forwardprops
      * Set parameter gradients zero
      * Execute backprop.
      * Inplace allreduce (THIS IS THE MAIN difference from a single device training)
      * Solver updates parameters by using gradients computed by backprop.
      * Compute training error
    """
    # Parse args
    args = get_args()
    n_train_samples = 50000
    bs_valid = args.batch_size
    
    # Create contexts
    extension_module = args.context
    if extension_module != "cuda" and \
        extension_module != "cuda.cudnn":
        raise Exception("Use `cuda` or `cuda.cudnn` extension_module.")
    n_devices = args.n_devices
    ctxs = []
    for i in range(n_devices):
        ctx = extension_context(extension_module, device_id=i)
        ctxs.append(ctx)
    ctx = ctxs[-1]
    
    # Create training graphs
    input_image_train = []
    preds_train = []
    losses_train = []
    test = False
    for i in range(n_devices):
        image = nn.Variable((args.batch_size, 3, 32, 32))
        label = nn.Variable((args.batch_size, 1))
        device_scope_name = "device{}".format(i)
        
        pred = cifar10_resnet23_prediction(
            image, ctxs[i], device_scope_name, test)
        loss = cifar10_resnet32_loss(pred, label)
        
        input_image_train.append({"image": image, "label": label})
        preds_train.append(pred)
        losses_train.append(loss)
        
    # Create validation graph
    test = True
    device_scope_name = "device{}".format(0)
    image_valid = nn.Variable((bs_valid, 3, 32, 32))
    pred_valid = cifar10_resnet23_prediction(
        image_valid, ctxs[i], device_scope_name, test)
    input_image_valid = {"image": image_valid}
    
    # Solvers
    solvers = []
    for i in range(n_devices):
        with nn.context_scope(ctxs[i]):
            solver = S.Adam()
            device_scope_name = "device{}".format(i)
            with nn.parameter_scope(device_scope_name):
                params = nn.get_parameters()
                solver.set_parameters(params)
            solvers.append(solver)
            
    # Communicator
    comm = C.DataParalellCommunicator(ctx)
    for i in range(n_devices):
        device_scope_name = "device{}".format(i)
        with nn.parameter_scope(device_scope_name):
            ctx = ctxs[i]
            params = nn.get_parameters()
            comm.add_context_and_parameters((ctx, params))
    comm.init()
    
    # Create threadpools with one thread
    pools = []
    for _ in range(n_devices):
        pool = ThreadPool(processes=1)
        pools.append(pool)
        
    # Once forward/backward to safely secure memory
    for device_id in range(n_devices):
        data, label = \
            (np.random.randn(*input_image_train[device_id]["image"].shape),
             (np.random.rand(*input_image_train[device_id]["label"].shape) * 10).astype(np.int32))
                        
        ret = pools[device_id].apply_async(forward_backward, 
                        (input_image_train[device_id]["image"], data,
                        input_image_train[device_id]["label"], label, 
                        losses_train[device_id], solvers[device_id]))
        ret.get()
        losses_train[device_id].d  # sync to host
    
    # Create monitor.
    from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=10)
    monitor_err = MonitorSeries("Training error", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=100)
    monitor_verr = MonitorSeries("Test error", monitor, interval=10)
    with data_iterator_cifar10(args.batch_size, True) as tdata, \
                    data_iterator_cifar10(bs_valid, False) as vdata:
        # Training-loop
        for i in range(int(args.max_iter / n_devices)):
            # Validation
            if i % int(n_train_samples / args.batch_size / n_devices) == 0:
                ve = 0.
                for j in range(args.val_iter):
                    image, label = vdata.next()
                    input_image_valid["image"].d = image
                    pred_valid.forward()
                    ve += categorical_error(pred_valid.d, label)
                ve /= args.val_iter
                monitor_verr.add(i*n_devices, ve)
            if i % int(args.model_save_interval / n_devices) == 0:
                nn.save_parameters(os.path.join(
                    args.model_save_path, 'params_%06d.h5' % i))
            
            # Forwards/Zerograd/Backwards
            fb_results = []
            for device_id in range(n_devices):
                image, label = tdata.next()
                
                res = pools[device_id].apply_async(forward_backward, 
                                 (input_image_train[device_id]["image"], image,
                                  input_image_train[device_id]["label"], label, 
                                  losses_train[device_id], solvers[device_id])) 
                fb_results.append(res)
            for device_id in range(n_devices):
                fb_results[device_id].get()
                        
            # In-place Allreduce
            comm.allreduce()
            
            # Solvers update
            for device_id in range(n_devices):
                solvers[device_id].update()

            e = categorical_error(preds_train[-1].d, input_image_train[-1]["label"].d)
            monitor_loss.add(i*n_devices, losses_train[-1].d.copy())
            monitor_err.add(i*n_devices, e)
            monitor_time.add(i*n_devices)

    nn.save_parameters(os.path.join(
        args.model_save_path, 
        'params_%06d.h5' % (args.max_iter/ n_devices)))
            
if __name__ == '__main__':
    train()
