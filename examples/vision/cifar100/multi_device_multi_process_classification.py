import os
import time

from args import get_args
from cifar100_data import data_iterator_cifar100
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

def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()
    
def cifar100_resnet23_prediction(image, 
                                ctx, test=False):
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
    nmaps = 384
    ncls = 100
    
    # Conv -> BN -> Relu
    with nn.context_scope(ctx):
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
            calc_uniform_lim_glorot(int(np.prod(h.shape[1:])), ncls, kernel=(1, 1)), rng=rng)
        pred = PF.affine(h, ncls, w_init=w_init)

    return pred
        
def cifar100_resnet32_loss(pred, label):
    loss = F.mean(F.softmax_cross_entropy(pred, label))
    return loss

def train():
    """
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
    # Parse args
    args = get_args()
    n_train_samples = 50000
    bs_valid = args.batch_size

    # Communicator and Context
    extension_module = "cuda.cudnn"
    ctx = extension_context(extension_module)
    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    device_id = mpi_rank
    ctx = extension_context(extension_module, device_id=device_id)
    
    # Create training graphs
    test = False
    image_train = nn.Variable((args.batch_size, 3, 32, 32))
    label_train = nn.Variable((args.batch_size, 1))
    pred_train = cifar100_resnet23_prediction(
        image_train, ctx, test)
    loss_train = cifar100_resnet32_loss(pred_train, label_train)
    input_image_train = {"image": image_train, "label": label_train}
    
    # add parameters to communicator
    comm.add_context_and_parameters((ctx, nn.get_parameters()))
        
    # Create validation graph
    test = True
    image_valid = nn.Variable((bs_valid, 3, 32, 32))
    pred_valid = cifar100_resnet23_prediction(
        image_valid, ctx, test)
    input_image_valid = {"image": image_valid}
    
    # Solvers
    solver = S.Adam()
    solver.set_parameters(nn.get_parameters())
    base_lr = args.learning_rate
    warmup_iter = int(1. * n_train_samples / args.batch_size / n_devices) * args.warmup_epoch
    warmup_slope = 1. * n_devices / warmup_iter
    
    # Create monitor
    from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=10)
    monitor_err = MonitorSeries("Training error", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=100)
    monitor_verr = MonitorSeries("Test error", monitor, interval=10)
    with data_iterator_cifar100(args.batch_size, True) as tdata, \
                    data_iterator_cifar100(bs_valid, False) as vdata:
        # Training-loop
        for i in range(int(args.max_iter / n_devices)):
            # Validation
            if mpi_rank == 0:
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

            # Forward/Zerograd/Backward
            image, label = tdata.next()
            input_image_train["image"].d = image
            input_image_train["label"].d = label
            loss_train.forward()
            solver.zero_grad()
            loss_train.backward()

            # In-place Allreduce
            comm.allreduce(division=True)
            
            # Solvers update
            solver.update()
            
            # Linear Warmup
            if i < warmup_iter:
                lr = base_lr * n_devices * warmup_slope * i
                solver.set_learning_rate(lr)
            else:
                lr = base_lr * n_devices
                solver.set_learning_rate(lr)

            if mpi_rank == 0:
                e = categorical_error(pred_train.d, input_image_train["label"].d)
                monitor_loss.add(i*n_devices, loss_train.d.copy())
                monitor_err.add(i*n_devices, e)
                monitor_time.add(i*n_devices)
    if mpi_rank == 0:
        nn.save_parameters(os.path.join(
            args.model_save_path, 
            'params_%06d.h5' % (args.max_iter/ n_devices)))
            
if __name__ == '__main__':
    """
    Call this script with `mpirun` or `mpiexec`
    
    $ mpirun -n 2 python multi_device_multi_process.py
     
    """
    train()
