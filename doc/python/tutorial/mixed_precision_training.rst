
Mixed Precision Training
========================

Introduction
------------

Traditionally, for training a neural network, we used to use ``FP32``
for weights and activations; however computation costs for trainig a
neural network rapidly increase over years as the success of deep
learning and the growing size of a neural nework. It indiates that we
need to spend much more time for training a huge size of a neural
network while we would like to do lots of trials before a product
launch. To address this problem, companys (e.g., NVIDIA) introduced an
accelarator for speeding up computation. For example, NVIDIA Volta has
`Tensor
Cores <https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/>`__
to speed up computation.

However, it uses ``FP16`` weights, activations, gradients, and the range
of ``FP16`` is very limited when compared to that of ``FP32``, meaning
that sometimes (or often) values of gradients overflow and/or underflow,
which affects the performance of a neural network or makes it collapse
during training.

Mixed precision training is one of the algorithms to circumvent that
problem while maintaining the same results that we could obtain with
``FP32`` networks. It is well-described in `The Training with Mixed
Precision User
Guide <https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html>`__
and `Mixed Precision Training <https://arxiv.org/abs/1710.03740>`__.

This tutorial explains how to do the mixed precision training in NNabla
step-by-step.

Step-by-Step Instruction
------------------------

Basically, the mixed precision training are composed of three parts.

1. Use the accelarator for computation (here we assume Tensor Cores)
2. Use loss scaling to prevent underflow
3. Use dynamic loss caling to prevent overflow/underflow

In NNabla, we can do the correspondinces as follows.

1. Use Tensor Cores
~~~~~~~~~~~~~~~~~~~

.. code:: python

    ctx = get_extension_context("cudnn", type_config="half")

2. Use loss scaling to prevent underflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    loss_scale = 8
    loss.backward(loss_scale)
    solver.scale_grad(1. / loss_scale)  # do some graident clipping, etc. after this
    solver.update()

3. Use dynamic loss scaling to prevent overflow/underflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    loss_scale = 8
    scaling_factor = 2
    counter = 0
    interval = 2000
    ...
    loss.backward(loss_scale, ...)
    ...
    if solver.check_inf_or_nan_grad():
        loss_scale /= scaling_factor
        counter = 0
    else:
        solver.scale_grad(1. / loss_scale) # do some graident clipping, etc. after this
        solver.update()
        if counter > interval:
            loss_scale *= scaling_factor
            counter = 0
        counter += 1

**Note** that currently the procedures of 2nd (Use loss scaling to
prevent underflow) and 3rd (Use loss scaling to prevent overflow) are
exprimental, and we are now trying to speed up the mixed precision
training, so API might change for future use, especially 3rd.

All-in-one Instruction
----------------------

In the previous step-by-step example, the 3rd step is lengthy in a
training loop, thus we can write a wrapper class like the following.

.. code:: python

    class DynamicLossScalingUpdater(object):
        '''Dynamic Loss Scaling Updater for the mixed precision training.
    
        Args:
            solver (:obj:`nnabla.solvers.Solver`): Solver object. E.g., Momentum or Adam.
            loss (:obj:`nnabla.Variable`): Loss variable from which the forward and the backward is called.
            data_feeder (callable :obj:`object`, function, or lambda): Data feeder
            scale (:obj:`float`): Loss scale constant. This is dynamically changing during training.
            scaling_factor (:obj:`float`): Scaling factor for the dynamic loss scaling.
            N (:obj:`int`): Interval, the number of iterations in training for increasing `loss scale` by `scaling_factor`.
            clear_buffer (:obj:`bool`): Clears the no longer referenced variables during backpropagation to save memory.
            accum_grad (:obj:`int`): Number of accumulation of gradients. Update method of the `solver` is called after the `accum_grad` number of the forward and backward is called.
            weight_decay (:obj:`float`): Decay constant. Default is `None`, not applying the weight decay.
            comm (:obj:`nnabla.communicators.Communicator`): Communicator when to do distributed training. Defalt is :obj:`None`.
            grads (:obj:`list` of :obj:`nnabla._nd_array.NdArray`): The list of gradients to be exchanged when to do distributed training. Defalt is the empty :obj:`list`.
    
        Attributes:
            solver (:obj:`nnabla.solvers.Solver`): Solver object. E.g., Momentum or Adam.
            loss (:obj:`nnabla.Variable`): Loss variable from which the forward and the backward is called.
            data_feeder (callable :obj:`object`, function, lambda): Data feeder
            scale (:obj:`float`): Loss scale constant. This is dynamically changing during training.
            scaling_factor (:obj:`float`): Scaling factor for the dynamic loss scaling.
            N (:obj:`int`): Interval, the number of iterations in training for increasing `loss scale` by `scaling_factor`.
            clear_buffer (:obj:`bool`): Clears the no longer referenced variables during backpropagation to save memory.
            accum_grad (:obj:`int`): Number of accumulation of gradients. Update method of the `solver` is called after the `accum_grad` number of the forward and backward is called.
            weight_decay (:obj:`float`): Decay constant. Default is `None`, not applying the weight decay.
            comm (:obj:`nnabla.communicators.Communicator`): Communicator when to do distributed training.
            grads (:obj:`list` of :obj:`nnabla._nd_array.NdArray`): The list of gradients to be exchanged when to do distributed training.
    
        Example:
    
            .. code-block:: python
                solver = <Solver>
                loss = <Loss Variable of Network>
                data_feeder = <DataFeeder>
    
                updater = DynamicLossScalingUpdater(solver, loss, data_feeder)
    
                # Training iteration
                for itr in range(max_iter):
                    # Call solver.zero_grad, data_feeder, loss.forward, loss.backward
                    # and solver.update with the dynamic loss scaling.
                    updater.update()
    
        Reference:
        
            https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#scalefactor
        
        '''
    
        def __init__(self, solver, loss, data_feeder=lambda x: x, 
                      scale=8.0, scaling_factor=2.0, N=2000, clear_buffer=True,
                      accum_grad=1, weight_decay=None, 
                      comm=None,
                      grads=[]):
            self.solver = solver
            self.loss = loss
            self.data_feeder = data_feeder
            self.scale = scale
            self.scaling_factor = scaling_factor
            self.N = N
            self.clear_buffer = clear_buffer
            self.accum_grad = accum_grad
            self.weight_decay = weight_decay
            self.comm = comm
            self.grads = grads
            self._counter = 0
            self._recursive_count = 0
            self._max_recursive_count = 100
    
        def update(self):
            """Monolithic update method.
    
            This method calls the following methods with the dynamic loss scaling.
    
            1. solver.zerograd
            2. feed data
            3. loss.forward
            4. loss.backward
            5. comm.all_reduce (if it is specified)
            6. solver.update
            
            """
    
            # Initialize gradients.
            self.solver.zero_grad()
    
            # Forward and backward
            for _ in range(self.accum_grad):
                # feed data
                self.data_feeder()
                
                # forward
                self.loss.forward(clear_no_need_grad=self.clear_buffer)
    
                # backward with scale
                self.loss.backward(self.scale, clear_buffer=self.clear_buffer)
    
            # AllReduce
            if self.comm and len(self.grads) != 0:
                self.comm.all_reduce(self.grads, division=False, inplace=False)
    
            # Check Inf/NaN in grads
            if self.solver.check_inf_or_nan_grad():
                self.scale /= self.scaling_factor
                self._counter = 0
    
                # Recursively call udpate function until no inf nor nan.
                self._recursive_count += 1
                if self._recursive_count > self._max_recursive_count:
                    self._recursive_count = 0
                    return  # skip
                return self.update()
            self._recursive_count = 0
    
            # Rescale grads
            self.solver.scale_grad(1. / self.scale)
    
            # Do some graident clipping, etc.
            if self.weight_decay is not None:
                self.solver.weight_decay(self.weight_decay)
            
            # Update
            self.solver.update()
            if self._counter > self.N:
                self.scale *= self.scaling_factor
                self._counter = 0
            self._counter += 1

Then, call the update method in a training loop:

.. code:: python

    from nnabla.experimental.mixed_precision_training import DynamicLossScalingUpdater
    
    solver = <Solver>
    loss = <Loss Variable of Network>
    data_feeder = <DataFeeder>
    
    updater = DynamicLossScalingUpdater(solver, loss, data_feeder)
    
    # Training iteration
    for itr in range(max_iter):
        # Call solver.zero_grad, data_feeder, loss.forward, loss.backward
        # and solver.update with the dynamic loss scaling.
        updater.update()

Notice
------

In the mixed-precision training, the followings are premise:

1. Solver contains ``FP16`` weights and the ``FP32`` copy of weights.
   Solvers in NNabla hold ``FP32`` weights and weight gradients and cast
   it to ``FP16`` weights in forward pass and to ``FP16`` weight
   gradients in backward pass if one sets ``type_config="half"``.

2. Reductions should be left in ``FP32``, for examples, the statistics
   (mean and variance) computed by the batch-normalization, Mean, Sum,
   SoftMax, SoftMaxCrossEntropy, etc. (see `The Training with Mixed
   Precision User
   Guide <https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html>`__).
   In NNabla, these functions are automatically fallbacked to use
   ``FP32``.
