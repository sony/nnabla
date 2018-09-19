
Debugging
=========

Deep neural networks are going deeper and deeper every year, requiring
more components in the networks. Such complexity often misleads us to
mal-configure the networks that can turn out be critical. Even if we
correctly configure a neural network as desired, we may still want to
find out its performance bottleneck, e.g., from which layer(s) the
computational bottleneck comes from.

In this debugging tutorial, we introduce three techniques to deal with
such cases:

1. ``visit`` method of a variable
2. simple graph viewer
3. profiling utils

We will go over each technique, but first prepare the following
reference model.

.. code:: python

    import numpy as np
    import nnabla as nn
    import nnabla.logger as logger
    import nnabla.functions as F
    import nnabla.parametric_functions as PF
    import nnabla.solvers as S
    
    def block(x, maps, test=False, name="block"):
        h = x
        with nn.parameter_scope(name):
            with nn.parameter_scope("in-block-1"):
                h = PF.convolution(h, maps, kernel=(3, 3), pad=(1, 1), with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            with nn.parameter_scope("in-block-2"):
                h = PF.convolution(h, maps // 2, kernel=(3, 3), pad=(1, 1), with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            with nn.parameter_scope("in-block-3"):
                h = PF.convolution(h, maps, kernel=(3, 3), pad=(1, 1), with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                
            if h.shape[1] != x.shape[1]:
                with nn.parameter_scope("skip"):
                    s = PF.convolution(x, maps, kernel=(3, 3), pad=(1, 1), with_bias=False)
                    s = PF.batch_normalization(s, batch_stat=not test)
    
        return F.relu(h + s)
    
    def network(x, maps=16, test=False):
        h = x
        h = PF.convolution(h, maps, kernel=(3, 3), pad=(1, 1), name="first-conv", with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test, name="first-bn")
        h = F.relu(h)
        for l in range(4):
            h = block(h, maps * 2 ** (l + 1), name="block-{}".format(l))
            h = F.max_pooling(h, (2, 2))
        h = F.average_pooling(h, h.shape[2:])
        pred = PF.affine(h, 100, name="pred")
        return pred      

Visit Method
------------

Visit method of a variable takes either lambda, function, callable
object as an argument and calls it over all NNabla functions where the
variable can traverse in the forward order. It is easier to see the
usage than expalined.

First of all, define the callable class.

.. code:: python

    class PrintFunc(object):
        def __call__(self, nnabla_func):
            print("==========")
            print(nnabla_func.info.type_name)
            print(nnabla_func.inputs)
            print(nnabla_func.outputs)
            print(nnabla_func.info.args)

This callable object takes a NNabla function, e.g., convolution, relu,
etc., so a user can get information of that function.

.. code:: python

    nn.clear_parameters()  # this call is just in case to do the following code again
    
    x = nn.Variable([4, 3, 128, 128])
    pred = network(x)
    pred.visit(PrintFunc())

Simple Graph Viewer
-------------------

Visit method is very useful for getting information about each function
used in a graph, but it is hard to see the details of the whole network
structure, e.g., which variable is connected to which variable. So we
have a graph viewer that visually shows the whole structure of network,
enabling us to debug more efficiently. Using this graph viewer is
straightforward, as shown in the following code:

.. code:: python

    # Create graph again just in case
    
    nn.clear_parameters()  # call this in case you want to run the following code agian
    
    x = nn.Variable([4, 3, 128, 128])
    pred = network(x)

.. code:: python

    import nnabla.experimental.viewers as V
    
    graph = V.SimpleGraph(verbose=False)
    graph.view(pred)

If one would like to see more detailed information as in ``visit``
method case, change verbose option to ``True``.

.. code:: python

    graph = V.SimpleGraph(verbose=True)
    graph.view(pred)

Now one can see detailed information!

Note that this viewer is mainly for NNabla users who want to write codes
in python, so for those who like to see more beautiful network and play
with that, please use Neural Network Console and visit
https://dl.sony.com/.

Profiling utils
---------------

Basically, this feature is **for developers** who want to know the whole
stats in speed and which functions could be bottlenecks. NNabla provides
a simple profiling tool. Once a network is prepared, one better to have
other components to train the network like a loss function and solvers.

First, to create the profile and see the results, run the following
codes.

.. code:: python

    # Create graph again just in case
    
    nn.clear_parameters()  # call this in case you want to run the following code agian
    
    # Context
    from nnabla.ext_utils import get_extension_context
    device = "cudnn"
    ctx = get_extension_context(device)
    nn.set_default_context(ctx)
    
    # Network
    x = nn.Variable([4, 3, 128, 128])
    t = nn.Variable([4, 1])
    pred = network(x)
    loss = F.mean(F.softmax_cross_entropy(pred, t))
    
    # Solver
    solver = S.Momentum()
    solver.set_parameters(nn.get_parameters())
    
    # Profiler
    from nnabla.utils.profiler import GraphProfiler
    B = GraphProfiler(loss, solver=solver, device_id=0, ext_name=device, n_run=100)
    B.run()
    print("Profile finished.")
    
    # Report
    from nnabla.utils.profiler import GraphProfilerCsvWriter
    with open("./profile.csv", "w") as f:
        writer = GraphProfilerCsvWriter(B, file=f)
        writer.write()
    print("Report is prepared.")
