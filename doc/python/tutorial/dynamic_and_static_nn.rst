
Static vs Dynamic Neural Networks in NNabla
===========================================

.. code:: ipython2

    %matplotlib inline
    import nnabla as nn
    import nnabla.functions as F
    import nnabla.parametric_functions as PF
    import nnabla.solvers as S
    
    import numpy as np
    np.random.seed(0)
    
    GPU = 0


.. parsed-literal::

    2017-06-26 02:11:14,682 [nnabla][INFO]: Initializing CPU extension...


Loading dataset
---------------

.. code:: ipython2

    from tiny_digits import *
    digits = load_digits()
    data =data_iterator_tiny_digits(digits, batch_size=256, shuffle=True)
    img, label = data.next()
    print img.shape, label.shape


.. parsed-literal::

    2017-06-26 02:11:14,921 [nnabla][INFO]: DataSource with shuffle(True)
    2017-06-26 02:11:14,922 [nnabla][INFO]: Using DataSourceWithMemoryCache
    2017-06-26 02:11:14,922 [nnabla][INFO]: DataSource with shuffle(True)
    2017-06-26 02:11:14,923 [nnabla][INFO]: On-memory
    2017-06-26 02:11:14,923 [nnabla][INFO]: Using DataIterator


.. parsed-literal::

    (256, 1, 8, 8) (256, 1)


Network definition
------------------

.. code:: ipython2

    def cnn(x):
        """Unnecessarily Deep CNN.
        
        Args:
            
            x : Variable, shape (B, 3, 8, 8)
            
        Returns:
        
            y : Variable, shape (B, 10)
        """
        with nn.parameter_scope("cnn"):  # Parameter scope can be nested
            with nn.parameter_scope("conv1"):
                h = F.tanh(PF.batch_normalization(
                    PF.convolution(x, 64, (3, 3), pad=(1, 1))))
            for i in range(30):  # unnecessarily deep
                with nn.parameter_scope("conv{}".format(i + 2)):
                    h = F.tanh(PF.batch_normalization(
                        PF.convolution(h, 128, (3, 3), pad=(1, 1))))
            with nn.parameter_scope("conv_last"):
                h = F.tanh(PF.batch_normalization(
                    PF.convolution(h, 512, (3, 3), pad=(1, 1))))
                h = F.average_pooling(h, (2, 2))
            with nn.parameter_scope("fc"):
                h = F.tanh(PF.affine(h, 1024))
            with nn.parameter_scope("classifier"):
                y = PF.affine(h, 10)
        return y

Static computation graph
------------------------

.. code:: ipython2

    x = nn.Variable(img.shape)
    t = nn.Variable(label.shape)
    from nnabla.contrib.context import extension_context
    ctx_cuda = extension_context('cuda.cudnn', device_id=GPU)  # replace it 'cpu' if you do not have CUDA extension
    nn.set_default_context(ctx_cuda)
    y = cnn(x)
    l = F.mean(F.softmax_cross_entropy(y, t))


.. parsed-literal::

    2017-06-26 02:11:15,106 [nnabla][INFO]: Initializing CUDA extension...
    2017-06-26 02:11:15,273 [nnabla][INFO]: Initializing cuDNN extension...


Solver

.. code:: ipython2

    solver = S.Adam(alpha=1e-3)
    solver.set_parameters(nn.get_parameters())

Executing forwardprop.

.. code:: ipython2

    x.d, t.d = data.next()
    l.forward()

Backprop and update.

.. code:: ipython2

    solver.zero_grad()
    l.backward()
    solver.update()

Create data iterator

.. code:: ipython2

    loss = []
    def epoch_end_callback(epoch):
        global loss
        print "[", epoch, np.mean(loss), itr, "]",
        loss = []
    
    data = data_iterator_tiny_digits(digits, batch_size=256, shuffle=True)
    data.register_epoch_end_callback(epoch_end_callback)


.. parsed-literal::

    2017-06-26 02:11:16,536 [nnabla][INFO]: DataSource with shuffle(True)
    2017-06-26 02:11:16,539 [nnabla][INFO]: Using DataSourceWithMemoryCache
    2017-06-26 02:11:16,541 [nnabla][INFO]: DataSource with shuffle(True)
    2017-06-26 02:11:16,543 [nnabla][INFO]: On-memory
    2017-06-26 02:11:16,545 [nnabla][INFO]: Using DataIterator


Iterate updates.

.. code:: ipython2

    %%time
    for epoch in range(30):
        itr = 0
        while data.epoch == epoch:
            x.d, t.d = data.next()
            l.forward(clear_no_need_grad=True)
            solver.zero_grad()
            l.backward(clear_buffer=True)
            solver.update()
            loss.append(l.d.copy())
            itr += 1
    print ''


.. parsed-literal::

    [ 0 5.03612 7 ] [ 1 1.16284 6 ] [ 2 0.345992 6 ] [ 3 0.168834 6 ] [ 4 0.0944248 6 ] [ 5 0.0329602 6 ] [ 6 0.0205269 6 ] [ 7 0.0100916 6 ] [ 8 0.00503768 6 ] [ 9 0.00332624 6 ] [ 10 0.00252542 6 ] [ 11 0.00179686 6 ] [ 12 0.00157729 6 ] [ 13 0.00129931 6 ] [ 14 0.00109707 6 ] [ 15 0.000951522 6 ] [ 16 0.000890525 6 ] [ 17 0.000766493 6 ] [ 18 0.000762373 6 ] [ 19 0.000662311 6 ] [ 20 0.00061368 6 ] [ 21 0.000610432 6 ] [ 22 0.00050406 6 ] [ 23 0.000537643 6 ] [ 24 0.00047761 6 ] [ 25 0.000453521 6 ] [ 26 0.000412768 6 ] [ 27 0.000382283 6 ] [ 28 0.000384979 6 ] [ 29 0.000368561 6 ] 
    CPU times: user 9.7 s, sys: 6.98 s, total: 16.7 s
    Wall time: 16.6 s


.. code:: ipython2

    del y, l, solver

Dynamic computation graph
-------------------------

.. code:: ipython2

    nn.clear_parameters()

Forwardprop during graph building.

.. code:: ipython2

    x.d, t.d = data.next()
    with nn.auto_forward():
        x.data.cast(np.float32, ctx_cuda)
        t.data.cast(np.int32, ctx_cuda)
        d_y = cnn(x)
        d_l = F.mean(F.softmax_cross_entropy(d_y, t))

Backprop can be executed through dynamically built graph.

.. code:: ipython2

    solver = S.Adam(alpha=1e-3)
    solver.set_parameters(nn.get_parameters())
    solver.zero_grad()
    d_l.backward()
    solver.update()

.. code:: ipython2

    nn.clear_parameters()
    solver = S.Adam(alpha=1e-3)
    
    loss = []
    def epoch_end_callback(epoch):
        global loss
        print "[", epoch, np.mean(loss), itr, "]",
        loss = []
    data = data_iterator_tiny_digits(digits, batch_size=256, shuffle=True)
    data.register_epoch_end_callback(epoch_end_callback)


.. parsed-literal::

    2017-06-26 02:11:33,970 [nnabla][INFO]: DataSource with shuffle(True)
    2017-06-26 02:11:33,973 [nnabla][INFO]: Using DataSourceWithMemoryCache
    2017-06-26 02:11:33,975 [nnabla][INFO]: DataSource with shuffle(True)
    2017-06-26 02:11:33,977 [nnabla][INFO]: On-memory
    2017-06-26 02:11:33,979 [nnabla][INFO]: Using DataIterator


.. code:: ipython2

    %%time
    for epoch in range(30):
        itr = 0
        while data.epoch == epoch:
            x.d, t.d = data.next()
            with nn.auto_forward():
                x.data.cast(np.float32, ctx_cuda)
                t.data.cast(np.int32, ctx_cuda)
                d_y = cnn(x)
                d_l = F.mean(F.softmax_cross_entropy(d_y, t))
            solver.set_parameters(nn.get_parameters(), reset=False, retain_state=True) # Able to set dynamically.
            solver.zero_grad()
            d_l.backward(clear_buffer=True)
            solver.update()
            loss.append(d_l.d.copy())
            itr += 1
    print ''


.. parsed-literal::

    [ 0 4.78392 7 ] [ 1 5.25743 6 ] [ 2 0.81035 6 ] [ 3 0.322123 6 ] [ 4 0.166708 6 ] [ 5 0.0927392 6 ] [ 6 0.0614254 6 ] [ 7 0.0420595 6 ] [ 8 0.0283023 6 ] [ 9 0.0143313 6 ] [ 10 0.010226 6 ] [ 11 0.00725726 6 ] [ 12 0.00536173 6 ] [ 13 0.00411181 6 ] [ 14 0.00374438 6 ] [ 15 0.00300302 6 ] [ 16 0.00285863 6 ] [ 17 0.00216548 6 ] [ 18 0.00199987 6 ] [ 19 0.00195086 6 ] [ 20 0.00156022 6 ] [ 21 0.00177088 6 ] [ 22 0.0014853 6 ] [ 23 0.00136038 6 ] [ 24 0.00131281 6 ] [ 25 0.00122088 6 ] [ 26 0.0011147 6 ] [ 27 0.00100175 6 ] [ 28 0.000976665 6 ] [ 29 0.00102304 6 ] 
    CPU times: user 10.5 s, sys: 6.44 s, total: 17 s
    Wall time: 16.9 s


.. code:: ipython2

    del d_l, d_y, solver
