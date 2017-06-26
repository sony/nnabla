
NNabla Python API Demonstration Tutorial
========================================

Let us import nnabla first, and some additional useful tools.

.. code:: ipython2

    import nnabla as nn  # Abbreviate as nn for convenience.
    
    import numpy as np
    %matplotlib inline
    import matplotlib.pyplot as plt


.. parsed-literal::

    2017-06-26 11:58:08,046 [nnabla][INFO]: Initializing CPU extension...


NdArray
-------

NdArray is a data container of a multi-dimensional array. NdArray is
device (e.g. CPU, CUDA) and type (e.g. uint8, float32) agnostic, in
which both type and device are implicitly casted or transferred when it
is used. Below, you create a NdArray with a shape of ``(2, 3, 4)``.

.. code:: ipython2

    a = nn.NdArray((2, 3, 4))

You can see the values held inside ``a`` by the following. The values
are not initialized, and are created as float32 by default.

.. code:: ipython2

    print a.data


.. parsed-literal::

    [[[ -2.61849964e+38   4.57692104e-41   9.55322336e-38   0.00000000e+00]
      [             nan   4.57692104e-41   0.00000000e+00   0.00000000e+00]
      [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]]
    
     [[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
      [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
      [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]]]


The accessor ``.data`` returns a reference to the values of NdArray as
``numpy.ndarray``. You can modify these by using the Numpy API as
follows.

.. code:: ipython2

    print '[Substituting random values]'
    a.data = np.random.randn(*a.shape)
    print a.data
    print '[Slicing]'
    a.data[0, :, ::2] = 0
    print a.data


.. parsed-literal::

    [Substituting random values]
    [[[ 0.40705302  0.28001803 -0.20453408  0.13093373]
      [ 0.90947026 -0.17857344  0.32959503 -0.20124471]
      [ 0.68288428 -0.63338071 -3.12650847  1.35729706]]
    
     [[-1.15608287 -0.17183913  2.01262951 -0.59164506]
      [-1.60144043  0.21257734 -0.20791137 -0.52922416]
      [ 3.31629848 -0.59307563 -1.67073536 -0.13047314]]]
    [Slicing]
    [[[ 0.          0.28001803  0.          0.13093373]
      [ 0.         -0.17857344  0.         -0.20124471]
      [ 0.         -0.63338071  0.          1.35729706]]
    
     [[-1.15608287 -0.17183913  2.01262951 -0.59164506]
      [-1.60144043  0.21257734 -0.20791137 -0.52922416]
      [ 3.31629848 -0.59307563 -1.67073536 -0.13047314]]]


Note that the above operation is all done in the host device (CPU).
NdArray provides more efficient functions in case you want to fill all
values with a constant, ``.zero`` and ``.fill``. They are lazily
evaluated when the data is requested (when neural network computation
requests the data, or when numpy array is requested by Python) The
filling operation is executed within a specific device (e.g. CUDA GPU),
and more efficient if you specify the device setting, which we explain
later.

.. code:: ipython2

    a.fill(1)  # Filling all values with one.
    print a.data


.. parsed-literal::

    [[[ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]]
    
     [[ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]]]


You can create an NdArray instance directly from a Numpy array object.

.. code:: ipython2

    b = nn.NdArray.from_numpy_array(np.ones(a.shape))
    print b.data


.. parsed-literal::

    [[[ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]]
    
     [[ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]]]


NdArray is used in Variable class, as well as NNabla's imperative
computation of neural networks. We describe them in the later sections.

Variable
--------

Variable class is used when you construct a neural network. The neural
network can be described as a graph in which an edge represents a
function (a.k.a operator and layer) which defines operation of a minimum
unit of computation, and a node represents a variable which holds
input/output values of a function (Function class is explained later).
The graph is called "Computation Graph".

In NNabla, a Variable, a node of a computation graph, holds two
``NdArray``\ s, one for storing the input or output values of a function
during forward propagation (executing computation graph in the forward
order), while another for storing the backward error signal (gradient)
during backward propagation (executing computation graph in backward
order to propagate error signals down to parameters (weights) of neural
networks). The first one is called ``data``, the second is ``grad`` in
NNabla.

The following line creates a Variable instance with a shape of (2, 3,
4). It has ``data`` and ``grad`` as ``NdArray``. The flag ``need_grad``
is used to omit unnecessary gradient computation during backprop if set
to False.

.. code:: ipython2

    x = nn.Variable([2, 3, 4], need_grad=True)
    print 'x.data:', x.data
    print 'x.grad:', x.grad


.. parsed-literal::

    x.data: <NdArray((2, 3, 4)) at 0x7f96c2ca3120>
    x.grad: <NdArray((2, 3, 4)) at 0x7f96c2ca3120>


You can get the shape by:

.. code:: ipython2

    x.shape




.. parsed-literal::

    (2, 3, 4)



Since both ``data`` and ``grad`` are ``NdArray``, you can get a
reference to its values as NdArray with the ``.data`` accessor, but also
it can be refered by ``.d`` or ``.g`` property for ``data`` and ``grad``
respectively.

.. code:: ipython2

    print 'x.data'
    print x.d
    x.d = 1.2345  # To avoid NaN
    assert np.all(x.d == x.data.data), 'd: {} != {}'.format(x.d, x.data.data)
    print 'x.grad'
    print x.g
    x.g = 1.2345  # To avoid NaN
    assert np.all(x.g == x.grad.data), 'g: {} != {}'.format(x.g, x.grad.data)
    
    # Zeroing grad values
    x.grad.zero()
    print 'x.grad (after `.zero()`)'
    print x.g


.. parsed-literal::

    x.data
    [[[ -2.61862296e+38   4.57692104e-41  -2.61862296e+38   4.57692104e-41]
      [  3.44276208e-37   0.00000000e+00   3.44276208e-37   0.00000000e+00]
      [  1.69275966e+22   4.80112800e+30   1.21230330e+25   7.22962302e+31]]
    
     [[  1.10471027e-32   4.63080422e+27   1.74678549e+25   1.03958307e+21]
      [  1.89904644e+28   2.20458068e-10   1.12893616e+27   6.38286091e-10]
      [  5.44217996e+22   2.07717912e-10   9.72737793e+02   2.18638954e-10]]]
    x.grad
    [[[ -2.61845097e+38   4.57692104e-41  -2.61845097e+38   4.57692104e-41]
      [             nan              nan   0.00000000e+00   0.00000000e+00]
      [  0.00000000e+00   0.00000000e+00   1.32472255e-38   0.00000000e+00]]
    
     [[  1.96181785e-44   0.00000000e+00   9.12583854e-38   0.00000000e+00]
      [ -3.75903959e+34   4.57692104e-41              nan   4.57692104e-41]
      [ -6.27068418e+37   4.57692104e-41   0.00000000e+00   0.00000000e+00]]]
    x.grad (after `.zero()`)
    [[[ 0.  0.  0.  0.]
      [ 0.  0.  0.  0.]
      [ 0.  0.  0.  0.]]
    
     [[ 0.  0.  0.  0.]
      [ 0.  0.  0.  0.]
      [ 0.  0.  0.  0.]]]


Like ``NdArray``, a ``Variable`` can also be created from Numpy
array(s).

.. code:: ipython2

    x2 = nn.Variable.from_numpy_array(np.ones((3,)), need_grad=True)
    print x2
    print x2.d
    x3 = nn.Variable.from_numpy_array(np.ones((3,)), np.zeros((3,)), need_grad=True)
    print x3
    print x3.d
    print x3.g


.. parsed-literal::

    <Variable((3,), need_grad=True) at 0x7f96c2cabbb0>
    [ 1.  1.  1.]
    <Variable((3,), need_grad=True) at 0x7f96c2cab4c8>
    [ 1.  1.  1.]
    [ 0.  0.  0.]


Besides storing values of a computation graph, pointing a parent edge
(function) to trace the computation graph is an important role. Here
``x`` doesn't have any connection. Therefore, the ``.parent`` property
returns None.

.. code:: ipython2

    print x.parent


.. parsed-literal::

    None


Function
--------

A function defines a operation block of a computation graph as we
described above. The module ``nnabla.functions`` offers various
functions (e.g. Convolution, Affine and ReLU). You can see the list of
functions available in the `API reference
guide <http://nnabla.readthedocs.io/en/latest/python/api/function.html#module-nnabla.functions>`__.

.. code:: ipython2

    import nnabla.functions as F

As an example, here you will defines a computation graph that computes
the element-wise Sigmoid function outputs for the input variable and
sums up all values into a scalar. (This is simple enough to explain how
it behaves but a meaningless example in the context of neural network
training. We will show you a neural network example later.)

.. code:: ipython2

    sigmoid_output = F.sigmoid(x)
    sum_output = F.reduce_sum(sigmoid_output)

The function API in ``nnabla.functions`` takes one (or several)
Variable(s) and arguments (if any), and returns one (or several) output
Variable(s). The ``.parent`` points to the function instance which
created it. Note that no computation occurs at this time since we just
define the graph. (This is the default behavior of NNabla computation
graph API. You can also fire actual computation during graph definition
which we call "Dynamic mode" (explained later)).

.. code:: ipython2

    print "sigmoid_output.parent.name:", sigmoid_output.parent.name
    print "x:", x
    print "sigmoid_output.parent.inputs refers to x:", sigmoid_output.parent.inputs


.. parsed-literal::

    sigmoid_output.parent.name: Sigmoid
    x: <Variable((2, 3, 4), need_grad=True) at 0x7f96c2cabc80>
    sigmoid_output.parent.inputs refers to x: [<Variable((2, 3, 4), need_grad=True) at 0x7f96c29ec390>]


.. code:: ipython2

    print "sum_output.parent.name:", sum_output.parent.name
    print "sigmoid_output:", sigmoid_output
    print "sum_output.parent.inputs refers to sigmoid_output:", sum_output.parent.inputs


.. parsed-literal::

    sum_output.parent.name: ReduceSum
    sigmoid_output: <Variable((2, 3, 4), need_grad=True) at 0x7f96c29ec188>
    sum_output.parent.inputs refers to sigmoid_output: [<Variable((2, 3, 4), need_grad=True) at 0x7f96c29ec3f8>]


The ``.forward()`` at a leaf Variable executes the forward pass
computation in the computation graph.

.. code:: ipython2

    sum_output.forward()
    print "CG output:", sum_output.d
    print "Reference:", np.sum(1.0 / (1.0 + np.exp(-x.d)))


.. parsed-literal::

    CG output: 18.5905208588
    Reference: 18.5905


The ``.backward()`` does the backward propagation through the graph.
Here we initialize the ``grad`` values as zero before backprop since the
NNabla backprop algorithm always accumulates the gradient in the root
variables.

.. code:: ipython2

    x.grad.zero()
    sum_output.backward()
    print "d sum_o / d sigmoid_o:"
    print sigmoid_output.g
    print "d sum_o / d x:"
    print x.g


.. parsed-literal::

    d sum_o / d sigmoid_o:
    [[[ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]]
    
     [[ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]
      [ 1.  1.  1.  1.]]]
    d sum_o / d x:
    [[[ 0.17459197  0.17459197  0.17459197  0.17459197]
      [ 0.17459197  0.17459197  0.17459197  0.17459197]
      [ 0.17459197  0.17459197  0.17459197  0.17459197]]
    
     [[ 0.17459197  0.17459197  0.17459197  0.17459197]
      [ 0.17459197  0.17459197  0.17459197  0.17459197]
      [ 0.17459197  0.17459197  0.17459197  0.17459197]]]


NNabla is developed by mainly focused on neural network training and
inference. Neural networks have parameters to be learned associated with
computation blocks such as Convolution, Affine (a.k.a. fully connected,
dense etc.). In NNabla, the learnable parameters are also represented as
``Variable`` objects. Just like input variables, those parameter
variables are also used by passing into ``Function``\ s. For example,
Affine function takes input, weights and biases as inputs.

.. code:: ipython2

    x = nn.Variable([5, 2])  # Input
    w = nn.Variable([2, 3], need_grad=True)  # Weights
    b = nn.Variable([3], need_grad=True)  # Biases
    affine_out = F.affine(x, w, b)  # Create a graph including only affine

The above example takes an input with B=5 (batchsize) and D=2
(dimensions) and maps it to D'=3 outputs, i.e. (B, D') output.

You may also notice that here you set ``need_grad=True`` only for
parameter variables (w and b). The x is a non-parameter variable and the
root of computation graph. Therefore, it doesn't requires gradient
computation. In this configuration, the gradient computation for x is
not executed in the first affine, which will omit the computation of
unnecessary backpropagation.

The next block sets data and initializes grad, then applies forward and
backward computation.

.. code:: ipython2

    # Set random input and parameters
    x.d = np.random.randn(*x.shape)
    w.d = np.random.randn(*w.shape)
    b.d = np.random.randn(*b.shape)
    # Initialize grad
    x.grad.zero()  # Just for showing gradients are not computed when need_grad=False (default).
    w.grad.zero()
    b.grad.zero()
    
    # Forward and backward
    affine_out.forward()
    affine_out.backward()
    # Note: Calling backward at non-scalar Variable propagates 1 as error message from all element of outputs. .

You can see that affine\_out holds an output of Affine.

.. code:: ipython2

    print 'F.affine'
    print affine_out.d
    print 'Reference'
    print np.dot(x.d, w.d) + b.d


.. parsed-literal::

    F.affine
    [[ 0.07710284 -2.41815495 -0.18701762]
     [-0.91420352 -2.75334573 -2.18264437]
     [ 0.73566234 -3.54556227 -0.21898538]
     [-2.35871768  1.07976413 -0.74461746]
     [-0.11192489 -2.43834972 -0.52358592]]
    Reference
    [[ 0.07710284 -2.41815495 -0.18701762]
     [-0.91420352 -2.75334573 -2.18264437]
     [ 0.73566234 -3.54556227 -0.21898541]
     [-2.35871768  1.07976413 -0.7446174 ]
     [-0.11192489 -2.43834972 -0.52358592]]


Also the gradients of weights and biases are computed as follows.

.. code:: ipython2

    print "dw"
    print w.g
    print "db"
    print b.g


.. parsed-literal::

    dw
    [[ 0.68349785  0.68349785  0.68349785]
     [ 0.15355726  0.15355726  0.15355726]]
    db
    [ 5.  5.  5.]


The gradient of ``x`` is not changed because ``need_grad`` is set as
False.

.. code:: ipython2

    print x.g


.. parsed-literal::

    [[ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]]


Parametric Function
-------------------

Considering parameters as inputs of ``Function`` enhances expressiveness
and flexibility of computation graphs. However, to define all parameters
for each learnable function is annoying for users to define a neural
network. In NNabla, trainable models are usually created by composing
functions that have optimizable parameters. These functions are called
"Parametric Functions". The Parametric Function API provides various
parametric functions and an interface for composing trainable models.

To use parametric functions, import:

.. code:: ipython2

    import nnabla.parametric_functions as PF

The function with optimizable parameter can be created as below.

.. code:: ipython2

    with nn.parameter_scope("affine1"):
        c1 = PF.affine(x, 3)

The first line creates a **parameter scope**. The second line then
applies ``PF.affine`` - an affine transform - to ``x``, and creates a
variable ``c1`` holding that result. The parameters are created and
initialized randomly at function call, and registered by a name
"affine1" using ``parameter_scope`` context. The registered parameters
can be gotten by ``nnabla.get_parameters()`` function.

.. code:: ipython2

    nn.get_parameters()




.. parsed-literal::

    OrderedDict([('affine1/affine/W',
                  <Variable((2, 3), need_grad=True) at 0x7f96c29ecb48>),
                 ('affine1/affine/b',
                  <Variable((3,), need_grad=True) at 0x7f96c29ecbb0>)])



The ``name=`` argument of any PF function creates the equivalent
parameter space to the above definition of ``PF.affine`` transformation
as below. It could save the space of your Python code. The
``nnabla.parametric_scope`` is more useful when you group multiple
parametric functions such as Convolution-BatchNormalization found in a
typical unit of CNNs.

.. code:: ipython2

    c1 = PF.affine(x, 3, name='affine1')
    nn.get_parameters()




.. parsed-literal::

    OrderedDict([('affine1/affine/W',
                  <Variable((2, 3), need_grad=True) at 0x7f96c29ecb48>),
                 ('affine1/affine/b',
                  <Variable((3,), need_grad=True) at 0x7f96c29ecbb0>)])



It is worth noting that the shapes of both outputs and parameter
variables (as you can see above) are automatically determined by only
providing the number output dimensions of affine transformation(=3).
That lets us to create a graph easier.

.. code:: ipython2

    c1.shape




.. parsed-literal::

    (5, 3)



Parameter scope can be nested as follows (although a meaningless
example).

.. code:: ipython2

    with nn.parameter_scope('foo'):
        h = PF.affine(x, 3)
        with nn.parameter_scope('bar'):
            h = PF.affine(h, 4)

This creates the following.

.. code:: ipython2

    nn.get_parameters()




.. parsed-literal::

    OrderedDict([('affine1/affine/W',
                  <Variable((2, 3), need_grad=True) at 0x7f96c29ecb48>),
                 ('affine1/affine/b',
                  <Variable((3,), need_grad=True) at 0x7f96c29ecbb0>),
                 ('foo/affine/W',
                  <Variable((2, 3), need_grad=True) at 0x7f96c29ecef0>),
                 ('foo/affine/b',
                  <Variable((3,), need_grad=True) at 0x7f96c29ecf58>),
                 ('foo/bar/affine/W',
                  <Variable((3, 4), need_grad=True) at 0x7f96c098b0b8>),
                 ('foo/bar/affine/b',
                  <Variable((4,), need_grad=True) at 0x7f96c098b120>)])



Also, ``get_parameters()`` can be used in ``parameter_scope``. For
example:

.. code:: ipython2

    with nn.parameter_scope("foo"):
        print nn.get_parameters()


.. parsed-literal::

    OrderedDict([('affine/W', <Variable((2, 3), need_grad=True) at 0x7f96c29ecef0>), ('affine/b', <Variable((3,), need_grad=True) at 0x7f96c29ecf58>), ('bar/affine/W', <Variable((3, 4), need_grad=True) at 0x7f96c098b0b8>), ('bar/affine/b', <Variable((4,), need_grad=True) at 0x7f96c098b120>)])


``nnabla.clear_parameters()`` can be used to delete registered
parameters under the scope.

.. code:: ipython2

    with nn.parameter_scope("foo"):
        nn.clear_parameters()
    print nn.get_parameters()


.. parsed-literal::

    OrderedDict([('affine1/affine/W', <Variable((2, 3), need_grad=True) at 0x7f96c29ecb48>), ('affine1/affine/b', <Variable((3,), need_grad=True) at 0x7f96c29ecbb0>)])


MLP Example For Explanation
---------------------------

The following block creates a computation graph to predict one
dimensional output from two dimensional inputs by 2 layered fully
connected neural network (multi-layered perceptron).

.. code:: ipython2

    nn.clear_parameters()
    batchsize = 16
    x = nn.Variable([batchsize, 2])
    with nn.parameter_scope("fc1"):
        h = F.tanh(PF.affine(x, 512))
    with nn.parameter_scope("fc2"):
        y = PF.affine(h, 1)
    print "Shapes:", h.shape, y.shape


.. parsed-literal::

    Shapes: (16, 512) (16, 1)


This will create the following parameter variables.

.. code:: ipython2

    nn.get_parameters()




.. parsed-literal::

    OrderedDict([('fc1/affine/W',
                  <Variable((2, 512), need_grad=True) at 0x7f96c098b390>),
                 ('fc1/affine/b',
                  <Variable((512,), need_grad=True) at 0x7f96c098b328>),
                 ('fc2/affine/W',
                  <Variable((512, 1), need_grad=True) at 0x7f96c098b050>),
                 ('fc2/affine/b',
                  <Variable((1,), need_grad=True) at 0x7f96c098b4c8>)])



As we described above, you can execute the forward pass by calling
forward method at the terminal variable.

.. code:: ipython2

    x.d = np.random.randn(*x.shape)  # Set random input
    y.forward()
    print y.d


.. parsed-literal::

    [[-0.06116363]
     [-0.02471643]
     [-0.01327007]
     [-0.07057016]
     [ 0.07050993]
     [ 0.01684903]
     [ 0.05290569]
     [ 0.06554788]
     [ 0.01608899]
     [ 0.00639781]
     [-0.02477875]
     [-0.04054631]
     [ 0.00025999]
     [ 0.02205839]
     [ 0.01139119]
     [-0.0065173 ]]


Training a neural networks needs a loss value to be minimized by
gradient descent with backpop. In NNabla, loss function is also a just a
function, and packaged in the functions module.

.. code:: ipython2

    # Variable for label
    label = nn.Variable([batchsize, 1])
    # Set loss
    loss = F.reduce_mean(F.squared_error(y, label))
    
    # Execute forward pass.
    label.d = np.random.randn(*label.shape)  # Randomly generate labels
    loss.forward()
    print loss.d


.. parsed-literal::

    0.656837761402


As you've seen above, NNabla backprop accumulates the gradients at the
root variables. You have to initialize the grad of the parameter
variables before backprop (We will show you the more easiest way with
"Solver" API).

.. code:: ipython2

    # Collect all parameter variables and init grad.
    for name, param in nn.get_parameters().items():
        param.grad.zero()
    # Grdients are accumulated to grad of params.
    loss.backward()

Imperative Mode
---------------

After performing backprop, gradients are held in parameter variable
grads. The next block will update the parameters with vanilla gradient
descent.

.. code:: ipython2

    for name, param in nn.get_parameters().items():
        F.sub2(param.data, F.mul_scalar(param.grad, 0.001), outputs=[param.data])  # 0.001 as learning rate

The above computation is an example of NNabla's "Imperative Mode" of
execution of neural networks. If any input of a function takes as an
``NdArray``, the function computation will be fired immediately, and
returns NdArray(s) as output(s). Hence, the "Imperative mode" doesn't
create a computation graph, and can be used like Numpy with device
acceleration (if CUDA etc is enabled). Parametric functions can also be
used with NdArray input(s). The following block demonstrates a simple
imperative execution example.

.. code:: ipython2

    # A simple example of imperative mode.
    xi = nn.NdArray((2, 2))
    xi.data = np.arange(4).reshape(2, 2) - 1
    yi = F.relu(xi)
    print xi.data
    print yi.data


.. parsed-literal::

    [[-1.  0.]
     [ 1.  2.]]
    [[ 0.  0.]
     [ 1.  2.]]


Solver
------

NNabla provides stochastic gradient descent algorithms to optimize
parameters listed in the ``nnabla.solvers`` module. The parameter
updates demonstrated above can be replace with this Solver API, which is
easier and usually faster.

.. code:: ipython2

    from nnabla import solvers as S
    solver = S.Sgd(lr=0.00001)
    solver.set_parameters(nn.get_parameters())

.. code:: ipython2

    # Set random data
    x.d = np.random.randn(*x.shape)
    label.d = np.random.randn(*label.shape)
    
    # Forward
    loss.forward()

Just call the the following solver method to fill zero grad region, then
backprop

.. code:: ipython2

    solver.zero_grad()
    loss.backward()

The following block updates parameters with the Vanilla Sgd rule
(equivalent to the imperative example above).

.. code:: ipython2

    solver.update()

Toy Problem To Demonstrate Training
-----------------------------------

Here we define a regression problem that maps the two dimensional vector
into the length of it. The following function is the exact system of
this mapping.

.. code:: ipython2

    def vector2length(x):
        # x : [B, 2] where B is number of samples.
        return np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))

We visualize this mapping with the contour plot by matplotlib as
follows.

.. code:: ipython2

    # Data for plotting contour on a grid data.
    xs = np.linspace(-1, 1, 100)
    ys = np.linspace(-1, 1, 100)
    grid = np.meshgrid(xs, ys)
    X = grid[0].flatten()
    Y = grid[1].flatten()
    
    def plot_true():
        """Plotting contour of true mapping from a grid data created above."""
        plt.contourf(xs, ys, vector2length(np.hstack([X[:, None], Y[:, None]])).reshape(100, 100))
        plt.axis('equal')
        plt.colorbar()
        
    plot_true()



.. image:: python_api_files/python_api_95_0.png


We define a neural network which predicts the output of the unknown
system (although we know).

.. code:: ipython2

    def length_mlp(x):
        h = x
        for i, hnum in enumerate([4, 8, 4, 2]):
            h = F.tanh(PF.affine(h, hnum, name="fc{}".format(i)))
        y = PF.affine(h, 1, name='fc')
        return y

.. code:: ipython2

    nn.clear_parameters()
    batchsize = 100
    x = nn.Variable([batchsize, 2])
    y = length_mlp(x)
    label = nn.Variable([batchsize, 1])
    loss = F.reduce_mean(F.squared_error(y, label))

We created a 5 layers deep MLP using for-loop. Note that only 3 lines of
the code pottentially create infinitely deep neural networks. The next
block adds helper functions to visialize the learned function.

.. code:: ipython2

    def predict(inp):
        ret = []
        for i in range(0, inp.shape[0], x.shape[0]):
            xx = inp[i:i + x.shape[0]]
            # Imperative execution
            xi = nn.NdArray.from_numpy_array(xx)
            yi = length_mlp(xi)
            ret.append(yi.data.copy())
        return np.vstack(ret)
    
    def plot_prediction():
        plt.contourf(xs, ys, predict(np.hstack([X[:, None], Y[:, None]])).reshape(100, 100))
        plt.colorbar()
        plt.axis('equal')

Next we instantiate a solver object as follows. We use Adam optimizer
which is one of the most popular SGD algorithm used in the literature.

.. code:: ipython2

    from nnabla import solvers as S
    solver = S.Adam(alpha=0.01)
    solver.set_parameters(nn.get_parameters())

The following function generates data from the true system infinitely.

.. code:: ipython2

    def random_data_provider(n):
        x = np.random.uniform(-1, 1, size=(n, 2))
        y = vector2length(x)
        return x, y

In the next block, we run 2000 training steps (SGD updates).

.. code:: ipython2

    num_iter = 2000
    for i in range(num_iter):
        # Sample data and set them to input variables of training. 
        xx, ll = random_data_provider(batchsize)
        x.d = xx
        label.d = ll
        # Forward propagation given inputs.
        loss.forward(clear_no_need_grad=True)
        # Parameter gradients initialization and gradients computation by backprop.
        solver.zero_grad()
        loss.backward(clear_buffer=True)
        # Apply weight decay and update by Adam rule.
        solver.weight_decay(1e-6)
        solver.update()
        # Just print progress.
        if i % 100 == 0 or i == num_iter - 1:
            print "Loss@{:4d}: {}".format(i, loss.d)


.. parsed-literal::

    Loss@   0: 0.921138167381
    Loss@ 100: 0.0694101303816
    Loss@ 200: 0.00196460122243
    Loss@ 300: 0.000965791172348
    Loss@ 400: 0.00102860713378
    Loss@ 500: 0.000733042717911
    Loss@ 600: 0.000787019962445
    Loss@ 700: 0.000703430268914
    Loss@ 800: 0.000588007620536
    Loss@ 900: 0.00110029696953
    Loss@1000: 0.000527524854988
    Loss@1100: 0.000469014747068
    Loss@1200: 0.000429775944212
    Loss@1300: 0.00113972637337
    Loss@1400: 0.00045635754941
    Loss@1500: 0.000435676192865
    Loss@1600: 0.000812261539977
    Loss@1700: 0.000817801104859
    Loss@1800: 0.000387296662666
    Loss@1900: 0.000466798053822
    Loss@1999: 0.000753718253691


**Memory usage optimization**: You may notice that, in the above
updates, ``.forward()`` is called with the ``clear_no_need_grad=``
option, and ``.backward()`` is called with the ``clear_buffer=`` option.
Training of neural network in more realistic scenarios usually consumes
huge memory due to the nature of backpropagation algorithm, in which all
of the forward variable buffer ``data`` should be kept in order to
compute the gradient of a function. In a naive implementation, we keep
all the variable ``data`` and ``grad`` living until the ``NdArray``
objects are not referenced (i.e. the graph is deleted). The ``clear_*``
options in ``.forward()`` and ``.backward()`` enables to save memory
consumptions due to that by clearing (erasing) memory of ``data`` and
``grad`` when it is not referenced by any subsequent computation. (More
precisely speaking, it doesn't free memory actually. We use our memory
pool engine by default to avoid memory alloc/free overhead). The
unreferenced buffers can be re-used in subequent computation. See the
document of ``Variable`` for more details. Note that the following
``loss.forward(clear_buffer=True)`` clears ``data`` of any intermediate
variables. If you are interested in intermediate variables for some
purposes (e.g. debug, log), you can use the ``.persistent`` flag to
prevent clearing buffer of a specific ``Variable`` like below.

.. code:: ipython2

    loss.forward(clear_buffer=True)
    print "The prediction `y` is cleared because it's an intermedicate variable."
    print y.d.flatten()[:4]  # to save space show only 4 values
    y.persistent = True
    loss.forward(clear_buffer=True)
    print "The prediction `y` is kept by the persistent flag."
    print y.d.flatten()[:4]  # to save space show only 4 value


.. parsed-literal::

    The prediction `y` is cleared because it's an intermedicate variable.
    [ 0.00027053  0.00095906  0.00815325  0.00023331]
    The prediction `y` is kept by the persistent flag.
    [ 0.25289738  0.90301794  0.18608618  0.78919256]


We can confirm the prediction performs fairly well by looking at the
following visualization of the ground truth and prediction function.

.. code:: ipython2

    plt.subplot(121)
    plt.title("Ground truth")
    plot_true()
    plt.subplot(122)
    plt.title("Prediction")
    plot_prediction()



.. image:: python_api_files/python_api_110_0.png


You can save learned parameters by ``nnabla.save_parameters`` and load
by ``nnabla.load_parameters``.

.. code:: ipython2

    path_param = "param-vector2length.h5"
    nn.save_parameters(path_param)
    # Remove all once
    nn.clear_parameters()
    nn.get_parameters()


.. parsed-literal::

    2017-06-26 11:58:15,396 [nnabla][INFO]: Parameter save (hdf5): param-vector2length.h5




.. parsed-literal::

    OrderedDict()



.. code:: ipython2

    # Load again
    nn.load_parameters(path_param)
    print '\n'.join(map(str, nn.get_parameters().items()))


.. parsed-literal::

    2017-06-26 11:58:15,522 [nnabla][INFO]: Parameter load (<built-in function format>): param-vector2length.h5


.. parsed-literal::

    (u'fc0/affine/W', <Variable((2, 4), need_grad=True) at 0x7f96c098e808>)
    (u'fc0/affine/b', <Variable((4,), need_grad=True) at 0x7f96c098e3f8>)
    (u'fc1/affine/W', <Variable((4, 8), need_grad=True) at 0x7f96c098e1f0>)
    (u'fc1/affine/b', <Variable((8,), need_grad=True) at 0x7f96c098e328>)
    (u'fc2/affine/W', <Variable((8, 4), need_grad=True) at 0x7f96c098e0b8>)
    (u'fc2/affine/b', <Variable((4,), need_grad=True) at 0x7f96c098e258>)
    (u'fc3/affine/W', <Variable((4, 2), need_grad=True) at 0x7f96c098e120>)
    (u'fc3/affine/b', <Variable((2,), need_grad=True) at 0x7f96c098e188>)
    (u'fc/affine/W', <Variable((2, 1), need_grad=True) at 0x7f96c098e050>)
    (u'fc/affine/b', <Variable((1,), need_grad=True) at 0x7f96c06367a0>)


Both save and load functions can also be used in a parameter scope.

.. code:: ipython2

    with nn.parameter_scope('foo'):
        nn.load_parameters(path_param)
    print '\n'.join(map(str, nn.get_parameters().items()))


.. parsed-literal::

    2017-06-26 11:58:15,647 [nnabla][INFO]: Parameter load (<built-in function format>): param-vector2length.h5


.. parsed-literal::

    (u'fc0/affine/W', <Variable((2, 4), need_grad=True) at 0x7f96c098e808>)
    (u'fc0/affine/b', <Variable((4,), need_grad=True) at 0x7f96c098e3f8>)
    (u'fc1/affine/W', <Variable((4, 8), need_grad=True) at 0x7f96c098e1f0>)
    (u'fc1/affine/b', <Variable((8,), need_grad=True) at 0x7f96c098e328>)
    (u'fc2/affine/W', <Variable((8, 4), need_grad=True) at 0x7f96c098e0b8>)
    (u'fc2/affine/b', <Variable((4,), need_grad=True) at 0x7f96c098e258>)
    (u'fc3/affine/W', <Variable((4, 2), need_grad=True) at 0x7f96c098e120>)
    (u'fc3/affine/b', <Variable((2,), need_grad=True) at 0x7f96c098e188>)
    (u'fc/affine/W', <Variable((2, 1), need_grad=True) at 0x7f96c098e050>)
    (u'fc/affine/b', <Variable((1,), need_grad=True) at 0x7f96c06367a0>)
    (u'foo/fc0/affine/W', <Variable((2, 4), need_grad=True) at 0x7f96dc4f3ae0>)
    (u'foo/fc0/affine/b', <Variable((4,), need_grad=True) at 0x7f96dc4f3bb0>)
    (u'foo/fc1/affine/W', <Variable((4, 8), need_grad=True) at 0x7f96dc4f3738>)
    (u'foo/fc1/affine/b', <Variable((8,), need_grad=True) at 0x7f96dc4f3c80>)
    (u'foo/fc2/affine/W', <Variable((8, 4), need_grad=True) at 0x7f96dc4f3ce8>)
    (u'foo/fc2/affine/b', <Variable((4,), need_grad=True) at 0x7f96dc4f3d50>)
    (u'foo/fc3/affine/W', <Variable((4, 2), need_grad=True) at 0x7f96dc4f3e20>)
    (u'foo/fc3/affine/b', <Variable((2,), need_grad=True) at 0x7f96dc4f3f58>)
    (u'foo/fc/affine/W', <Variable((2, 1), need_grad=True) at 0x7f96dc4f3c18>)
    (u'foo/fc/affine/b', <Variable((1,), need_grad=True) at 0x7f96dc4f3328>)


.. code:: ipython2

    !rm {path_param}  # Clean ups

