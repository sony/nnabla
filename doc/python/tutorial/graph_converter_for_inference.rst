
Graph Converter for Inference
=============================

In this tutorial, we demonstrate several graph converters mainly used
for inference. Graph converters are basically used for a trained graph,
neural network, so once you train a neural network, you can use graph
converters.

We show how to use the following graph converters step-by-step according
to usecases.

1. BatchNormalizationLinearConverter
2. BatchNormalizationFoldedConverter
3. FixedPointWeightConverter
4. FixedPointActivationConverter

**Note** before starting the following instruction, import python
modules needed.

.. code:: python

    # Import
    import numpy as np
    import nnabla as nn
    import nnabla.functions as F
    import nnabla.parametric_functions as PF
    
    import nnabla.experimental.viewers as V
    import nnabla.experimental.graph_converters as GC

Also, define LeNet as the motif.

.. code:: python

    # LeNet
    def LeNet(image, test=False):
        h = PF.convolution(image, 16, (5, 5), (1, 1), with_bias=False, name='conv1')
        h = PF.batch_normalization(h, batch_stat=not test, name='conv1-bn')
        h = F.max_pooling(h, (2, 2))
        h = F.relu(h)
    
        h = PF.convolution(h, 16, (5, 5), (1, 1), with_bias=True, name='conv2')
        h = PF.batch_normalization(h, batch_stat=not test, name='conv2-bn')
        h = F.max_pooling(h, (2, 2))
        h = F.relu(h)
         
        h = PF.affine(h, 10, with_bias=False, name='fc1')
        h = PF.batch_normalization(h, batch_stat=not test, name='fc1-bn')
        h = F.relu(h)
    
        pred = PF.affine(h, 10, with_bias=True, name='fc2')
        return pred


BatchNormalizationLinearConverter
---------------------------------

Typical networks contain the batch normalization layers. It serves as
normalization in a network and uses the batch stats (the batch mean and
variance) to normalize inputs as

.. math::


   z = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,

in training. :math:`\mu` and :math:`\sigma^2` are the batch mean and
variance, and :math:`\gamma` and :math:`\beta` are the scale and bias
parameter to be learnt.

At the same time, it computes the running stats (the exponential moving
average :math:`\mu_r` and variance :math:`\sigma_r^2` of inputs to the
batch normalization layer), which are used later for inference.

If nothing changes, in inference time, the batch normalization is
performed as in the above equation using the running stats.

.. math::


   z = \gamma \frac{x - \mu_r}{\sqrt{\sigma_r^2 + \epsilon}} + \beta.

This is the explicit normalization, so as you can see, there are many
redundant computations (subtraction, devision, pow2, sqrt,
multiplication, addition) in inference, which should be avoided in
inference graph. We can do it by ourselves, but it is apparently
troublesome.

BatchNormalizationLinearConverter automatically converts this equation
of the batch normalization to the simple linear form as

.. math::


   z = c_0 x + c_1, \\
   c_0 = \frac{\gamma}{\sqrt{\sigma_r^2 + \epsilon}}, \\
   c_1 = \beta - \frac{\gamma \mu_r}{\sqrt{\sigma_r^2 + \epsilon}}.

After the conversion, we just have one multiplication and one addition
since :math:`c_0` and :math:`c_1` can be precomputed in inference.

Specifically, suppose that :math:`x` is the output of the
2D-Convolution, so :math:`x` is 3D-Tensor (e.g.,
:math:`N \times H \times W`). In the batch normalization, the number of
:math:`c`\ s is the map size :math:`N`, respectively for :math:`c_0` and
:math:`c_1`. Thus, the multiplication (:math:`c_0 \times x`) is
:math:`N \times H \times W` and the addition :math:`c_0 \times x + c_1` is same
:math:`N \times H \times W`. We can see much reduction compared to the
native implementation.

Example
~~~~~~~

First, create LeNet.

.. code:: python

    x = nn.Variable.from_numpy_array(np.random.rand(4, 3, 28, 28))
    y = LeNet(x, test=True)

Now look at LeNet visually.

.. code:: python

    viewer = V.SimpleGraph()
    viewer.view(y)

Convert it to the one with the batch normalization linearly folded.

.. code:: python

    converter = GC.BatchNormalizationLinearConverter(name="bn-linear-lenet")
    y = converter.convert(y, [x])

Also, show the converted graph.

.. code:: python

    viewer = V.SimpleGraph()
    viewer.view(y)

BatchNormalizationFoldedConverter
---------------------------------

As you can see in the previous converter,
BatchNormalizationLinearConverter is the linear folding of the batch
normalization layer in inference. However, if the preceding layer of the
batch normalization is the convolution, affine or another layer
performing inner-product, that the linear folding is further folded into
the weights of the preceding layers.

Suppose the sequence of a convolution and a batch normalization in
inference, it can be written as,

.. math::


   z = c_0 \times (w \ast x + b) + c_1,

where :math:`\ast` is the convolutional operator, :math:`w` is the
convolutional weights, and :math:`b` is the bias of the convolution
layer. Since :math:`\ast` has linearity, we can further fold :math:`c_0`
into the weights :math:`w` and bias :math:`b`, such that we have the
simpler form.

.. math::


   z = w' \ast x + b', \\
   w' = c_0 w, \\
   b' = c_0 b + c_1.

BatchNormalizationFoldedConverter automatically finds a sequence of the
convolution and the batch normalization in a given graph, then folds all
parameters related to the batch normalization into the preceding
convolution layer. Now, we do not need the multiplication and addition
seen in the previous case, BatchNormalizationLinearConverter.

Example
~~~~~~~

First, create LeNet.

.. code:: python

    x = nn.Variable.from_numpy_array(np.random.rand(4, 3, 28, 28))
    y = LeNet(x, test=True)

Now look at LeNet visually.

.. code:: python

    viewer = V.SimpleGraph()
    viewer.view(y)

Convert it to the one with the batch normalization linearly folded.

.. code:: python

    converter = GC.BatchNormalizationFoldedConverter(name="bn-folded-lenet")
    y = converter.convert(y, [x])

Also, show the converted graph.

.. code:: python

    viewer = V.SimpleGraph()
    viewer.view(y)

FixedPointWeightConverter
-------------------------

Once training finishes, where to deploy? Your destination of deployment
of a trained model might be on Cloud or an embedded device. In either
case, the typical data type, FloatingPoint32 (FP32) might be redundant
for inference, so you may want to use SIMD operation with e.g., 4-bit or
8-bit of your target device. Training is usually performed using FP32,
while interfence might be performed FixedPoint. Hence, you have to
change corresponding layers, e.g., the convolution and affine.

FixedPointWeightConverter automatically converts the affine,
convolution, and deconvolution of a given graph to that of fixed point
version.

Example
~~~~~~~

First, create LeNet.

.. code:: python

    x = nn.Variable.from_numpy_array(np.random.rand(4, 3, 28, 28))
    y = LeNet(x, test=True)

Now look at LeNet visually.

.. code:: python

    viewer = V.SimpleGraph()
    viewer.view(y)

Convert it to the one with the batch normalization linearly folded.

.. code:: python

    converter = GC.FixedPointWeightConverter(name="fixed-point-weight-lenet")
    y = converter.convert(y, [x])

Also, show the converted graph.

.. code:: python

    viewer = V.SimpleGraph()
    viewer.view(y)

FixedPointActivationConverter
-----------------------------

FixedPointWeightConverter converts layers of weights, but
FixedPointActivationConverter automatically converts activation layers,
e.g., ReLU. The typial neural network architecture contains the sequence
of the block ``ReLU -> Convolution -> BatchNormalization``; therefore,
when you convert both ``ReLU`` and ``Convolution`` to the fixed-point
ones with proper hyper-paremters (step-size and bitwidth), you can
utilize your SIMD operation of your target device because both of the
weights and inputs of the convolution are fixed-point.

Example
~~~~~~~

First, create LeNet.

.. code:: python

    x = nn.Variable.from_numpy_array(np.random.rand(4, 3, 28, 28))
    y = LeNet(x, test=True)

Now look at LeNet visually.

.. code:: python

    viewer = V.SimpleGraph()
    viewer.view(y)

Convert it to the one with the batch normalization linearly folded.

.. code:: python

    converter = GC.FixedPointActivationConverter(name="fixed-point-activation-lenet")
    y = converter.convert(y, [x])

Also, show the converted graph.

.. code:: python

    viewer = V.SimpleGraph()
    viewer.view(y)

Tipically, FixedPointWeightConverter and FixedPointActivationConverter
are used togather. For such purposes, you can use
``GC.SequentialConverter``.

.. code:: python

    converter_w = GC.FixedPointWeightConverter(name="fixed-point-lenet")
    converter_a = GC.FixedPointActivationConverter(name="fixed-point-lenet")
    converter = GC.SequentialConverter([converter_w, converter_a])
    y = converter.convert(y, [x])

Needless to say, ``GC.SequentialConverter`` is not limited to using this
case. One you creat your own ``Conveterter``\ s, then you can add these
converters to ``GC.SequentialConverter`` if these are used togather.

Look at the converted graph visually.

.. code:: python

    viewer = V.SimpleGraph()
    viewer.view(y)
