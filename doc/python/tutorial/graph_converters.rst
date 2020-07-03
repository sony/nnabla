
Graph Converters
================

As neural networks becomes complex and one of components in a system, we
sometimes want to convert a network as we want. Typical usecase is for
inference. We want to merge or change some layers in a network as a
high-level optimization for the inference speed. Also, there are other
usecases: adding new layers to keep track some stats, adding
quantize/dequantize layers for a quantized inference, decomposing a
layer as combination of a low-rank ones, changing a network architecture
for the neural architecture search based on an original network
architecture, changing the tensor format from the channel first to
channel last and opposite, and so on.

Let's look at the simple cases 1. batch normalization folding 2. channel
last conversion

As a reference network, use the follows.

.. code:: ipython3

    # ResNet-50 for inference
    import nnabla as nn
    import nnabla.functions as F
    import nnabla.parametric_functions as PF
    import numpy as np
    from nnabla.utils.inspection import pprint
    from nnabla.models.imagenet import ResNet50
    
    model = ResNet50()
    
    batch_size = 1
    x = nn.Variable((batch_size,) + model.input_shape)
    y = model(x, training=False)

Batch Normalization Folding
---------------------------

See the resnet architecture.

.. code:: ipython3

    pprint(y)

Now, we can see the batch normalization. For the inference, we do not
need to compute the batch normalization explicitly by folding the batch
normalization parameters if there is e.g., a convolution before the
batch normalization.

To fold the batch normalization, use BatchNormalizationFoldingModifier
as the following.

.. code:: ipython3

    import nnabla.experimental.graph_converters as GC
    
    modifiers = [GC.BatchNormalizationFoldingModifier()]
    gc = GC.GraphConverter(modifiers)
    yy = gc.convert(y)

Again, see the resnet architecture converted.

.. code:: ipython3

    pprint(yy)

You can see that the converterd network does not contain the batch
normalization any more!

In some cases, we can not fold the batch normalization, but the batch
normalization can also be self-folded, i.e., the four parameters: scale,
bias, running mean, running variance can be two other scale and bias.
For doing this, use BatchNormalizationSelfFoldingModifier.

Channel Last Conversion
-----------------------

In NVIDIA latest GPU architectures since Volta, it supports TensorCore
to accelerate the computatoinal performance. To boost the performance as
maximum as possible, we need the channel-last tensor format aka NHWC. In
NNabla, the default tensor format is the channel first aka NCHW, so as
to utilize TensorCore, we need to change the tensor format to NHWC
format.

ChannelLastModifier convert a network with NCHW tesnor format to another
network with NHWC tensor format.

.. code:: ipython3

    import nnabla.experimental.graph_converters as GC
    
    modifiers = [GC.ChannelLastModifier([x])]
    gc = GC.GraphConverter(modifiers)
    yy = gc.convert(y)

Let's see the resnet architecture converted.

.. code:: ipython3

    pprint(yy)

We can find the channel dimension changed at the last!

If we want to access to the inputs of which tensor format converted,

.. code:: ipython3

    x_cl = modifiers[0].inputs_cl[0]
    print(x_cl)

Note that ChannelLastModifier supports a set of layers: Convolution,
Deconvolution, BatchNormalization, MaxPooling, AveragePooling,
SumPooling, Unpooling, Concatenate and also supposes NCHW format.

There also exists ChannelFirstModifier in the opposite change.
