ImageNet Models
===============

This subpackage provides a variety of pre-trained state-of-the-art models which is trained on ImageNet_ dataset.

.. _ImageNet: http://www.image-net.org/

The pre-trained models can be used for both inference and training as following:

.. code-block:: python

    # Create ResNet-18 for inference
    from nnabla.models.imagenet import ResNet
    model = ResNet(18)
    batch_size = 1
    # model.input_shape returns (3, 224, 224) when ResNet-18
    x = nn.Variable((batch_size,) + model.input_shape)
    y = model(x, training=False)

    # Execute inference
    # Load input image as uint8 array with shape of (3, 224, 224)
    from nnabla.utils.image_utils import imread
    img = imread('example.jpg', size=model.input_shape[1:], channel_first=True)
    x.d[0] = img
    y.forward()
    predicted_label = np.argmax(y.d[0])
    print('Predicted label:', model.category_names[predicted_label])


    # Create ResNet-18 for fine-tuning
    batch_size=32
    x = nn.Variable((batch_size,) + model.input_shape)
    # * By training=True, it sets batch normalization mode for training
    #   and gives trainable attributes to parameters.
    # * By use_up_to='pool', it creats a network up to the output of
    #   the final global average pooling.
    pool = model(x, training=True, use_up_to='pool')

    # Add a classification layer for another 10 category dataset
    # and loss function
    num_classes = 10
    y = PF.affine(pool, num_classes, name='classifier10')
    t = nn.Variable((batch_size, 1))
    loss = F.sum(F.softmax_cross_entropy(y, t))

    # Training...


Common interfaces
-----------------

.. automodule:: nnabla.models.imagenet.base
.. autoclass:: ImageNetBase
    :members: input_shape, category_names
    :special-members: __call__

List of models
--------------

.. automodule:: nnabla.models.imagenet

.. autoclass:: ResNet
    :members:

.. autoclass:: MobileNetV2
    :members:
