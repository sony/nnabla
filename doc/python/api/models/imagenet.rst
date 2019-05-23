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

Available models are summarized in the following table. Error rates are calculated using single center crop.


.. csv-table:: Available ImageNet models
    :header: "Name", "Class", "Top-1 error", "Top-5 error", "Trained by/with"

    "ResNet-18", "ResNet", 30.28, 10.90, Neural Network Console
    "ResNet-34", "ResNet", 26.72, 8.89, Neural Network Console
    "ResNet-50", "ResNet", 24.59, 7.48, Neural Network Console
    "ResNet-101", "ResNet", 23.81, 7.01, Neural Network Console
    "ResNet-152", "ResNet", 23.48, 7.09, Neural Network Console
    "MobileNet", "MobileNet", 29.51, 10.34, Neural Network Console
    "MobileNetV2", "MobileNetV2", 29.94, 10.82, Neural Network Console
    "SENet-154", "SENet", 22.04, 6.29, Neural Network Console
    "SqueezeNet v1.1", "SqueezeNet", 41.23, 19.18, Neural Network Console
    "VGG-11", "VGG", 30.85, 11.38, Neural Network Console
    "VGG-13", "VGG", 29.51, 10.46, Neural Network Console
    "VGG-16", "VGG", 29.03, 10.07, Neural Network Console
    "NIN", "NIN", 42.91, 20.66, Neural Network Console
    "DenseNet-161", "DenseNet", 23.82, 7.02, Neural Network Console
    "InceptionV3", "InceptionV3", 21.82, 5.88, Neural Network Console
    "Xception", "Xception", 23.59, 6.91, Neural Network Console


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

.. autoclass:: MobileNet
    :members:

.. autoclass:: MobileNetV2
    :members:

.. autoclass:: SENet
    :members:

.. autoclass:: SqueezeNet
    :members:

.. autoclass:: VGG
    :members:
    
.. autoclass:: NIN
    :members:
    
.. autoclass:: DenseNet
    :members:

.. autoclass:: InceptionV3
    :members:

.. autoclass:: Xception
    :members:
