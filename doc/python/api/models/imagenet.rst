ImageNet Models
===============

This subpackage provides a variety of pre-trained state-of-the-art models which is trained on ImageNet_ dataset.

.. _ImageNet: http://www.image-net.org/

The pre-trained models can be used for both inference and training as following:

.. code-block:: python

    # Create ResNet-50 for inference
    from nnabla.models.imagenet import ResNet50
    model = ResNet50()
    batch_size = 1
    # model.input_shape returns (3, 224, 224) when ResNet-50
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


    # Create ResNet-50 for fine-tuning
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

    "`ResNet-18 <https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-18/Resnet-18.nnp>`_", "ResNet18", 30.28, 10.90, Neural Network Console
    "`ResNet-34 <https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-34/Resnet-34.nnp>`_", "ResNet34", 26.72, 8.89, Neural Network Console
    "`ResNet-50 <https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-50/Resnet-50.nnp>`_", "ResNet50", 24.59, 7.48, Neural Network Console
    "`ResNet-101 <https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-101/Resnet-101.nnp>`_", "ResNet101", 23.81, 7.01, Neural Network Console
    "`ResNet-152 <https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-152/Resnet-152.nnp>`_", "ResNet152", 23.48, 7.09, Neural Network Console
    "`MobileNet <https://nnabla.org/pretrained-models/nnp_models/imagenet/MobileNet/MobileNet.nnp>`_", "MobileNet", 29.51, 10.34, Neural Network Console
    "`MobileNetV2 <https://nnabla.org/pretrained-models/nnp_models/imagenet/MobileNet-v2/MobileNet-v2.nnp>`_", "MobileNetV2", 29.94, 10.82, Neural Network Console
    "`SENet-154 <https://nnabla.org/pretrained-models/nnp_models/imagenet/SENet-154/SENet-154.nnp>`_", "SENet", 22.04, 6.29, Neural Network Console
    "`SqueezeNet v1.0 <https://nnabla.org/pretrained-models/nnp_models/imagenet/SqueezeNet-1.0/SqueezeNet-1.0.nnp>`_", "SqueezeNetV10", 42.71, 20.12, Neural Network Console
    "`SqueezeNet v1.1 <https://nnabla.org/pretrained-models/nnp_models/imagenet/SqueezeNet-1.1/SqueezeNet-1.1.nnp>`_", "SqueezeNetV11", 41.23, 19.18, Neural Network Console
    "`VGG-11 <https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-11/VGG-11.nnp>`_", "VGG11", 30.85, 11.38, Neural Network Console
    "`VGG-13 <https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-13/VGG-13.nnp>`_", "VGG13", 29.51, 10.46, Neural Network Console
    "`VGG-16 <https://nnabla.org/pretrained-models/nnp_models/imagenet/VGG-16/VGG-16.nnp>`_", "VGG16", 29.03, 10.07, Neural Network Console
    "`NIN <https://nnabla.org/pretrained-models/nnp_models/imagenet/NIN/NIN.nnp>`_", "NIN", 42.91, 20.66, Neural Network Console
    "`DenseNet-161 <https://nnabla.org/pretrained-models/nnp_models/imagenet/DenseNet-161/DenseNet-161.nnp>`_", "DenseNet", 23.82, 7.02, Neural Network Console
    "`InceptionV3 <https://nnabla.org/pretrained-models/nnp_models/imagenet/Inception-v3/Inception-v3.nnp>`_", "InceptionV3", 21.82, 5.88, Neural Network Console
    "`Xception <https://nnabla.org/pretrained-models/nnp_models/imagenet/Xception/Xception.nnp>`_", "Xception", 23.59, 6.91, Neural Network Console
    "`GoogLeNet <https://nnabla.org/pretrained-models/nnp_models/imagenet/GoogLeNet/GoogLeNet.nnp>`_", "GoogLeNet", 31.22, 11.34, Neural Network Console
    "`ResNeXt-50 <https://nnabla.org/pretrained-models/nnp_models/imagenet/ResNeXt-50/ResNeXt-50.nnp>`_", "ResNeXt50", 22.95, 6.73, Neural Network Console
    "`ResNeXt-101 <https://nnabla.org/pretrained-models/nnp_models/imagenet/ResNeXt-101/ResNeXt-101.nnp>`_", "ResNeXt101", 22.80, 6.74, Neural Network Console
    "`ShuffleNet <https://nnabla.org/pretrained-models/nnp_models/imagenet/ShuffleNet/ShuffleNet.nnp>`_", "ShuffleNet10", 34.15, 13.85, Neural Network Console
    "`ShuffleNet-0.5x <https://nnabla.org/pretrained-models/nnp_models/imagenet/ShuffleNet-0.5x/ShuffleNet-0.5x.nnp>`_", "ShuffleNet05", 41.99, 19.64, Neural Network Console
    "`ShuffleNet-2.0x <https://nnabla.org/pretrained-models/nnp_models/imagenet/ShuffleNet-2.0x/ShuffleNet-2.0x.nnp>`_", "ShuffleNet20", 30.34, 11.12, Neural Network Console


Common interfaces
-----------------

.. automodule:: nnabla.models.imagenet.base
.. autoclass:: ImageNetBase
    :members: input_shape, category_names
    :special-members: __call__

List of models
--------------

.. automodule:: nnabla.models.imagenet

.. autoclass:: ResNet18
    :members:

.. autoclass:: ResNet34
    :members:

.. autoclass:: ResNet50
    :members:

.. autoclass:: ResNet101
    :members:

.. autoclass:: ResNet152
    :members:

.. autoclass:: ResNet
    :members:

.. autoclass:: MobileNet
    :members:

.. autoclass:: MobileNetV2
    :members:

.. autoclass:: SENet
    :members:

.. autoclass:: SqueezeNetV10
    :members:

.. autoclass:: SqueezeNetV11
    :members:

.. autoclass:: SqueezeNet
    :members:

.. autoclass:: VGG11
    :members:

.. autoclass:: VGG13
    :members:

.. autoclass:: VGG16
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

.. autoclass:: GoogLeNet
    :members:

.. autoclass:: ResNeXt50
    :members:

.. autoclass:: ResNeXt101
    :members:

.. autoclass:: ResNeXt
    :members:

.. autoclass:: ShuffleNet10
    :members:

.. autoclass:: ShuffleNet05
    :members:

.. autoclass:: ShuffleNet20
    :members:

.. autoclass:: ShuffleNet
    :members:
