ImageNet Models
===============

This subpackage provides a variety of pre-trained state-of-the-art models which is trained on ImageNet_ dataset.

.. _ImageNet: http://www.image-net.org/

The pre-trained models can be used for both inference and training as following:

.. code-block:: python

    # Create ResNet-50 for inference
    import nnabla as nn
    import nnabla.functions as F
    import nnabla.parametric_functions as PF
    import numpy as np
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
    # * By use_up_to='pool', it creates a network up to the output of
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

    "`ResNet-18 <https://zenodo.org/records/16962677/files/Resnet-18.nnp?download=1>`_", "ResNet18", 30.28, 10.90, Neural Network Console
    "`ResNet-34 <https://zenodo.org/records/16962677/files/Resnet-34.nnp?download=1>`_", "ResNet34", 26.72, 8.89, Neural Network Console
    "`ResNet-50 <https://zenodo.org/records/16962677/files/Resnet-50.nnp?download=1>`_", "ResNet50", 24.59, 7.48, Neural Network Console
    "`ResNet-101 <https://zenodo.org/records/16962677/files/Resnet-101.nnp?download=1>`_", "ResNet101", 23.81, 7.01, Neural Network Console
    "`ResNet-152 <https://zenodo.org/records/16962677/files/Resnet-152.nnp?download=1>`_", "ResNet152", 23.48, 7.09, Neural Network Console
    "`MobileNet <https://zenodo.org/records/16962677/files/MobileNet.nnp?download=1>`_", "MobileNet", 29.51, 10.34, Neural Network Console
    "`MobileNetV2 <https://zenodo.org/records/16962677/files/MobileNet-v2.nnp?download=1>`_", "MobileNetV2", 29.94, 10.82, Neural Network Console
    "`SENet-154 <https://zenodo.org/records/16962677/files/SENet-154.nnp?download=1>`_", "SENet", 22.04, 6.29, Neural Network Console
    "`SqueezeNet v1.0 <https://zenodo.org/records/16962677/files/SqueezeNet-1.0.nnp?download=1>`_", "SqueezeNetV10", 42.71, 20.12, Neural Network Console
    "`SqueezeNet v1.1 <https://zenodo.org/records/16962677/files/SqueezeNet-1.1.nnp?download=1>`_", "SqueezeNetV11", 41.23, 19.18, Neural Network Console
    "`VGG-11 <https://zenodo.org/records/16962677/files/VGG-11.nnp?download=1>`_", "VGG11", 30.85, 11.38, Neural Network Console
    "`VGG-13 <https://zenodo.org/records/16962677/files/VGG-13.nnp?download=1>`_", "VGG13", 29.51, 10.46, Neural Network Console
    "`VGG-16 <https://zenodo.org/records/16962677/files/VGG-16.nnp?download=1>`_", "VGG16", 29.03, 10.07, Neural Network Console
    "`NIN <https://zenodo.org/records/16962677/files/NIN.nnp?download=1>`_", "NIN", 42.91, 20.66, Neural Network Console
    "`DenseNet-161 <https://zenodo.org/records/16962677/files/DenseNet-161.nnp?download=1>`_", "DenseNet", 23.82, 7.02, Neural Network Console
    "`InceptionV3 <https://zenodo.org/records/16962677/files/Inception-v3.nnp?download=1>`_", "InceptionV3", 21.82, 5.88, Neural Network Console
    "`Xception <https://zenodo.org/records/16962677/files/Xception.nnp?download=1>`_", "Xception", 23.59, 6.91, Neural Network Console
    "`GoogLeNet <https://zenodo.org/records/16962677/files/GoogLeNet.nnp?download=1>`_", "GoogLeNet", 31.22, 11.34, Neural Network Console
    "`ResNeXt-50 <https://zenodo.org/records/16962677/files/ResNeXt-50.nnp?download=1>`_", "ResNeXt50", 22.95, 6.73, Neural Network Console
    "`ResNeXt-101 <https://zenodo.org/records/16962677/files/ResNeXt-101.nnp?download=1>`_", "ResNeXt101", 22.80, 6.74, Neural Network Console
    "`ShuffleNet <https://zenodo.org/records/16962677/files/ShuffleNet.nnp?download=1>`_", "ShuffleNet10", 34.15, 13.85, Neural Network Console
    "`ShuffleNet-0.5x <https://zenodo.org/records/16962677/files/ShuffleNet-0.5x.nnp?download=1>`_", "ShuffleNet05", 41.99, 19.64, Neural Network Console
    "`ShuffleNet-2.0x <https://zenodo.org/records/16962677/files/ShuffleNet-2.0x.nnp?download=1>`_", "ShuffleNet20", 30.34, 11.12, Neural Network Console


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
