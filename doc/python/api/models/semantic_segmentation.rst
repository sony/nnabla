Semantic Segmentation Models
============================

This subpackage provides a pre-trained state-of-the-art model for the purpose of semantic segmentation (DeepLabv3+, Xception-65 as backbone) which is trained on ImageNet_ dataset and fine-tuned on
`Pascal VOC`_ and `MS COCO`_ dataset.

.. _ImageNet: http://www.image-net.org/
.. _`Pascal VOC`: http://host.robots.ox.ac.uk/pascal/VOC/
.. _`MS COCO`: http://www.cocodataset.org/

The pre-trained models can be used for inference as following:

.. code-block:: python

    #Import required modules
    import numpy as np
    import nnabla as nn
    from nnabla.utils.image_utils import imread
    from nnabla.models.semantic_segmentation import DeepLabV3plus
    from nnabla.models.semantic_segmentation.utils import ProcessImage

    target_h = 513
    target_w = 513
    # Get context
    from nnabla.ext_utils import get_extension_context
    nn.set_default_context(get_extension_context('cudnn', device_id='0'))

    # Build a Deeplab v3+ network
    image = imread("./test.jpg")
    x = nn.Variable((1, 3, target_h, target_w), need_grad=False)
    deeplabv3 = DeepLabV3plus('voc-coco',output_stride=8)
    y = deeplabv3(x)

    # preprocess image
    processed_image = ProcessImage(image, target_h, target_w)
    input_array = processed_image.pre_process()

    # Compute inference
    x.d = input_array
    y.forward(clear_buffer=True)
    print ("done")
    output = np.argmax(y.d, axis=1)

    # Apply post processing
    post_processed = processed_image.post_process(output[0])

    #Display predicted class names
    predicted_classes = np.unique(post_processed).astype(int)
    for i in range(predicted_classes.shape[0]):
        print('Classes Segmented: ', deeplabv3.category_names[predicted_classes[i]])

    # save inference result
    processed_image.save_segmentation_image("./output.png")


.. csv-table:: Available models trained on voc dataset
    :header: "Name", "Class", "Output stride", "mIOU", "Training framework", "Notes"

    "`DeepLabv3+ <https://nnabla.org/pretrained-models/nnp_models/semantic_segmentation/DeepLabV3-voc-os-8.nnp>`_", "DeepLabv3+", 8, 81.48 , "Nnabla", "Backbone (Xception-65) weights converted from `author's model <http://download.tensorflow.org/models/deeplabv3_xception_2018_01_04.tar.gz>`_ and used for finetuning"
    "`DeepLabv3+ <https://nnabla.org/pretrained-models/nnp_models/semantic_segmentation/DeepLabV3-voc-os-16.nnp>`_", "DeepLabv3+", 16, 82.20 , "Nnabla", "Backbone (Xception-65) weights converted from `author's model <http://download.tensorflow.org/models/deeplabv3_xception_2018_01_04.tar.gz>`_ and used for finetuning"

.. csv-table:: Available models trained on Voc and coco dataset
    :header: "Name", "Class", "Output stride", "mIOU", "Training framework", "Notes"

    "`DeepLabv3+ <https://nnabla.org/pretrained-models/nnp_models/semantic_segmentation/DeepLabV3-voc-coco-os-8.nnp>`_", "DeepLabv3+", 8, 82.20, "Tensorflow", "Weights converted from `author's model <http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz>`_"
    "`DeepLabv3+ <https://nnabla.org/pretrained-models/nnp_models/semantic_segmentation/DeepLabV3-voc-coco-os-16.nnp>`_", "DeepLabv3+", 16, 83.58, "Tensorflow", "Weights converted from `author's model <http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz>`_"
    
Common interfaces
-----------------

.. automodule:: nnabla.models.semantic_segmentation.base
.. autoclass:: SemanticSegmentation
    :members: input_shape
    :special-members: __call__

.. automodule:: nnabla.models.semantic_segmentation.utils

.. autoclass:: ProcessImage
    :members: pre_process
    :members: post_process
    :members: save_segmentation_image

    
List of models
--------------

.. automodule:: nnabla.models.semantic_segmentation

.. autoclass:: DeepLabV3plus
    :members:
