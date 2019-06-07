Object Detection Models
=======================

This subpackage provides a pre-trained state-of-the-art models for the purpose of object detection which is trained on ImageNet_ dataset and fine-tuned on
`Pascal VOC`_ and `MS COCO`_ dataset.

.. _ImageNet: http://www.image-net.org/
.. _`Pascal VOC`: http://host.robots.ox.ac.uk/pascal/VOC/
.. _`MS COCO`: http://www.cocodataset.org/

The pre-trained models can be used for both inference and training as following:

.. code-block:: python

    # Import required modules
    import nnabla as nn
    from nnabla.models.object_detection import YoloV2
    from nnabla.models.object_detection.utils import (
        LetterBoxTransform,
        draw_bounding_boxes)
    from nnabla.utils.image_utils import imread, imsave
    import numpy as np

    # Set device
    from nnabla.ext_utils import get_extension_context
    nn.set_default_context(get_extension_context('cudnn', device_id='0'))

    # Load and create a detection model
    h, w = 608, 608
    yolov2 = YoloV2('coco')
    x = nn.Variable((1, 3, h, w))
    y = yolov2(x)

    # Load an image and scale it to fit inside the (h, w) frame
    img_orig = imread('dog.jpg')
    lbt = LetterBoxTransform(img_orig, h, w)

    # Execute detection
    x.d = lbt.image.transpose(2, 0, 1)[None]
    y.forward(clear_buffer=True)

    # Draw bounding boxes to the original image
    bboxes = lbt.inverse_coordinate_transform(y.d[0])
    img_draw = draw_bounding_boxes(
        img_orig, bboxes, yolov2.get_category_names())
    imsave("detected.jpg", img_draw)


.. csv-table:: Available models trained on COCO dataset
    :header: "Name", "Class", "mAP", "Training framework", "Notes"

    "`YOLO v2 <https://nnabla.org/pretrained-models/nnp_models/object_detection/yolov2-coco.nnp>`_", "YoloV2", 44.12 , "Darknet", "Weights converted from `author's model <https://pjreddie.com/darknet/yolov2/>`_"

.. csv-table:: Available models trained on VOC dataset
    :header: "Name", "Class", "mAP", "Training framework", "Notes"

    "`YOLO v2 <https://nnabla.org/pretrained-models/nnp_models/object_detection/yolov2-voc.nnp>`_", "YoloV2", 76.00, "Darknet", "Weights converted from `author's model <https://pjreddie.com/darknet/yolov2/>`_"
    
Common interfaces
-----------------

.. automodule:: nnabla.models.object_detection.base
.. autoclass:: ObjectDetection
    :members: input_shape
    :special-members: __call__

.. automodule:: nnabla.models.object_detection.utils
.. autoclass:: LetterBoxTransform
    :members: inverse_coordinate_transform
.. autofunction:: draw_bounding_boxes
.. .. autofunction:: letterbox
.. .. autofunction:: apply_inverse_letterbox_coordinate_transform
    
List of models
--------------

.. automodule:: nnabla.models.object_detection

.. autoclass:: YoloV2
    :members:
