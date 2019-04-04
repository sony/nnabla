Object Detection Models
=======================

This subpackage provides a pre-trained state-of-the-art models for the purpose of object detection which is trained on ImageNet_ dataset and fine-tuned on
Pascal_Voc_ and Coco_ dataset.

.. _ImageNet: http://www.image-net.org/
.. _Pascal_Voc: http://host.robots.ox.ac.uk/pascal/VOC/
.. _Coco: http://www.cocodataset.org/

The pre-trained models can be used for both inference and training as following:

.. code-block:: python

    # Create YoloV2 for inference
    from nnabla.models.object_detection import YoloV2
    from nnabla.models.object_detection.utils import *
    from nnabla.utils.image_utils import imread, imsave
    import nnabla as nn
    import numpy as np
    batch_size=1
    # Currently YoloV2 model can be used for Pascal VOC and MSCOCO dataset.
    model = YoloV2('VOC') # Arguments can be 'COCO' or 'VOC'
    image_height=608
    image_width=608
    # model.input_shape
    x = nn.Variable(((batch_size), 3, image_height,image_width))
    # * By training=True, it sets batch normalization mode for training
    #   and gives trainable attributes to parameters.
    # * By use_up_to='detection', it creats a network up to the output of
    #   last function of the model
    y = model(x, training=False,use_up_to='detection') #Specify the dataset
    # Path to category names of the dataset
    names = np.genfromtxt("voc.names", dtype=str, delimiter='?')
    img_orig = imread("dog.jpg", num_channels=3)
    # apply letterbox preprocessing on the input image
    w = image_width
    h = image_height
    img,new_w,new_h,im_h,im_w=letterbox(img_orig,h,w)
    # Execute YOLO v2
    print ("forward")
    in_img = img.transpose(2, 0, 1).reshape(1, 3, image_height,image_width)
    x.d[0] = in_img
    y.forward()
    bboxes = y.d[0]
    #apply inverse letterbox transform on co-ordinates
    bboxes_transformed = apply_inverse_letterbox_coordinate_transform(
                         bboxes,im_w,im_h,new_w * 1.0 / w, new_h * 1.0 / h)
    #Draw bounding boxes
    rng = np.random.RandomState(1223)
    colors = rng.randint(0, 256, (20, 3)).astype(np.uint8)
    colors = [tuple(c.tolist()) for c in colors]
    img_draw = draw_bounding_boxes(
               img_orig, bboxes_transformed, im_w, im_h, names, colors,.5)
    imsave("detected.jpg", img_draw)
    
.. csv-table:: Available Object Detection models
    :header: "Name", "Class", "mAP", "Training framework", "Notes"

    "YOLO v2 (trained on VOC)", "YoloV2", 76.00, "Darknet", "Weights converted from `author's model <https://pjreddie.com/darknet/yolov2/>`_"
    "YOLO v2 (trained on COCO)", "YoloV2",44.12 , "Darknet", "Weights converted from `author's model <https://pjreddie.com/darknet/yolov2/>`_"
    
Common interfaces
-----------------

.. automodule:: nnabla.models.object_detection.base
.. autoclass:: ObjectDetection
    :members: input_shape
    :special-members: __call__

.. automodule:: nnabla.models.object_detection.utils    
.. autofunction:: letterbox
.. autofunction:: draw_bounding_boxes
.. autofunction:: apply_inverse_letterbox_coordinate_transform
    
List of models
--------------

.. automodule:: nnabla.models.object_detection

.. autoclass:: YoloV2
    :members:
