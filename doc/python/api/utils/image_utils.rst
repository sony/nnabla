Image Utils
==============

This module provides read, write and resize functions for images. The backends of these functions are automatically changed, depending on the user`s environment.
The priority of the backends is as below (upper is higher priority):
    - OpenCV (cv2)
    - scikit-image (skimage)
    - pillow (PIL) (need to be installed)

At least one of these modules needs to be installed to use this module.

.. automodule:: nnabla.utils.image_utils

.. autofunction:: imread

.. autofunction:: imsave

.. autofunction:: imresize
