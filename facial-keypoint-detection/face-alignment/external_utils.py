# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The code in this file is forked from the bellow mentioned github repository and it is modified as required.
# https://github.com/1adrianb/face-alignment

import numpy as np
import cv2
import nnabla as nn
import nnabla.functions as F
import math


def transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {numpy.array} -- the input 2D point
        center {numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    point.append(1)

    h = 200.0 * scale
    t = F.matrix_diag(F.constant(1, [3]))
    t.d[0, 0] = resolution / h
    t.d[1, 1] = resolution / h
    t.d[0, 2] = resolution * (-center[0] / h + 0.5)
    t.d[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = F.reshape(F.batch_inv(F.reshape(t, [1, 3, 3])), [3, 3])

    _pt = nn.Variable.from_numpy_array(point)

    new_point = F.reshape(F.batch_matmul(
        F.reshape(t, [1, 3, 3]), F.reshape(_pt, [1, 3, 1])), [3, ])[0:2]

    return new_point.d.astype(int)


def crop(image, center, scale, resolution=256):
    """Center crops an image or set of heatmaps

    Arguments:
        image {numpy.array} -- an rgb image
        center {numpy.array} -- the center of the object, usually the same as of the bounding box
        scale {float} -- scale of the face

    Keyword Arguments:
        resolution {float} -- the size of the output cropped image (default: {256.0})

    Returns:
        [type] -- [description]
    """  # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)

    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                           image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array(
        [int(max(1, ul[0] + 1)), int(min(br[0], wd))], dtype=np.int32)
    oldY = np.array(
        [int(max(1, ul[1] + 1)), int(min(br[1], ht))], dtype=np.int32)

    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
           ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]

    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                        interpolation=cv2.INTER_LINEAR)
    return newImg


def get_preds_fromhm(hm, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.

    Arguments:
        hm {numpy.array} -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {numpy.array} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    idx = F.max(F.reshape(
        hm, (hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])), axis=2, only_index=True)
    idx.d += 1
    idx = F.reshape(idx, (1, 68, 1))
    preds = F.concatenate(idx, idx, axis=2)
    preds.d[..., 0] = preds[..., 0].apply(
        d=(preds[..., 0].d - 1) % hm.shape[3] + 1).d
    preds.d[..., 1] = preds[..., 1].apply(
        d=(preds[..., 1].d + 1) // hm.shape[2] + 1).d

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0].d) - 1, int(preds[i, j, 1].d) - 1

            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                preds.d[i, j] += np.sign(hm_.d[pY, pX + 1] - hm_.d[pY, pX - 1]) * .25, np.sign(
                    hm_.d[pY + 1, pX] - hm_.d[pY - 1, pX]) * .25

    preds.d -= .5
    preds_orig = F.constant(shape=preds.shape)
    if center is not None and scale is not None:
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                d = transform(list(preds.d[i][j]),
                              center, scale, hm.shape[2], True)
                preds_orig.d[i, j] = d[0], d[1]

    return preds, preds_orig


def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
           int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
           int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image
