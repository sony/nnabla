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


# This file was forked from https://github.com/marvis/pytorch-yolo2 ,
# licensed under the MIT License (see LICENSE.external for more details).


import random
import os
import cv2
import numpy as np


def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out


def convert_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def convert_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def scale_clip_inplace(x, scale, upper=255):
    '''
    Note:
    Very slow computation is occasionally observed when a equivalent
    ``np.clip(x * scale, 0, upper, out=x)`` is used.
    It only happens if the image size is large like 608x608.
    A small test using %time in ipython shown ``sys`` time is dominant
    rather than ``cpu`` time.
    By using this alternative equivalent, it is not happening so far.
    '''
    tmp = x * np.float32(scale)
    tmp[tmp > upper] = upper
    x[...] = tmp


def distort_h(im, hue):
    tim = im[..., 0] + np.float32(255 * hue + 256)
    np.fmod(tim, 256, out=tim)
    im[..., 0] = tim


def distort_s(im, sat):
    scale_clip_inplace(im[..., 1], sat, 255)


def distort_v(im, val):
    scale_clip_inplace(im[..., 2], val, 255)


def distort_image(im, hue, sat, val):
    im = convert_to_hsv(im)
    distort_h(im, hue)
    distort_s(im, sat)
    distort_v(im, val)
    return convert_to_rgb(im)


def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1, 10000) % 2):
        return scale
    return 1./scale


def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res


def crop_image(img, l, t, r, b):
    w = r - l + 1
    h = b - t + 1
    M = np.array([[1, 0, l], [0, 1, t]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    # This was slower.
    # ind_x = np.clip(np.arange(l, r + 1), 0, img.shape[1] - 1)
    # ind_y = np.clip(np.arange(t, b + 1), 0, img.shape[0] - 1)
    # return img[np.ix_(ind_y, ind_x)]


def resize_image(img, shape):
    return cv2.resize(img, shape, interpolation=cv2.INTER_NEAREST)


def crop_resize_image(img, l, t, r, b, shape, flip):
    w, h = shape
    pts1 = np.float32([[l, t], [r, t], [l, b], [r, b]])
    pts2 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    # This was wrong
    # pts1 = np.float32([[l, t], [r + 1, t], [l, b + 1], [r + 1, b + 1]])
    # pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if flip:
        pts2[:, 0] = pts2[::-1, 0]
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, shape, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh, ow = img.shape[:2]

    dw = int(ow*jitter)
    dh = int(oh*jitter)

    pleft = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    pbot = random.randint(-dh, dh)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth) / ow
    sy = float(sheight) / oh

    flip = random.randint(1, 10000) % 2

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) / oh)/sy

    # cropped = crop_image(img, pleft, ptop, pleft +
    #                      swidth - 1, ptop + sheight - 1)
    # sized = resize_image(cropped, shape)

    # if flip:
    #     sized = sized[:, ::-1]

    sized = crop_resize_image(img, pleft, ptop, pleft +
                              swidth - 1, ptop + sheight - 1, shape,
                              flip)

    img = random_distort_image(sized, hue, saturation, exposure)

    return img, flip, dx, dy, sx, sy


def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes, 5))
    if labpath is not None or (isinstance(labpath, str) and os.path.getsize(labpath)):
        if isinstance(labpath, str):
            bs = np.loadtxt(labpath)
        else:
            if labpath is None:
                return label
            bs = labpath.copy()
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3]/2
            y1 = bs[i][2] - bs[i][4]/2
            x2 = bs[i][1] + bs[i][3]/2
            y2 = bs[i][2] + bs[i][4]/2

            x1 = min(0.999, max(0, x1 * sx - dx))
            y1 = min(0.999, max(0, y1 * sy - dy))
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))

            bs[i][1] = (x1 + x2)/2
            bs[i][2] = (y1 + y2)/2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] = 0.999 - bs[i][1]

            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label


def load_data_detection(imgpath, labpath, shape, jitter, hue, saturation, exposure):

    # data augmentation
    if isinstance(imgpath, str):
        img = cv2.imread(imgpath)
    else:
        img = imgpath
    img, flip, dx, dy, sx, sy = data_augmentation(
        img, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(
        labpath, img.shape[1], img.shape[0], flip, dx, dy, 1./sx, 1./sy)
    return img[..., ::-1], label
