# Copyright 2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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
import numpy as np
from nnabla.models.object_detection.draw_utils import DrawBoundingBoxes
from nnabla.utils.image_utils import imresize


def draw_bounding_boxes(img, bboxes, names, colors=None, thresh=0.5):
    '''
    The transformed cordinates are further used to draw bounding boxes for the detected objects.

    Args:
        img (numpy.ndarray) : Input image
        bboxes (numpy.ndarray): 
            Transformed bounding box coorinates from the model.
        names (list of str): Name of categories in the dataset
        colors (list of tuple of 3 ints): Colors for bunding boxes
        thresh (float): Threshold of bounding boxes.

    '''
    if colors is None:
        rng = np.random.RandomState(1223)
        colors = rng.randint(0, 256, (len(names), 3)).astype(np.uint8)
        colors = [tuple(c.tolist()) for c in colors]

    im_h, im_w = img.shape[:2]
    draw = DrawBoundingBoxes(img, colors)
    for bb in bboxes:
        x, y, w, h = bb[:4]
        dw = w / 2.
        dh = h / 2.
        x0 = int(np.clip(x - dw, 0, im_w))
        y0 = int(np.clip(y - dh, 0, im_h))
        x1 = int(np.clip(x + dw, 0, im_w))
        y1 = int(np.clip(y + dh, 0, im_h))
        det_ind = np.where(bb[5:] > thresh)[0]
        if len(det_ind) == 0:
            continue
        prob = bb[5 + det_ind]
        label = ', '.join("{}: {:.2f}%".format(
            names[det_ind[j]], prob[j] * 100) for j in range(len(det_ind)))
        print("[INFO] {}".format(label))
        draw.draw((x0, y0, x1, y1), det_ind[0], label)
    return draw.get()


def apply_inverse_letterbox_coordinate_transform(bboxes, im_w, im_h, letterbox_w, letterbox_h):
    '''
    The predicted bounding box coordinates from the model are not according to original image but the pre-processed image. This function transforms the coorinates
    according to original image by applying inverse letterbox co-rdinate trasforms mathematically.

    Args:

        bboxes: 
             The bounding box coordinates predicted from the model.
        im_w : 
             Width of original input image.
        im_h :
             Height of original input image.

    '''
    bboxes = bboxes.copy()
    for bb in bboxes:
        x, y, w, h = bb[:4]
        x1 = (x - (1 - letterbox_w) / 2.) / letterbox_w * im_w
        y1 = (y - (1 - letterbox_h) / 2.) / letterbox_h * im_h
        w1 = w * im_w / letterbox_w
        h1 = h * im_h / letterbox_h
        bb[:4] = x1, y1, w1, h1
    return bboxes


def letterbox(img_orig, h, w):
    '''
    Input image is pre-processed before passing it to the network in YoloV2. This function applies the pre-processing to input image.

    Args:
        img_orig: Input image
        w : Desired width of output image after pre-processing. Should be a multiple of 32.
        h : Desired height of output image after pre-processing. Should be a multiple of 32.
    '''
    assert img_orig.dtype == np.uint8
    im_h, im_w, _ = img_orig.shape
    if (w * 1.0 / im_w) < (h * 1. / im_h):
        new_w = w
        new_h = int((im_h * w) / im_w)
    else:
        new_h = h
        new_w = int((im_w * h) / im_h)

    patch = imresize(img_orig, (new_w, new_h))
    img = np.ones((h, w, 3), np.uint8) * 127
    # resize
    x0 = int((w - new_w) / 2)
    y0 = int((h - new_h) / 2)
    img[y0:y0 + new_h, x0:x0 + new_w] = patch
    return img, new_w, new_h


class LetterBoxTransform(object):
    '''Create an object holding a new letterboxed image as `image` attribute.

    Letterboxing is defined as scaling the input image to fit inside the
    desired output image frame (letterbox) while preserving the aspect
    ratio of the original image. The pixels that are not filled with the
    original image pixels become 127.

    The created object also provides a functionality to convert bounding box
    coordinates back to the original image frame.

    Args:

        image (numpy.ndarray): An uint8 3-channel image 
        height (int): Letterbox height
        width (int): Letterbox width

    '''

    def __init__(self, image, height, width):
        self.height, self.width = height, width
        self.im_h, self.im_w = image.shape[:2]
        self.image, self.new_w, self.new_h = letterbox(image, height, width)

    def inverse_coordinate_transform(self, coords):
        '''Convert the bounding boxes back to the original image frame.

        Args:
            coords (numpy.ndarray):
                `N` x `M` array where `M >= 4` and first 4 elements
                of `M` are `x`, `y` (center coordinates of bounding box),
                `w` and `h` (bouding box width and height).

        '''
        return apply_inverse_letterbox_coordinate_transform(
            coords, self.im_w, self.im_h, self.new_w * 1.0 / self.width, self.new_h * 1.0 / self.height)
