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


import yolov2
from draw_utils import DrawBoundingBoxes

import nnabla as nn
import nnabla.functions as F

import time
import numpy as np
from nnabla.utils.image_utils import imread, imresize, imsave


def get_args():
    import argparse
    from os.path import dirname, basename, join
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, default=608)
    p.add_argument('--weights', type=str, default='yolov2.h5')
    p.add_argument('--context', '-c', type=str, default='cudnn')
    p.add_argument('--device-id', '-d', type=str, default='0')
    p.add_argument('--type-config', '-t', type=str, default='float')
    p.add_argument('--output', type=str, default=None)
    p.add_argument('--anchors', type=int, default=5)
    p.add_argument('--classes', type=int, default=80)
    p.add_argument('--class-names', type=str, default='coco.names')
    p.add_argument('--thresh', type=float, default=.5)
    p.add_argument('--nms', type=float, default=.45)
    p.add_argument('--nms-per-class', type=bool, default=True)
    p.add_argument(
        '--biases', nargs='*',
        default=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                 5.47434, 7.88282, 3.52778, 9.77052, 9.16828], type=float)
    p.add_argument('input', type=str, default='dog.jpg')
    args = p.parse_args()
    assert args.width % 32 == 0
    assert len(args.biases) == args.anchors * 2
    args.biases = np.array(args.biases).reshape(-1, 2)
    if args.output is None:
        args.output = join(dirname(args.input),
                           'detect.' + basename(args.input))
    return args


def draw_bounding_boxes(img, bboxes, im_w, im_h, names, colors, sub_w, sub_h, thresh):
    draw = DrawBoundingBoxes(img, colors)
    for bb in bboxes:
        if bb[4] <= 0:
            continue
        # x, y, w, h = bb[:4] * np.array([im_w, im_h, im_w, im_h])
        x, y, w, h = bb[:4]
        x = (x - (1 - sub_w) / 2.) / sub_w * im_w
        y = (y - (1 - sub_h) / 2.) / sub_h * im_h
        w = w * im_w / sub_w
        h = h * im_h / sub_h
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
        # Object detection with deep learning and OpenCV
        # https://goo.gl/q4RdcZ
        label = ', '.join("{}: {:.2f}%".format(
            names[det_ind[j]], prob[j] * 100) for j in range(len(det_ind)))
        print("[INFO] {}".format(label))
        draw.draw((x0, y0, x1, y1), det_ind[0], label)
    return draw.get()


def main():
    args = get_args()
    names = np.genfromtxt(args.class_names, dtype=str, delimiter='?')
    rng = np.random.RandomState(1223)
    colors = rng.randint(0, 256, (args.classes, 3)).astype(np.uint8)
    colors = [tuple(c.tolist()) for c in colors]

    # Set context
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Load parameter
    _ = nn.load_parameters(args.weights)

    # Build a YOLO v2 network
    feature_dict = {}
    x = nn.Variable((1, 3, args.width, args.width))
    y = yolov2.yolov2(x, args.anchors, args.classes,
                      test=True, feature_dict=feature_dict)
    y = yolov2.yolov2_activate(y, args.anchors, args.biases)
    y = F.nms_detection2d(y, args.thresh, args.nms, args.nms_per_class)

    # Read image
    img_orig = imread(args.input)
    im_h, im_w, _ = img_orig.shape
    # letterbox
    w = args.width
    h = args.width

    if (w * 1.0 / im_w) < (h * 1. / im_h):
        new_w = w
        new_h = int((im_h * w) / im_w)
    else:
        new_h = h
        new_w = int((im_w * h) / im_h)

    patch = imresize(img_orig, (new_h, new_w)) / 255.
    img = np.ones((h, w, 3), np.float32) * 0.5
    # resize
    x0 = int((w - new_w) / 2)
    y0 = int((h - new_h) / 2)
    img[y0:y0 + new_h, x0:x0 + new_w] = patch

    # Execute YOLO v2
    print ("forward")
    in_img = img.transpose(2, 0, 1).reshape(1, 3, args.width, args.width)
    x.d = in_img
    y.forward(clear_buffer=True)
    print ("done")

    bboxes = y.d[0]
    img_draw = draw_bounding_boxes(
        img_orig, bboxes, im_w, im_h, names, colors, new_w * 1.0 / w, new_h * 1.0 / h, args.thresh)
    imsave(args.output, img_draw)

    # Timing
    s = time.time()
    n_time = 10
    for i in range(n_time):
        x.d = in_img
        y.forward(clear_buffer=True)
        # Invoking device-to-host copy if CUDA
        # so that time contains data transfer.
        _ = y.d
    print("Processing time: {:.1f} [ms/image]".format(
        (time.time() - s) / n_time * 1000))


if __name__ == '__main__':
    main()
