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


import dataset
import utils
import numpy as np
import os
import itertools
from multiprocessing.pool import ThreadPool
import nnabla
import nnabla_ext.cuda
import yolov2

from arg_utils import Yolov2OptionValid
args = Yolov2OptionValid().parse_args()


def valid(weightfile, outfile, outdir):
    pool = ThreadPool(1)
    valid_images = args.valid
    name_list = args.names
    prefix = outdir
    names = utils.load_class_names(name_list)

    utils.set_default_context_by_args(args)

    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]

    # Build the YOLO v2 network
    def create_losses(batchsize, imheight, imwidth, test=True):
        import gc
        gc.collect()
        nnabla_ext.cuda.clear_memory_cache()

        anchors = args.num_anchors
        classes = args.num_classes
        yolo_x = nnabla.Variable((batchsize, 3, imheight, imwidth))
        yolo_features = yolov2.yolov2(yolo_x, anchors, classes, test=test)
        return yolo_x, yolo_features

    yolo_x_nnabla, yolo_features_nnabla = create_losses(
        args.valid_batchsize, args.height, args.width, test=True)
    nnabla.load_parameters(weightfile)

    valid_dataset = dataset.data_iterator_yolo(valid_images, args,
                                               train=False,
                                               shape=(args.width, args.height), shuffle=False, batch_size=args.valid_batchsize)
    assert(args.valid_batchsize > 1)

    fps = [0]*args.num_classes
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for i in range(args.num_classes):
        buf = '%s/%s%s.txt' % (prefix, outfile, names[i])
        fps[i] = open(buf, 'w')

    lineId = 0
    total_samples = len(valid_files)
    total_batches = (total_samples+args.valid_batchsize -
                     1)//args.valid_batchsize

    for each_batch in range(0, total_batches):
        ret = valid_dataset.next()
        data, target = ret
        yolo_x_nnabla.d = data
        yolo_features_nnabla.forward(clear_buffer=True)
        batch_boxes = utils.get_region_boxes(
            yolo_features_nnabla.d, args.conf_thresh, args.num_classes, args.anchors, args.num_anchors, 0, 1)
        for i in range(yolo_features_nnabla.d.shape[0]):
            if lineId >= total_samples:
                print("Reached End of total_samples")
                break
            fileId = os.path.basename(valid_files[lineId]).split('.')[0]
            width, height = utils.get_image_size(valid_files[lineId])
            print(valid_files[lineId])
            lineId += 1
            boxes = batch_boxes[i]
            boxes = utils.nms(boxes, args.nms_thresh)
            for box in boxes:
                x1 = (box[0] - box[2]/2.0) * width
                y1 = (box[1] - box[3]/2.0) * height
                x2 = (box[0] + box[2]/2.0) * width
                y2 = (box[1] + box[3]/2.0) * height

                det_conf = box[4]
                for j in range((len(box)-5)//2):
                    cls_conf = box[5+2*j]
                    cls_id = box[6+2*j]
                    prob = det_conf * cls_conf
                    fps[cls_id].write('%s %f %f %f %f %f\n' %
                                      (fileId, prob, x1, y1, x2, y2))

    for i in range(args.num_classes):
        fps[i].close()


if __name__ == '__main__':
    weightfile = args.weight
    outdir = args.output
    outfile = 'comp4_det_test_'
    valid(weightfile, outfile, outdir)
