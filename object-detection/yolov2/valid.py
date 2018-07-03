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
import nnabla
import nnabla_ext.cuda
import yolov2

args = utils.parse_args()


def valid(weightfile, outfile, outdir):
    valid_images = args.valid
    name_list = args.names
    prefix = outdir
    names = utils.load_class_names(name_list)

    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context("cudnn")
    nnabla.set_default_context(ctx)

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

    valid_dataset = dataset.listDataset(valid_images, args,
                                        batch_size=args.valid_batchsize,
                                        train=False,
                                        shape=(args.width, args.height), shuffle=False)
    assert(args.valid_batchsize > 1)

    def batch_iter(it, batch_size):
        def list2np(t):
            imgs, labels = zip(*t)
            retimgs = np.zeros((len(imgs),) + imgs[0].shape, dtype=np.float32)
            retlabels = np.zeros(
                (len(labels),) + labels[0].shape, dtype=np.float32)
            for i, img in enumerate(imgs):
                retimgs[i, :, :, :] = img
            for i, label in enumerate(labels):
                retlabels[i, :] = label
            return retimgs, retlabels
        retlist = []
        for i, item in enumerate(it):
            retlist.append(item)
            if i % batch_size == batch_size - 1:
                ret = list2np(retlist)
                # # Don't train for batches that contain no labels
                # # TODO: fix this
                # if not (np.sum(ret[1].numpy()) == 0):
                yield ret
                retlist = []
        # Excess data is discarded
        if len(retlist) > 0:
            ret = list2np(retlist)
            # # Don't train for batches that contain no labels
            # # TODO: fix this
            # if not (np.sum(ret[1].numpy()) == 0):
            yield ret

    valid_loader = batch_iter(
        iter(valid_dataset), batch_size=args.valid_batchsize)

    fps = [0]*args.num_classes
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for i in range(args.num_classes):
        buf = '%s/%s%s.txt' % (prefix, outfile, names[i])
        fps[i] = open(buf, 'w')

    lineId = -1

    for batch_idx, (data, target) in enumerate(valid_loader):
        yolo_x_nnabla.d = data
        yolo_features_nnabla.forward()
        batch_boxes = utils.get_region_boxes(
            yolo_features_nnabla.d, args.conf_thresh, args.num_classes, args.anchors, args.num_anchors, 0, 1)
        for i in range(yolo_features_nnabla.d.shape[0]):
            lineId = lineId + 1
            fileId = os.path.basename(valid_files[lineId]).split('.')[0]
            width, height = utils.get_image_size(valid_files[lineId])
            print(valid_files[lineId])
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
