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


import time
import collections

from utils import *
from region_loss_utils import *
import nnabla as nn
import nnabla.functions as F
import nnabla_ext.cuda
import yolov2

from nnabla.function import PythonFunction


class RegionLossTargets(PythonFunction):
    def __init__(self, num_classes, anchors, seen, coord_scale=1.0, noobject_scale=1.0, object_scale=5.0, class_scale=1.0, thresh=0.6):
        '''
        Args:
            thresh (float): boxes with best IoU<=``thresh`` are considered as
                negative samples wrt objectness.

        Returns:

        '''
        self.num_classes = num_classes
        self.anchors = anchors
        self.seen = seen
        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.thresh = thresh

        self.prev_miou = None
        self.prev_ngt = None
        self.prev_ncorrect = None

    @property
    def name(self):
        return 'PythonRegionLossTargets'

    def min_outputs(self):
        return 6

    def setup_impl(self, inputs, outputs):
        bbox_pred = inputs[0]
        nB, nA, _, nH, nW = bbox_pred.shape
        tcoord, mcoord, tconf, mconf, tcls, mcls = outputs
        mask_shape = (nB, nA, 1, nH, nW)
        tcoord.reset_shape(bbox_pred.shape, True)
        mcoord.reset_shape(mask_shape, True)
        tconf.reset_shape(mask_shape, True)
        mconf.reset_shape(mask_shape, True)
        tcls.reset_shape(mask_shape, True)
        mcls.reset_shape(mask_shape, True)

    def forward_impl(self, inputs, outputs):
        bbox, target = inputs
        nB, nA, _, nH, nW = bbox.shape
        nC = self.num_classes
        anchor_step = len(self.anchors) // nA

        grid_x = np.arange(0, nW)[None, None, None, None, :]
        grid_y = np.arange(0, nH)[None, None, None, :, None]

        pred_boxes = np.zeros((4, nB*nA*nH*nW))
        bb = bbox.data.get_data('r')
        pred_boxes[0, :] = (bb[:, :, 0] + grid_x).flat
        pred_boxes[1, :] = (bb[:, :, 1] + grid_y).flat
        anchors = np.array(self.anchors).reshape(-1, 2)  # nA * 2
        # clip without out buffer is sometimes slow
        tmpbuff = np.empty_like(bb[:, :, 2])
        tmpbuff[...] = bb[:, :, 2]
        tmpbuff[tmpbuff > 5] = 5
        pred_boxes[2, :] = (
            np.exp(tmpbuff) * anchors[None, :, 0, None, None]).flat
        tmpbuff[tmpbuff > 5] = 5
        pred_boxes[3, :] = (
            np.exp(tmpbuff) * anchors[None, :, 1, None, None]).flat
        pred_boxes = np.transpose(pred_boxes, (1, 0)).reshape(-1, 4)
        # tcoord, mcoord, tconf, mconf, tcls, mcls = outputs
        o = [v.data.get_data('w') for v in outputs]
        nGT, nCorrect, mIoU, o[1][...], o[3][...], o[5][...], o[0][...], o[2][...], o[4][...] = build_targets_numpy(
            pred_boxes, target.d, self.anchors, nA, nC, nH, nW,
            self.coord_scale, self.noobject_scale, self.object_scale,
            self.class_scale, self.thresh, self.seen)
        self.seen += nB
        self.prev_ngt = nGT
        self.prev_ncorrect = nCorrect
        self.prev_miou = mIoU

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        pass


def create_network(batchsize, imheight, imwidth, args, seen):
    import gc
    gc.collect()
    nnabla_ext.cuda.clear_memory_cache()

    anchors = args.num_anchors
    classes = args.num_classes
    yolo_x = nn.Variable((batchsize, 3, imheight, imwidth))
    target = nn.Variable((batchsize, 50 * 5))
    yolo_features = yolov2.yolov2(yolo_x, anchors, classes, test=False)

    nB = yolo_features.shape[0]
    nA = args.num_anchors
    nC = args.num_classes
    nH = yolo_features.shape[2]
    nW = yolo_features.shape[3]

    # Bouding box regression loss
    # pred.shape = [nB, nA, 4, nH, nW]
    output = F.reshape(yolo_features, (nB, nA, (5 + nC), nH, nW))
    xy = F.sigmoid(output[:, :, :2, ...])
    wh = output[:, :, 2:4, ...]
    bbox_pred = F.concatenate(xy, wh, axis=2)
    conf_pred = F.sigmoid(output[:, :, 4:5, ...])
    cls_pred = output[:, :, 5:, ...]

    region_loss_targets = RegionLossTargets(
        nC, args.anchors, seen, args.coord_scale, args.noobject_scale,
        args.object_scale, args.class_scale, args.thresh)

    tcoord, mcoord, tconf, mconf, tcls, mcls = region_loss_targets(
        bbox_pred, target)
    for v in tcoord, mcoord, tconf, mconf, tcls, mcls:
        v.need_grad = False

    # Bounding box regression
    bbox_loss = F.sum(F.squared_error(bbox_pred, tcoord) * mcoord)

    # Conf (IoU) regression loss
    conf_loss = F.sum(F.squared_error(conf_pred, tconf) * mconf)

    # Class probability regression loss
    cls_loss = F.sum(F.softmax_cross_entropy(cls_pred, tcls, axis=2) * mcls)

    # Note:
    # loss is devided by 2.0 due to the fact that the original darknet
    # code doesn't multiply the derivative of square functions by 2.0
    # in region_layer.c.
    loss = (bbox_loss + conf_loss) / 2.0 + cls_loss

    return yolo_x, target, loss, region_loss_targets


Vars = collections.namedtuple('Vars', ['x', 't', 'loss'])
Stats = collections.namedtuple(
    'Stats', ['loss', 'nGT', 'nCorrect', 'nProposals', 'mIoU', 'seen', 'time'])


class TrainGraph(object):

    def __init__(self, args, default_hw=(416, 416)):
        self.args = args
        self.seen = 0
        self.v = None
        self.region_class = None
        self.create_graph([args.batch_size, 3] + list(default_hw))
        self.tic = time.time()

    def create_graph(self, shape):
        self.v = None
        self.region_class = None
        x, t, loss, region_class = create_network(
            shape[0], shape[2], shape[3],
            self.args, self.seen)
        self.v = Vars(x, t, loss)
        self.region_class = region_class

    def forward_backward(self, image, target):
        if self.v.x.shape != image.shape:
            self.create_graph(image.shape)
        self.v.x.d = image
        execution_time = (time.time() - self.tic) * 1000
        self.tic = time.time()
        self.v.t.d = target
        self.v.loss.forward(clear_no_need_grad=True)
        loss = self.v.loss.d.copy()
        self.v.loss.backward(clear_buffer=True)
        nGT = self.region_class.prev_ngt
        nCorrect = self.region_class.prev_ncorrect
        mIoU = self.region_class.prev_miou
        nProposals = -1
        self.seen = self.region_class.seen
        return Stats(loss, nGT, nCorrect, nProposals, mIoU, self.seen, execution_time)
