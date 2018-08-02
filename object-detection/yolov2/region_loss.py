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


from utils import *
import nnabla as nn
import nnabla.functions as F
import nnabla_ext.cuda
import yolov2


def build_targets_numpy(pred_boxes, target, anchors, num_anchors,
                        num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    nB = target.shape[0]
    nA = num_anchors
    nC = num_classes
    anchor_step = len(anchors)//num_anchors
    conf_mask = np.ones((nB, nA, nH, nW), dtype=np.float32) * noobject_scale
    coord_mask, cls_mask, tx, ty, tw, th, tconf, tcls = [
        np.zeros((nB, nA, nH, nW), dtype=np.float32) for _ in range(8)]

    nAnchors = nA*nH*nW
    nPixels = nH*nW
    for b in range(nB):
        cur_pred_boxes = np.transpose(pred_boxes[b*nAnchors:(b+1)*nAnchors])
        cur_ious = np.zeros(nAnchors, dtype=np.float32)
        for t in range(50):
            if target[b][t*5+1] == 0:
                break
            gx = target[b][t*5+1]*nW
            gy = target[b][t*5+2]*nH
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            cur_gt_boxes = np.array([[gx, gy, gw, gh]], dtype=np.float32)
            cur_gt_boxes = np.repeat(cur_gt_boxes, nAnchors, axis=0)
            cur_gt_boxes = np.transpose(cur_gt_boxes)
            cur_ious = np.maximum(cur_ious, bbox_ious_numpy(
                cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        conf_mask_b = conf_mask[b]
        conf_mask_b_orig_shape = conf_mask_b.shape
        conf_mask_b = np.reshape(conf_mask_b, -1)
        conf_mask_b[cur_ious > sil_thresh] = 0
        conf_mask_b = np.reshape(conf_mask_b, conf_mask_b_orig_shape)
        conf_mask[b] = conf_mask_b
    if seen < 12800:
        if anchor_step == 4:
            tx = anchors.reshape((nA, anchor_step))[
                :, 2].reshape((1, nA, 1, 1))
            for t_axis, t_n in zip([0, 2, 3], [nB, nH, nW]):
                tx = np.repeat(tx, t_n, axis=t_axis)
            ty = anchors.reshape((num_anchors, anchor_step))[
                :, 2].reshape((1, nA, 1, 1))
            for t_axis, t_n in zip([0, 2, 3], [nB, nH, nW]):
                ty = np.repeat(tx, t_n, axis=t_axis)*2
        else:
            tx.fill(0.5)
            ty.fill(0.5)
        tw.fill(0.0)
        th.fill(0.0)
        coord_mask.fill(0.01)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t*5+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t*5+1] * nW
            gy = target[b][t*5+2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            gt_box = [0, 0, gw, gh]
            for n in range(nA):
                aw = anchors[anchor_step*n]
                ah = anchors[anchor_step*n+1]
                anchor_box = [0, 0, aw, ah]
                iou = bbox_iou_numpy(anchor_box, gt_box, x1y1x2y2=False)
                if anchor_step == 4:
                    ax = anchors[anchor_step*n+2]
                    ay = anchors[anchor_step*n+3]
                    dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                elif anchor_step == 4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]

            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
            tw[b][best_n][gj][gi] = np.log(gw/anchors[anchor_step*best_n])
            th[b][best_n][gj][gi] = np.log(gh/anchors[anchor_step*best_n+1])
            iou = bbox_iou_numpy(gt_box, pred_box, x1y1x2y2=False)  # best_iou
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t*5]
            if iou > 0.5:
                nCorrect = nCorrect + 1
    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls


def create_network(batchsize, imheight, imwidth, args):
    import gc
    gc.collect()
    nnabla_ext.cuda.clear_memory_cache()

    anchors = args.num_anchors
    classes = args.num_classes
    yolo_x = nn.Variable((batchsize, 3, imheight, imwidth))
    yolo_features = yolov2.yolov2(yolo_x, anchors, classes, test=False)

    nB = yolo_features.shape[0]
    nA = args.num_anchors
    nC = args.num_classes
    nH = yolo_features.shape[2]
    nW = yolo_features.shape[3]

    output = yolo_features.get_unlinked_variable(need_grad=True)
    # TODO: Workaround until v1.0.2.
    # Explicitly enable grad since need_grad option above didn't work.
    output.need_grad = True

    output = F.reshape(output, (nB, nA, (5 + nC), nH, nW))
    output_splitted = F.split(output, 2)
    x, y, w, h, conf = [v.reshape((nB, nA, nH, nW))
                        for v in output_splitted[0:5]]
    x, y, conf = map(F.sigmoid, [x, y, conf])

    cls = F.stack(*output_splitted[5:], axis=2)
    cls = cls.reshape((nB*nA, nC, nH*nW))
    cls = F.transpose(cls, [0, 2, 1]).reshape((nB*nA*nH*nW, nC))

    tx, ty, tw, th, tconf, coord_mask, conf_mask_sq = [
        nn.Variable(v.shape) for v in [x, y, w, h, conf, x, conf]]
    cls_ones, cls_mask = [nn.Variable(cls.shape) for _ in range(2)]
    tcls, cls_mask_bb = [nn.Variable((cls.shape[0], 1)) for _ in range(2)]

    coord_mask_sq = F.pow_scalar(coord_mask, 2)
    loss_x = args.coord_scale * F.sum(F.squared_error(x, tx) * coord_mask_sq)
    loss_y = args.coord_scale * F.sum(F.squared_error(y, ty) * coord_mask_sq)
    loss_w = args.coord_scale * F.sum(F.squared_error(w, tw) * coord_mask_sq)
    loss_h = args.coord_scale * F.sum(F.squared_error(h, th) * coord_mask_sq)
    loss_conf = F.sum(F.squared_error(conf, tconf) * conf_mask_sq)
    loss_cls = args.class_scale * \
        F.sum(cls_mask_bb * F.softmax_cross_entropy(cls + cls_ones - cls_mask, tcls))
    loss_nnabla = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

    return yolo_x, yolo_features, (x, y, w, h, conf, cls), (tx, ty, tw, th, tconf, coord_mask, conf_mask_sq, cls_ones, cls_mask, tcls, cls_mask_bb), loss_nnabla


def forward_nnabla(args, region_loss_seen, output_in, target, vars, tvars):
    anchor_step = len(args.anchors)//args.num_anchors

    nB = output_in.shape[0]
    nA = args.num_anchors
    nC = args.num_classes
    nH = output_in.shape[2]
    nW = output_in.shape[3]

    x, y, w, h, conf, cls = vars

    grid_x = np.tile(np.tile(np.linspace(0, nW-1, nW), (nH, 1)),
                     (nB*nA, 1, 1)).reshape(nB*nA*nH*nW)
    grid_y = np.tile(np.tile(np.linspace(0, nH-1, nH), (nW, 1)
                             ).transpose(), (nB*nA, 1, 1)).reshape(nB*nA*nH*nW)

    anchor_w = np.array(args.anchors).reshape((nA, anchor_step))[:, 0:1]
    anchor_h = np.array(args.anchors).reshape((nA, anchor_step))[:, 1:2]
    anchor_w = np.tile(np.tile(anchor_w, (nB, 1)),
                       (1, 1, nH*nW)).reshape(nB*nA*nH*nW)
    anchor_h = np.tile(np.tile(anchor_h, (nB, 1)),
                       (1, 1, nH*nW)).reshape(nB*nA*nH*nW)

    pred_boxes = np.zeros((4, nB*nA*nH*nW))
    pred_boxes[0, :] = x.d.reshape(nB*nA*nH*nW) + grid_x
    pred_boxes[1, :] = y.d.reshape(nB*nA*nH*nW) + grid_y
    pred_boxes[2, :] = np.exp(w.d.reshape(nB*nA*nH*nW)) * anchor_w
    pred_boxes[3, :] = np.exp(h.d.reshape(nB*nA*nH*nW)) * anchor_h
    pred_boxes = np.transpose(pred_boxes, (1, 0)).reshape(-1, 4)

    tx, ty, tw, th, tconf, coord_mask, conf_mask_sq, cls_ones_v, cls_mask_v, tcls_v, cls_mask_bb_v = tvars
    nGT, nCorrect, coord_mask.d, conf_mask_sq.d, cls_mask, tx.d, ty.d, tw.d, th.d, tconf.d, tcls = build_targets_numpy(pred_boxes, target, args.anchors, nA, nC,
                                                                                                                       nH, nW, args.noobject_scale, args.object_scale, args.thresh, region_loss_seen)

    nProposals = int(np.sum(conf.d > 0.25))

    tcls = tcls.reshape(-1)
    cls_mask_bb = cls_mask.reshape((-1, 1))
    cls_mask = np.tile(cls_mask_bb, (1, nC))
    cls_ones = np.ones(cls.shape)
    tcls = tcls.reshape((tcls.shape[0], 1))

    cls_ones_v.d = cls_ones
    cls_mask_v.d = cls_mask
    tcls_v.d = tcls
    cls_mask_bb_v.d = cls_mask_bb

    return nGT, nCorrect, nProposals
