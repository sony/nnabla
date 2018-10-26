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
import numpy as np


def build_targets_numpy(pred_boxes, target, anchors, num_anchors,
                        num_classes, nH, nW, coord_scale, noobject_scale,
                        object_scale, cls_scale, sil_thresh, seen):
    nB = target.shape[0]
    nA = num_anchors
    nC = num_classes
    anchor_step = len(anchors)//num_anchors

    # The mask arrays `coord_mask`, `conf_mask` and `cls_mask` indicate not only
    # whether the specific bounding box is evaluated or not, but also are
    # multiplied by loss coefficients `coord_scale`, `noobject_scale`,
    # `object_scale` and `cls_scale`.
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
    avg_iou = 0.0
    count = 0
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
            truthw = target[b][t*5+3]
            truthh = target[b][t*5+4]
            gw = truthw * nW
            gh = truthh * nH
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

            coord_mask[b][best_n][gj][gi] = coord_scale * (2 - truthw * truthh)
            cls_mask[b][best_n][gj][gi] = cls_scale
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
            avg_iou += iou
            count += 1
    return nGT, nCorrect, avg_iou / count, coord_mask[:, :, None], conf_mask[:, :, None], cls_mask[:, :, None], np.stack((tx, ty, tw, th), axis=2), tconf[:, :, None], tcls[:, :, None]
