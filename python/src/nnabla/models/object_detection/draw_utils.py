# Copyright 2019,2020,2021 Sony Corporation.
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


class _DrawBoundingBoxesPil(object):
    def __init__(self, img, colors):
        self.img = Image.fromarray(img)
        self.colors = colors
        self.d = None

    def draw(self, bbox, det_ind, label):

        if self.d is None:
            self.d = ImageDraw.Draw(self.img)
        x0, y0, x1, y1 = bbox
        color = self.colors[det_ind]
        self.d.rectangle(bbox, outline=color)
        # cv2.rectangle(self.img, (x0, y0), (x1, y1), color, 2)
        text_y0 = y0 - 15 if y0 - 15 > 15 else y0 + 15
        self.d.text((x0, text_y0), label, fill=color)
        # cv2.putText(self.img, label, (x0, text_y0),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def get(self):
        self.d = None
        return np.array(self.img)


class _DrawBoundingBoxesCv2(object):
    def __init__(self, img, colors):
        self.img = img.copy()
        self.colors = colors

    def draw(self, bbox, det_ind, label):
        x0, y0, x1, y1 = bbox
        color = self.colors[det_ind]
        cv2.rectangle(self.img, (x0, y0), (x1, y1), color, 2)
        text_y0 = y0 - 15 if y0 - 15 > 15 else y0 + 15
        cv2.putText(self.img, label, (x0, text_y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def get(self):
        return self.img


try:
    import cv2
    DrawBoundingBoxes = _DrawBoundingBoxesCv2
except:
    from PIL import Image, ImageDraw
    DrawBoundingBoxes = _DrawBoundingBoxesPil
