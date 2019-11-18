# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

import nnabla as nn
import cv2
import models
from nnabla.utils import image_utils
from nnabla.ext_utils import get_extension_context
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='esrgan')
parser.add_argument('--loadmodel', default='./ESRGAN_NNabla_model.h5',
                    help='load model')
parser.add_argument('--input_image', default='./baboon.png',
                    help='input image')
args = parser.parse_args()

ctx = get_extension_context('cudnn', device_id=1)
nn.set_default_context(ctx)
nn.load_parameters(args.loadmodel)

img = cv2.imread(args.input_image, cv2.IMREAD_COLOR)
img = np.transpose(img, (2, 0, 1))[::-1]
img = img * 1.0/255
c, h, w = img.shape[0], img.shape[1], img.shape[2]
x = nn.Variable((1, c, h, w))
x.d = img

y = models.rrdb_net(x, 64, 23)
y.forward(clear_buffer=True)

out = y.d.squeeze(0)
output = out[::-1].transpose(1, 2, 0)
output = (output * 255.0).round()
cv2.imwrite('result.png', output)
print("done")
