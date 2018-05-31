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


import os
import numpy as np
import argparse
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.monitor import Monitor, MonitorImage, MonitorSeries
from nnabla.contrib.context import extension_context


class MonitorImageWithName(MonitorImage):
    def __init__(self, name, monitor, interval=1000, verbose=True,
                 num_images=16, normalize_method=None):
        super(MonitorImageWithName, self).__init__(
            name, monitor, interval=1000, verbose=True, num_images=16, normalize_method=None)

    def add(self, name, var):
        import nnabla as nn
        from scipy.misc import imsave

        if isinstance(var, nn.Variable):
            data = var.d.copy()
        elif isinstance(var, nn.NdArray):
            data = var.data.copy()
        else:
            assert isinstance(var, np.ndarray)
            data = var.copy()
        assert data.ndim > 2
        channels = data.shape[-3]
        data = data.reshape(-1, *data.shape[-3:])
        data = data[:min(data.shape[0], self.num_images)]
        data = self.normalize_method(data)
        if channels > 3:
            data = data[:, :3]
        elif channels == 2:
            data = np.concatenate(
                [data, np.ones((data.shape[0], 1) + data.shape[-2:])], axis=1)
        path_tmpl = os.path.join(self.save_dir, '{}.png')
        for j in range(min(self.num_images, data.shape[0])):
            img = data[j].transpose(1, 2, 0)
            if img.shape[-1] == 1:
                img = img[..., 0]
            path = path_tmpl.format(name)
            imsave(path, img)
