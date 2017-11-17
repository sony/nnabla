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

import numpy as np
import matplotlib.pylab as plt
from args import get_args

import nnabla as nn


def show():
    args = get_args()

    # Load model
    nn.load_parameters(args.model_load_path)
    params = nn.get_parameters()

    # Show heatmap
    for name, param in params.items():
        # SSL only on convolution weights
        if "conv/W" not in name:
            continue
        print(name)
        n, m, k0, k1 = param.d.shape
        w_matrix = param.d.reshape((n, m * k0 * k1))
        # Filter x Channel heatmap

        fig, ax = plt.subplots()
        ax.set_title("{} with shape {} \n Filter x (Channel x Heigh x Width)".format(
            name, (n, m, k0, k1)))
        heatmap = ax.pcolor(w_matrix)
        fig.colorbar(heatmap)

        plt.pause(0.5)
        raw_input("Press Key")
        plt.close()


if __name__ == '__main__':
    show()
