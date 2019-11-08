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

from __future__ import absolute_import

import sys
import os

# Set path to neu
common_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(common_utils_path)

from neu.reporter import Reporter
from neu.post_processing import Colorize
from neu.variable_utils import set_persistent_all, get_params_startswith
from neu.yaml_wrapper import read_yaml, write_yaml
from neu.misc import init_nnabla, get_current_time, AttrDict
from neu.losses import get_gan_loss, vgg16_perceptual_loss
from neu.lr_scheduler import LinearDecayScheduler
from neu.layers import PatchGAN
from neu.datasets.city_scapes import (create_data_iterator as create_cityscapes_iterator,
                                  get_cityscape_datalist,
                                  load_function as cityscapes_load_function)
