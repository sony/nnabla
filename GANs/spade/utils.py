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

common_utils_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(common_utils_path)

from neu.reporter import Reporter
from neu.post_processing import Colorize
from neu.variable_utils import *
from neu.layers import spade, PatchGAN, rescale_values
from neu.losses import vgg16_perceptual_loss
from neu.lr_scheduler import LinearDecayScheduler
from neu.initializer import w_init
from neu.callbacks import spectral_norm_callback
from neu.yaml_wrapper import read_yaml, write_yaml
from neu.misc import AttrDict, get_current_time, init_nnabla, get_iteration_per_epoch
from neu.datasets.city_scapes import get_cityscape_datalist, create_data_iterator as create_cityscapes_iterator
from neu.datasets.ade20k import get_ade20k_datalist, create_data_iterator as create_ade20k_iterator
