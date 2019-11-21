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

from .lr_scheduler import LearningRateScheduler

import sys
import os

common_utils_path = os.path.join("/", *os.path.abspath(__file__).split("/")[:-4], "utils")
sys.path.append(common_utils_path)

from reporter import Reporter
from post_processing import Colorize
from variable_utils import set_persistent_all, get_params_startswith
from yaml_wrapper import read_yaml, write_yaml
from misc import init_nnabla, get_current_time, AttrDict
from losses import get_gan_loss, vgg16_perceptual_loss

from datasets.city_scapes import (create_data_iterator as create_cityscapes_iterator,
                                  get_cityscape_datalist,
                                  load_function as cityscapes_load_function)
