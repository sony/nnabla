# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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
import yaml
import datetime


def _get_default_config():
    config = dict()

    config["logdir"] = './tmp.monitor'
    dt = datetime.datetime.now()
    year = str(dt.year)
    month = str(dt.month).zfill(2)
    day = str(dt.day).zfill(2)
    hour = str(dt.hour).zfill(2)
    minute = str(dt.minute).zfill(2)
    config["experiment_name"] = f"{year}{month}{day}{hour}{minute}"

    config["seed"] = 42
    config["context"] = "cudnn"
    config["device_id"] = 0

    config["dataset_mode"] = "encoder"
    config["use_reference"] = True

    config["path"] = dict()
    config["path"]["data_dir"] = "datasets/WFLW_heatmaps"

    config["model_name"] = 'model'
    config["model"] = dict()
    config["model"]["planes"] = 64
    config["model"]["output_nc"] = 15
    config["model"]["num_stacks"] = 2
    config["model"]["activation"] = 'tanh'

    config["finetune"] = dict()
    config["finetune"]["param_path"] = "ReenactGAN_encoder_pretrained_weights.h5"

    config["loss_name"] = 'mse'
    config["loss"] = dict()

    config["train"] = dict()
    config["train"]["epochs"] = 500
    config["train"]["batch_size"] = 32
    config["train"]["lr"] = 1e-3
    config["train"]["weight_decay"] = 1e-5
    config["train"]["with_memory_cache"] = False
    config["train"]["with_file_cache"] = False
    config["train"]["augmentation"] = False

    config["test"] = dict()
    config["test"]["batch_size"] = 32
    config["test"]["with_memory_cache"] = False
    config["test"]["with_file_cache"] = False

    config["monitor"] = dict()
    config["monitor"]["interval"] = 1
    config["monitor"]["save_interval"] = 10

    return config


def _merge_config(src, dst):
    if not isinstance(src, dict):
        return

    for k, v in src.items():
        if isinstance(v, dict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load_encoder_config(path=None):
    config = _get_default_config()
    if path is not None:
        with open(path, 'r', encoding='utf-8') as fp:
            config_overwrite = yaml.load(fp, Loader=yaml.SafeLoader)
        _merge_config(config_overwrite, config)
    return config
