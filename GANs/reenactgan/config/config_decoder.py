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

import yaml
import datetime


def _get_default_config():
    config = dict()

    # Common Configuration
    config["train_dir"] = 'datasets/CelebV'
    config["logdir"] = 'tmp.monitor'

    dt = datetime.datetime.now()
    year = str(dt.year)
    month = str(dt.month).zfill(2)
    day = str(dt.day).zfill(2)
    hour = str(dt.hour).zfill(2)
    minute = str(dt.minute).zfill(2)
    config["experiment_name"] = f"{year}{month}{day}{hour}{minute}"

    config["context"] = "cudnn"
    config["device_id"] = 0
    config["trg_celeb_name"] = "Donald_Trump"

    config["dataset_mode"] = "decoder"
    config["ref_dir"] = ''
    # If you prepared the preprocessed data (stored as .npz),
    # you can use the following instead.
    # config["ref_dir"] = 'datasets/celebv_heatmaps_GT'

    config["isTrain"] = True
    config["mode"] = "all"  # ["train", "test", "all"]

    # Training
    config["train"] = dict()
    config["train"]["epochs"] = 150
    config["train"]["lr_decay_start_at"] = 100
    config["train"]["batch_size"] = 32
    config["train"]["lr"] = 0.0002
    config["train"]["beta1"] = 0.5

    # weight for L1 loss
    config["train"]["weight_L1"] = 100

    # VGG feature loss
    config["train"]["feature_loss"] = dict()
    config["train"]["feature_loss"]["lambda"] = 10

    config["train"]["shuffle"] = True
    config["train"]["with_memory_cache"] = False
    config["train"]["with_file_cache"] = False

    config["test"] = dict()
    config["test"]["batch_size"] = 1
    config["test"]["with_memory_cache"] = False
    config["test"]["with_file_cache"] = False
    config["test"]["logdir"] = 'tmp.result'
    config["test"]["vis_interval"] = 1

    config["monitor"] = dict()
    config["monitor"]["interval"] = 500
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


def load_decoder_config(path=None):
    config = _get_default_config()
    if path is not None:
        with open(path, 'r', encoding='utf-8') as fp:
            config_overwrite = yaml.load(fp, Loader=yaml.SafeLoader)
        _merge_config(config_overwrite, config)
    return config
