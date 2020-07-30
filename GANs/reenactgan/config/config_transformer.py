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

    # config["seed"] = 42
    config["context"] = "cudnn"
    config["device_id"] = 0
    config["src_celeb_name"] = "Kathleen"
    config["trg_celeb_name"] = "Donald_Trump"

    config["dataset_mode"] = "transformer"
    config["ref_dir"] = ''
    # If you prepared the preprocessed data (stored as .npz),
    # you can use the following instead.
    # config["ref_dir"] = 'datasets/celebv_heatmaps_GT'

    config["mode"] = "all"  # ["train", "test", "all"]

    config["norm_type"] = "batch_norm"  # ["batch_norm", "instance_norm"]

    # Training
    config["train"] = dict()
    config["train"]["epochs"] = 200
    config["train"]["batch_size"] = 32
    config["train"]["lr"] = 0.00005
    config["train"]["beta1"] = 0.5
    config["train"]["beta2"] = 0.999
    config["train"]["weight_decay"] = 0.0001

    # cycle loss
    config["train"]["cycle_loss"] = dict()
    config["train"]["cycle_loss"]["lambda"] = 50.0

    # shape loss
    config["train"]["shape_loss"] = dict()
    config["train"]["shape_loss"]["lambda"] = 10.0
    config["train"]["shape_loss"]["align_param_path"] = "./Align_40000.h5"
    config["train"]["shape_loss"]["PCA_param_path"] = "./PCA_weights.h5"

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

    config["preprocess"] = dict()
    config["preprocess"]["resize_size"] = 64
    config["preprocess"]["line_thickness"] = 3
    config["preprocess"]["gaussian_kernel"] = 5
    config["preprocess"]["gaussian_sigma"] = 3

    return config


def _merge_config(src, dst):
    if not isinstance(src, dict):
        return

    for k, v in src.items():
        if isinstance(v, dict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load_transformer_config(path=None):
    config = _get_default_config()
    if path is not None:
        with open(path, 'r', encoding='utf-8') as fp:
            config_overwrite = yaml.load(fp, Loader=yaml.SafeLoader)
        _merge_config(config_overwrite, config)
    return config
