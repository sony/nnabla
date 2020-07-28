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
import glob
import argparse

import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as nm

import data
import models
from utils import combine_images
from config import load_transformer_config


def test_transformer(config, netG, train_iterators, monitor, param_file):

    netG_A2B = netG['netG_A2B']

    train_iterator_src, train_iterator_trg = train_iterators

    # Load boundary image to get Variable shapes
    bod_map_A = train_iterator_src.next()[0]
    bod_map_B = train_iterator_trg.next()[0]
    real_bod_map_A = nn.Variable(bod_map_A.shape)
    real_bod_map_B = nn.Variable(bod_map_B.shape)
    real_bod_map_A.persistent, real_bod_map_B.persistent = True, True

    ################### Graph Construction ####################
    # Generator
    with nn.parameter_scope('netG_transformer'):
        with nn.parameter_scope('netG_A2B'):
            fake_bod_map_B = netG_A2B(
                real_bod_map_A, test=True, norm_type=config["norm_type"])  # (1, 15, 64, 64)
    fake_bod_map_B.persistent = True

    # load parameters of networks
    with nn.parameter_scope('netG_transformer'):
        with nn.parameter_scope('netG_A2B'):
            nn.load_parameters(param_file)

    monitor_vis = nm.MonitorImage(
        'result', monitor, interval=config["test"]["vis_interval"], num_images=1, normalize_method=lambda x: x)

    # Test
    i = 0
    iter_per_epoch = train_iterator_src.size // config["test"]["batch_size"] + 1

    if config["num_test"]:
        num_test = config["num_test"]
    else:
        num_test = train_iterator_src.size

    for _ in range(iter_per_epoch):
        bod_map_A = train_iterator_src.next()[0]
        bod_map_B = train_iterator_trg.next()[0]
        real_bod_map_A.d, real_bod_map_B.d = bod_map_A, bod_map_B

        # Generate fake images
        fake_bod_map_B.forward(clear_buffer=True)

        i += 1

        images_to_visualize = [real_bod_map_A.d,
                               fake_bod_map_B.d, real_bod_map_B.d]
        visuals = combine_images(images_to_visualize)
        monitor_vis.add(i, visuals)

        if i > num_test:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default=None, type=str)
    parser.add_argument('--param-file', default=None, type=str)
    parser.add_argument('--num-test', '-n', default=None, type=int)
    args = parser.parse_args()
    param_file = args.param_file

    config = load_transformer_config(args.config)

    config["num_test"] = args.num_test

    #########################
    # Context Setting
    # Get context.
    from nnabla.ext_utils import get_extension_context
    logger.info(f'Running in {config["context"]}.')
    ctx = get_extension_context(
        config["context"], device_id=config["device_id"])
    nn.set_default_context(ctx)
    #########################

    # Data Loading
    logger.info('Initializing Datasource')
    train_iterator_src = data.celebv_data_iterator(dataset_mode="transformer",
                                                   celeb_name=config["src_celeb_name"],
                                                   data_dir=config["train_dir"],
                                                   ref_dir=config["ref_dir"],
                                                   mode="test",
                                                   batch_size=config["test"]["batch_size"],
                                                   shuffle=False,
                                                   with_memory_cache=config["test"]["with_memory_cache"],
                                                   with_file_cache=config["test"]["with_file_cache"],
                                                   resize_size=config["preprocess"]["resize_size"],
                                                   line_thickness=config["preprocess"]["line_thickness"],
                                                   gaussian_kernel=config["preprocess"]["gaussian_kernel"],
                                                   gaussian_sigma=config["preprocess"]["gaussian_sigma"]
                                                   )

    train_iterator_trg = data.celebv_data_iterator(dataset_mode="transformer",
                                                   celeb_name=config["trg_celeb_name"],
                                                   data_dir=config["train_dir"],
                                                   ref_dir=config["ref_dir"],
                                                   mode="test",
                                                   batch_size=config["test"]["batch_size"],
                                                   shuffle=False,
                                                   with_memory_cache=config["test"]["with_memory_cache"],
                                                   with_file_cache=config["test"]["with_file_cache"],
                                                   resize_size=config["preprocess"]["resize_size"],
                                                   line_thickness=config["preprocess"]["line_thickness"],
                                                   gaussian_kernel=config["preprocess"]["gaussian_kernel"],
                                                   gaussian_sigma=config["preprocess"]["gaussian_sigma"]
                                                   )
    train_iterators = (train_iterator_src, train_iterator_trg)

    # monitor
    monitor = nm.Monitor(os.path.join(
                    config["test"]["logdir"], "transformer",
                    f'{config["src_celeb_name"]}2{config["trg_celeb_name"]}', config["experiment_name"]))

    # Network
    netG = {'netG_A2B': models.netG_transformer,
            'netG_B2A': models.netG_transformer}
    if not param_file:
        param_file_A2B = sorted(glob.glob(os.path.join(
                                   config["logdir"],
                                   "transformer",
                                   f'{config["src_celeb_name"]}2{config["trg_celeb_name"]}',
                                   config["experiment_name"],
                                   "netG_transformer_A2B_*")), key=os.path.getmtime)[-1]
    else:
        param_file_A2B = param_file

    test_transformer(config, netG, train_iterators, monitor, param_file_A2B)


if __name__ == '__main__':
    main()
