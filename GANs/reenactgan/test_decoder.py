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
from config import load_decoder_config


def test(config, netG, train_iterator, monitor, param_file):
    # Load image and boundary image to get Variable shapes
    img, bod_map, bod_map_resize = train_iterator.next()

    real_img = nn.Variable(img.shape)
    real_bod_map = nn.Variable(bod_map.shape)
    real_bod_map_resize = nn.Variable(bod_map_resize.shape)

    ################### Graph Construction ####################
    # Generator
    with nn.parameter_scope('netG_decoder'):
        fake_img = netG(real_bod_map, test=False)
    fake_img.persistent = True

    # load parameters of networks
    with nn.parameter_scope('netG_decoder'):
        nn.load_parameters(param_file)

    monitor_vis = nm.MonitorImage(
        'result', monitor, interval=config["test"]["vis_interval"], num_images=4, normalize_method=lambda x: x)

    # Test
    i = 0
    iter_per_epoch = train_iterator.size // config["test"]["batch_size"] + 1

    if config["num_test"]:
        num_test = config["num_test"]
    else:
        num_test = train_iterator.size

    for _ in range(iter_per_epoch):
        img, bod_map, bod_map_resize = train_iterator.next()

        real_img.d = img
        real_bod_map.d = bod_map
        real_bod_map_resize.d = bod_map_resize

        # Generate fake image
        fake_img.forward(clear_buffer=True)

        i += 1

        images_to_visualize = [real_bod_map_resize.d, fake_img.d, img]
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

    config = load_decoder_config(args.config)

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
    logger.info('Initialing Datasource')
    train_iterator = data.celebv_data_iterator(dataset_mode="decoder",
                                               celeb_name=config["trg_celeb_name"],
                                               data_dir=config["train_dir"],
                                               ref_dir=config["ref_dir"],
                                               mode="test",
                                               batch_size=config["test"]["batch_size"],
                                               shuffle=False,
                                               with_memory_cache=config["test"]["with_memory_cache"],
                                               with_file_cache=config["test"]["with_file_cache"],
                                               )

    monitor = nm.Monitor(os.path.join(
                    config["test"]["logdir"], "decoder", config["trg_celeb_name"], config["experiment_name"]))

    # Network
    netG = models.netG_decoder
    if not param_file:
        param_file = sorted(glob.glob(os.path.join(
                                      config["logdir"],
                                      "decoder",
                                      config["trg_celeb_name"],
                                      config["experiment_name"],
                                      "netG_*")), key=os.path.getmtime)[-1]

    test(config, netG, train_iterator, monitor, param_file)


if __name__ == '__main__':
    main()
