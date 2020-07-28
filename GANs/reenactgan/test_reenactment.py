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
import numpy as np

import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as nm

import data
import models
from utils import combine_images
from config import load_decoder_config, load_encoder_config, load_transformer_config


def test(encoder_config, transformer_config, decoder_config,
         encoder_netG, transformer_netG, decoder_netG,
         src_celeb_name, trg_celeb_name,
         test_iterator, monitor,
         encoder_param_file, transformer_param_file, decoder_param_file):
    # prepare nn.Variable
    real_img = nn.Variable((1, 3, 256, 256))
    real_bod_map = nn.Variable((1, 15, 64, 64))
    real_bod_map_resize = nn.Variable((1, 15, 256, 256))

    # encoder
    with nn.parameter_scope(encoder_config["model_name"]):
        _, preds = encoder_netG(
            real_img,
            batch_stat=False,
            planes=encoder_config["model"]["planes"],
            output_nc=encoder_config["model"]["output_nc"],
            num_stacks=encoder_config["model"]["num_stacks"],
            activation=encoder_config["model"]["activation"],
        )
    preds.persistent = True
    preds_unlinked = preds.get_unlinked_variable()

    # load parameters of networks
    with nn.parameter_scope(encoder_config["model_name"]):
        nn.load_parameters(encoder_param_file)

    # transformer
    # Generator
    with nn.parameter_scope('netG_transformer'):
        with nn.parameter_scope('netG_A2B'):
            fake_bod_map = transformer_netG(
                preds, test=True, norm_type=transformer_config["norm_type"])
    with nn.parameter_scope('netG_transformer'):
        with nn.parameter_scope('netG_A2B'):
            nn.load_parameters(transformer_param_file)

    fake_bod_map.persistent = True
    fake_bod_map_unlinked = fake_bod_map.get_unlinked_variable()

    # decoder
    with nn.parameter_scope('netG_decoder'):
        fake_img = decoder_netG(fake_bod_map_unlinked, test=True)
    fake_img.persistent = True

    # load parameters of networks
    with nn.parameter_scope('netG_decoder'):
        nn.load_parameters(decoder_param_file)

    monitor_vis = nm.MonitorImage('result',
                                  monitor,
                                  interval=1,
                                  num_images=1,
                                  normalize_method=lambda x: x)

    # test
    num_test_batches = test_iterator.size
    for i in range(num_test_batches):
        _real_img, _, _real_bod_map_resize = test_iterator.next()

        real_img.d = _real_img
        real_bod_map_resize.d = _real_bod_map_resize

        # Generator
        preds.forward(clear_no_need_grad=True)
        fake_bod_map.forward(clear_no_need_grad=True)
        fake_img.forward(clear_no_need_grad=True)

        images_to_visualize = [real_img.d,
                               preds.d,
                               fake_bod_map.d,
                               fake_img.d,
                               real_bod_map_resize.d]
        visuals = combine_images(images_to_visualize)
        monitor_vis.add(i, visuals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder-config', default=None, type=str)
    parser.add_argument('--transformer-config', default=None, type=str)
    parser.add_argument('--decoder-config', default=None, type=str)

    parser.add_argument('--src-celeb-name', default=None, type=str)
    parser.add_argument('--trg-celeb-name', default=None, type=str)

    parser.add_argument('--encoder-param-file', default=None, type=str)
    parser.add_argument('--transformer-param-file', default=None, type=str)
    parser.add_argument('--decoder-param-file', default=None, type=str)
    parser.add_argument('--info', default=None, type=str)
    args = parser.parse_args()

    encoder_param_file = args.encoder_param_file
    transformer_param_file = args.transformer_param_file
    decoder_param_file = args.decoder_param_file

    encoder_config = load_encoder_config(args.encoder_config)
    transformer_config = load_transformer_config(args.transformer_config)
    decoder_config = load_decoder_config(args.decoder_config)

    src_celeb_name = args.src_celeb_name
    trg_celeb_name = args.trg_celeb_name

    assert trg_celeb_name == transformer_config["trg_celeb_name"], f"not trained on {trg_celeb_name}."

    if args.info:
        decoder_config["experiment_name"] += args.info

    #########################
    # Context Setting
    # Get context.
    from nnabla.ext_utils import get_extension_context
    logger.info(f'Running in {decoder_config["context"]}.')
    ctx = get_extension_context(
        decoder_config["context"], device_id=decoder_config["device_id"])
    nn.set_default_context(ctx)
    #########################

    # Data Loading
    logger.info('Initialing Datasource')
    test_iterator = data.celebv_data_iterator(dataset_mode="decoder",
                                              celeb_name=src_celeb_name,
                                              data_dir=decoder_config["train_dir"],
                                              ref_dir=decoder_config["ref_dir"],
                                              mode="test",
                                              batch_size=1,
                                              shuffle=False,
                                              with_memory_cache=decoder_config["test"]["with_memory_cache"],
                                              with_file_cache=decoder_config["test"]["with_file_cache"],
                                              )

    # Encoder
    encoder_netG = models.stacked_hourglass_net
    if not encoder_param_file:
        encoder_param_file = sorted(glob.glob(os.path.join(
                                    encoder_config["logdir"],
                                    encoder_config["dataset_mode"],
                                    encoder_config["experiment_name"],
                                    "model",
                                    "model_epoch-*")), key=os.path.getmtime)[-1]

    # Transformer
    transformer_netG = models.netG_transformer
    if not transformer_param_file:
        transformer_param_file = sorted(glob.glob(os.path.join(
                                   transformer_config["logdir"],
                                   transformer_config["dataset_mode"],
                                   f'{transformer_config["src_celeb_name"]}2{transformer_config["trg_celeb_name"]}',
                                   transformer_config["experiment_name"],
                                   "netG_transformer_A2B_*")), key=os.path.getmtime)[-1]

    # Decoder
    decoder_netG = models.netG_decoder
    if not decoder_param_file:
        decoder_param_file = sorted(glob.glob(os.path.join(
                                    decoder_config["logdir"],
                                    decoder_config["dataset_mode"],
                                    decoder_config["trg_celeb_name"],
                                    decoder_config["experiment_name"],
                                    "netG_*")), key=os.path.getmtime)[-1]

    monitor = nm.Monitor(os.path.join("reenactment_result",
                                      f'{src_celeb_name}2{decoder_config["trg_celeb_name"]}',
                                      decoder_config["experiment_name"]))

    test(encoder_config, transformer_config, decoder_config,
         encoder_netG, transformer_netG, decoder_netG,
         src_celeb_name, trg_celeb_name,
         test_iterator, monitor,
         encoder_param_file, transformer_param_file, decoder_param_file)


if __name__ == '__main__':
    main()
