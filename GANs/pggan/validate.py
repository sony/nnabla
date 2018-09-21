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


import argparse
from nnabla import logger
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from scipy import ndimage
import time

from args import get_args
from datasets import data_iterator
from functions import pixel_wise_feature_vector_normalization
from helpers import load_gen
import matplotlib.pylab as plt
from networks import Generator
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np


def main():
    # Args
    args = get_args()

    # Context
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    logger.info(ctx)
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)

    # Monitor
    monitor = Monitor(args.monitor_path)

    # Validation
    logger.info("Start validation")

    num_images = args.valid_samples
    num_batches = num_images // args.batch_size

    # DataIterator
    di = data_iterator(args.img_path, args.batch_size,
                       imsize=(args.imsize, args.imsize),
                       num_samples=args.valid_samples,
                       dataset_name=args.dataset_name)
    # generator
    gen = load_gen(args.model_load_path, use_bn=args.use_bn, last_act=args.last_act,
                   use_wscale=args.not_use_wscale, use_he_backward=args.use_he_backward)

    # compute metric
    if args.validation_metric == "ms-ssim":
        logger.info("Multi Scale SSIM")
        monitor_time = MonitorTimeElapsed(
            "MS-SSIM-ValidationTime", monitor, interval=1)
        monitor_metric = MonitorSeries("MS-SSIM", monitor, interval=1)
        from ms_ssim import compute_metric
        score = compute_metric(gen, args.batch_size,
                               num_images, args.latent, args.hyper_sphere)
        monitor_time.add(0)
        monitor_metric.add(0, score)
    elif args.validation_metric == "swd":
        logger.info("Sliced Wasserstein Distance")
        monitor_time = MonitorTimeElapsed(
            "SWD-ValidationTime", monitor, interval=1)
        monitor_metric = MonitorSeries("SWD", monitor, interval=1)
        nhoods_per_image = 128
        nhood_size = 7
        level_list = [128, 64, 32, 16]  # TODO: use argument
        dir_repeats = 4
        dirs_per_repeat = 128
        from sliced_wasserstein import compute_metric
        score = compute_metric(di, gen, args.latent, num_batches, nhoods_per_image, nhood_size,
                               level_list, dir_repeats, dirs_per_repeat, args.hyper_sphere)
        monitor_time.add(0)
        monitor_metric.add(0, score)  # averaged in the log
    else:
        logger.info("Set `validation-metric` as either `ms-ssim` or `swd`.")
    logger.info(score)
    logger.info("End validation")


if __name__ == "__main__":
    main()
