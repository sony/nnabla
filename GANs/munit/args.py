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


def get_args(monitor_path="tmp.monitor"):
    import argparse
    import os
    description = ("NNabla implementation of AugmentedCycleGAN")
    parser = argparse.ArgumentParser(description)
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension modules. ex) 'cpu', 'cudnn'.")
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type configuration (float or half)')
    parser.add_argument("--batch-size", "-b", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--monitor-path", "-m",
                        type=str, default=monitor_path,
                        help='Path monitoring logs saved.')
    parser.add_argument("--max-iter", "-i", type=int, default=1000000,
                        help='Max iterations.')
    parser.add_argument("--device-id", "-d", type=int, default=0,
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--model-save-interval",
                        type=float, default=100000,
                        help='Model save interval')
    parser.add_argument("--model-save-path", "-o",
                        type=str, default=monitor_path,
                        help='Path where model parameters are saved.')
    parser.add_argument("--model-load-path",
                        type=str,
                        help='Path where model parameters are loaded.')
    parser.add_argument("--inception-model-load-path",
                        type=str,
                        help='Path where inception model parameters are loaded. One does not necessarily use the inception-model, e.g., InceptionV3.')
    parser.add_argument("--img-path-a",
                        type=str,
                        help='Path to dataset in domain A')
    parser.add_argument("--img-path-b",
                        type=str,
                        help='Path to dataset in domain B')
    parser.add_argument("--img-files-a",
                        type=str, nargs='+',
                        help='Files to dataset in domain A')
    parser.add_argument("--img-files-b",
                        type=str, nargs='+',
                        help='Files to dataset in domain B')
    parser.add_argument("--dataset",
                        type=str, default="edges2shoes", choices=["edges2handbags",
                                                                  "edges2shoes",
                                                                  "cityscapes",
                                                                  "facades",
                                                                  "maps"],
                        help='Dataset to be used.')
    parser.add_argument("--data-type", type=str, default="train", choices=["train", "val"],
                        help='Prepare train or test dataset.')

    parser.add_argument('--tile-images', default=16,
                        help="Number of tiled images.")
    parser.add_argument('--num-repeats', type=int, default=3,
                        help="Number of repeats of using the same image for stochastic generation.")
    parser.add_argument('--maps', default=64,
                        help="Number of basis of maps.")
    parser.add_argument('--num-samples', default=-1, type=int,
                        help="Number of samples to be used. Default is -1, using all data.")
    parser.add_argument('--lr_g', default=1e-4, type=float,
                        help="Learning rate for generator")
    parser.add_argument('--lr_d', default=1e-4, type=float,
                        help="Learning rate for discriminator")
    parser.add_argument('--beta1', default=0.5, type=float,
                        help="Beta1 of Adam")
    parser.add_argument('--beta2', default=0.999, type=float,
                        help="Beta2 of Adam")
    parser.add_argument('--lambda_x', default=10.0, type=float,
                        help="Weight for the within-domain reconstruction term")
    parser.add_argument('--lambda_c', default=1.0, type=float,
                        help="Weight for the content reconstruction term")
    parser.add_argument('--lambda_s', default=1.0, type=float,
                        help="Weight for the style reconstruction term")
    parser.add_argument('--weight-decay-rate', default=1e-4, type=float,
                        help="Weight decay rate")
    parser.add_argument('--lr-decay-rate', default=0.5, type=float,
                        help="Learning decay rate")
    parser.add_argument("--lr-decay-at-every", type=int, default=100000,
                        help='Linear rate decay is performed at every `this` iterations')
    parser.add_argument("--example-guided", action="store_true",
                        help='Use style encodoing in the other domain.')
    parser.add_argument("--seed", default=412, type=int,
                        help='Use style encodoing in the other domain.')

    args = parser.parse_args()
    return args


def save_args(args, mode="train"):
    from nnabla import logger
    import os
    if not os.path.exists(args.monitor_path):
        os.makedirs(args.monitor_path)

    path = "{}/Arguments-{}.txt".format(args.monitor_path, mode)
    logger.info("Arguments are saved to {}.".format(path))
    with open(path, "w") as fp:
        for k, v in sorted(vars(args).items()):
            logger.info("{}={}".format(k, v))
            fp.write("{}={}\n".format(k, v))
