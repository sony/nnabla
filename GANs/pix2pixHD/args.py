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

import os
import argparse


def summarize_args(args):
    with open(os.path.join(args.save_path, "args.txt"), "w") as f:
        f.write("\n".join(["{}: {}".format(k, v)
                           for k, v in args.__dict__.items()]))


def base_parser():
    parser = argparse.ArgumentParser()

    # args for nnabla config
    parser.add_argument("--context", "-c", default="cudnn", type=str)
    parser.add_argument("--device-id", "-d", default="0", type=str)
    parser.add_argument("--type-config", "-t", default="float", type=str)

    # args for data config
    parser.add_argument("--batch-size", "-b", default=1, type=int)
    parser.add_argument("--data-dir", default="./data", type=str)

    # args for generator settings
    parser.add_argument("--g-n-scales", "-G", default=1, type=int)
    parser.add_argument("--use-enc", action="store_true")
    parser.add_argument("--n-label-ids", default=35, type=int)
    # args for saving
    parser.add_argument("--save-path", "-S", default="./result", type=str)

    return parser


def base_post_process(parser):
    # post-process
    args = parser.parse_args()

    # init output dir
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    return args


def get_training_args():
    parser = base_parser()

    parser.add_argument("--load-path", "-L", default=None, type=str)

    # args for network and training
    parser.add_argument("--epoch", "-e", default=200, type=int)
    parser.add_argument("--d-n-scales", "-D", default=2, type=int)
    parser.add_argument("--base-lr", "-l", default=2e-4, type=float)
    parser.add_argument("--without-flip", action="store_true")
    parser.add_argument("--fix-global-epoch", "-f", default=0, type=int)

    args = base_post_process(parser)

    summarize_args(args)

    return args


def get_generation_args():
    parser = base_parser()

    parser.add_argument("--load-path", "-L", required=True, type=str)

    args = base_post_process(parser)

    if args.load_path is None:
        raise ValueError("you must be pa")

    return args
