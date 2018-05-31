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


def get_args(monitor_path='tmp.monitor', max_epoch=200, model_save_path=None,
             learning_rate=2*1e-4,
             batch_size=1, description=None):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    import os
    if model_save_path is None:
        model_save_path = monitor_path
    if description is None:
        description = ("NNabla implementation of CycleGAN. The following help shared among examples in this folder. "
                       "Some arguments are valid or invalid in some examples.")
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size)
    parser.add_argument("--learning-rate", "-l",
                        type=float, default=learning_rate)
    parser.add_argument("--monitor-path", "-m",
                        type=str, default=monitor_path,
                        help='Path monitoring logs saved.')
    parser.add_argument("--max-epoch", "-e", type=int, default=max_epoch,
                        help='Max epoch of training. Epoch is determined by the max of the number of images for two domains.')
    parser.add_argument("--device-id", "-d", type=int, default=0,
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--model-save-path", "-o",
                        type=str, default=model_save_path,
                        help='Path where model parameters are saved.')
    parser.add_argument("--model-load-path",
                        type=str,
                        help='Path where model parameters are loaded.')
    parser.add_argument("--dataset",
                        type=str, default="horse2zebra", choices=["ae_photos",
                                                                  "apple2orange",
                                                                  "cezanne2photo",
                                                                  "cityscapes",
                                                                  "facades",
                                                                  "horse2zebra",
                                                                  "iphone2dslr_flower",
                                                                  "maps",
                                                                  "monet2photo",
                                                                  "summer2winter_yosemite",
                                                                  "ukiyoe2photo",
                                                                  "vangogh2photo"],
                        help='Dataset to be used.')
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension modules. ex) 'cpu', 'cudnn'.")
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type configuration (float or half)')
    parser.add_argument('--lambda-recon', type=float,
                        default=10.,
                        help="Coefficient for reconstruction loss.")
    parser.add_argument('--lambda-idt', type=float,
                        default=0,
                        help="Coefficient for identity loss. Default is 0, but set 0.5 to comply with the pytorch cycle-gan implementation.")
    parser.add_argument('--unpool', action='store_true')
    parser.add_argument('--init-method', default=None, type=str,
                        help="`None`|`paper`")

    args = parser.parse_args()
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    return args


def save_args(args):
    from nnabla import logger
    import os
    if not os.path.exists(args.monitor_path):
        os.makedirs(args.monitor_path)

    path = "{}/Arguments.txt".format(args.monitor_path)
    logger.info("Arguments are saved to {}.".format(path))
    with open(path, "w") as fp:
        for k, v in sorted(vars(args).items()):
            logger.info("{}={}".format(k, v))
            fp.write("{}={}\n".format(k, v))
