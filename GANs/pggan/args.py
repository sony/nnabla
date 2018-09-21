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


def get_args(batch_size=16):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    import os

    description = "Example of Progressive Growing of GANs."
    parser = argparse.ArgumentParser(description)

    parser.add_argument("-d", "--device-id", type=int, default=0,
                        help="Device id.")
    parser.add_argument("-c", "--context", type=str, default="cudnn",
                        help="Context.")
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size,
                        help="Batch size.")
    parser.add_argument("--img-path", type=str,
                        default="~/img_align_celeba_png",
                        help="Image path.")
    parser.add_argument("--dataset-name", type=str, default="CelebA",
                        choices=["CelebA"],
                        help="Dataset name used.")
    parser.add_argument("--save-image-interval", type=int, default=1,
                        help="Interval for saving images.")
    parser.add_argument("--epoch-per-resolution", type=int, default=4,
                        help="Number of epochs per resolution.")
    parser.add_argument("--imsize", type=int, default=128,
                        help="Input image size.")
    parser.add_argument("--train-samples", type=int, default=-1,
                        help="Number of data to be used. When -1 is set all data is used.")
    parser.add_argument("--valid-samples", type=int, default=16384,
                        help="Number of data used in validation.")
    parser.add_argument("--latent", type=int, default=512,
                        help="Number of latent variables.")
    parser.add_argument("--critic", type=int, default=1,
                        help="Number of critics.")
    parser.add_argument("--monitor-path", type=str, default="./result/example_0",
                        help="Monitor path.")
    parser.add_argument("--model-load-path", type=str,
                        default="./result/example_0/Gen_phase_128_epoch_4.h5",
                        help="Model load path used in generation and validation.")
    parser.add_argument("--use-bn", action='store_true',
                        help="Use batch normalization.")
    parser.add_argument("--use-ln", action='store_true',
                        help="Use layer normalization.")
    parser.add_argument("--not-use-wscale", action='store_false',
                        help="Not use the equalized learning rate.")
    parser.add_argument("--use-he-backward", action='store_true',
                        help="Use the He initialization using the so-called `fan_in`. Default is the backward.")
    parser.add_argument("--leaky-alpha", type=float, default=0.2,
                        help="Leaky alpha value.")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.0,
                        help="Beta1 of Adam solver.")
    parser.add_argument("--beta2", type=float, default=0.99,
                        help="Beta2 of Adam solver.")
    parser.add_argument("--l2-fake-weight", type=float, default=0.1,
                        help="Weight for the fake term in the discriminator loss in LSGAN.")
    parser.add_argument("--hyper-sphere", action='store_true',
                        help="Latent vector lie in the hyper sphere.")
    parser.add_argument("--last-act", type=str, default="tanh",
                        choices=["tanh"],
                        help="Last activation of the generator.")
    parser.add_argument("--validation-metric", type=str, default="swd",
                        choices=["swd", "ms-ssim"],
                        help="Validation metric for PGGAN.")

    args = parser.parse_args()

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
