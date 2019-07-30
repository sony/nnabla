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


def get_args(batch_size=64, image_size=32, n_classes=10, max_iter=100000, sample_size=50000):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    import os

    description = "Example of Self-Attention GAN (SAGAN)."
    parser = argparse.ArgumentParser(description)

    parser.add_argument("-d", "--device-id", type=str, default="0",
                        help="Device id.")
    parser.add_argument("-c", "--context", type=str, default="cudnn",
                        help="Context.")
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--image-size", type=int, default=image_size,
                        help="Image size.")
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size,
                        help="Batch size.")
    parser.add_argument("--max-iter", "-i", type=int, default=max_iter,
                        help="Max iterations.")
    parser.add_argument("--num-generation", "-n", type=int, default=1,
                        help="Number of iterations for generation.")
    parser.add_argument("--save-interval", type=int, default=sample_size // batch_size,
                        help="Interval for saving models.")
    parser.add_argument("--latent", type=int, default=128,
                        help="Number of latent variables.")
    parser.add_argument("--maps", type=int, default=128,
                        help="Number of latent variables.")
    parser.add_argument("--monitor-path", type=str, default="./result/example_0",
                        help="Monitor path.")
    parser.add_argument("--model-load-path", type=str,
                        help="Model load path to a h5 file used in generation and validation.")
    parser.add_argument("--lrg", type=float, default=1e-4,
                        help="Learning rate for generator")
    parser.add_argument("--lrd", type=float, default=1e-4,
                        help="Learning rate for discriminator")
    parser.add_argument("--n-critic", type=int, default=5,
                        help="Learning rate for discriminator")
    parser.add_argument("--beta1", type=float, default=0.5,
                        help="Beta1 of Adam solver.")
    parser.add_argument("--beta2", type=float, default=0.9,
                        help="Beta2 of Adam solver.")
    parser.add_argument("--lambda_", type=float, default=10.0,
                        help="Coefficient for gradient penalty.")
    parser.add_argument("--up", type=str,
                        choices=["nearest", "linear", "unpooling", "deconv"],
                        help="Upsample method used in the generator.")

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
