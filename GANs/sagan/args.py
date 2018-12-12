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


def get_args(batch_size=32, accum_grad=2, image_size=128, n_classes=1000, max_iter=500000):
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
    parser.add_argument("--n-classes", type=int, default=n_classes,
                        help="Image size.")
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size,
                        help="Batch size.")
    parser.add_argument("--accum-grad", "-a", type=int, default=accum_grad,
                        help="Batch size.")
    parser.add_argument("--max-iter", "-i", type=int, default=max_iter,
                        help="Max iterations.")
    parser.add_argument("--save-interval", type=int,
                        help="Interval for saving models.")
    parser.add_argument("--latent", type=int, default=128,
                        help="Number of latent variables.")
    parser.add_argument("--maps", type=int, default=1024,
                        help="Number of latent variables.")
    parser.add_argument("--not-sn", action='store_false',
                        help="Not use the spectral normalization")
    parser.add_argument("--monitor-path", type=str, default="./result/example_0",
                        help="Monitor path.")
    parser.add_argument("--model-load-path", type=str,
                        help="Model load path to a h5 file used in generation and validation.")
    parser.add_argument("--lrg", type=float, default=1e-4, #4e-4
                        help="Learning rate for generator")
    parser.add_argument("--lrd", type=float, default=4e-4,
                        help="Learning rate for discriminator")
    parser.add_argument("--n-critic", type=int, default=1,
                        help="Learning rate for discriminator")
    parser.add_argument("--beta1", type=float, default=0.0,
                        help="Beta1 of Adam solver.")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Beta2 of Adam solver.")
    parser.add_argument("--train-dir", "-T", type=str, default="",
                        help='Training data directory')
    parser.add_argument("--valid-dir", "-V", type=str, default="",
                        help='Validation data directory')
    parser.add_argument("--dirname-to-label-path", "-L", type=str, default="",
                        help='Dirname-to-label path')
    parser.add_argument("--sync-weight-every-itr",
                        type=int, default=1,
                        help="Sync weights every specified iteration. NCCL uses the ring all reduce, so gradients in each device are not exactly same. When it is accumulated in the weights, the weight values in each device diverge.")

    # Generation
    parser.add_argument("--tile-output", action='store_true',
                        help="Generated images are tiled in one image.")
    parser.add_argument("--class-id", type=int, default=-1,
                        help="Class id in the imagenet dataset. See *.txt file in the directory.")
    parser.add_argument("--generate-all", action='store_true',
                        help="Generate images for all classes.")
    parser.add_argument("--truncation-threshold", type=float, default=float("inf"),  
                        help="Threshold of the truncation trick.")

    # Morphing
    parser.add_argument("--from-class-id", type=int, default=947,
                        help="Class id used for class morphing in the imagenet dataset. See *.txt file in the directory.")
    parser.add_argument("--to-class-id", type=int, default=153,
                        help="Class id used for class morphing in the imagenet dataset. See *.txt file in the directory.")
    parser.add_argument("--n-morphs", type=int, default=8,
                        help="Number of morphing.")
    
    # Validation
    parser.add_argument("--val-iter", "-v", type=int, default=10000,
                        help="Max iterations for validation.")
    parser.add_argument("--nnp-inception-model-load-path", type=str,
                        help="Inception model load path to a NNP file used in validation. This model is not limited to using Inception models, one can use other models, e.g., VGG-16 or ResNet-50.")
    parser.add_argument("--evaluation-metric", type=str, default="IS",
                        choices=["IS", "FID"],
                        help="Validation metric for SAGAN; IS or FID, FID is the default.")
    parser.add_argument("--variable-name", type=str, default="",
                        help="Variable name to get. E.g., `VGG16/Affine` for VGG16, `AveragePooling_2` for Inception-V3, and `AveragePooling` for ResNet-50.")
    parser.add_argument("--nnp-preprocess", action='store_true',
                        help="Preprocess a given prediction model using scale and bias. Default scale and bias are 0.01753 and -1.99")

    # Matching
    parser.add_argument("--top-n", type=int, default=15,
                        help="Top-n images using a criteria of VGG feature mathing.")
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
