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


def get_args(monitor_path='tmp.monitor', max_iter=234375, model_save_path=None, learning_rate=1e-3, batch_size=64, weight_decay=0, description=None):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    import os
    if model_save_path is None:
        model_save_path = monitor_path
    if description is None:
        description = "Examples on MNIST dataset. The following help shared among examples in this folder. Some arguments are valid or invalid in some examples."
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size)
    parser.add_argument("--learning-rate", "-l",
                        type=float, default=learning_rate)
    parser.add_argument("--monitor-path", "-m",
                        type=str, default=monitor_path,
                        help='Path monitoring logs saved.')
    parser.add_argument("--max-iter", "-i", type=int, default=max_iter,
                        help='Max iteration of training.')
    parser.add_argument("--val-interval", "-v", type=int, default=1000,
                        help='Validation interval.')
    parser.add_argument("--weight-decay", "-w",
                        type=float, default=weight_decay,
                        help='Weight decay factor of SGD update.')
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--model-save-path", "-o",
                        type=str, default=model_save_path,
                        help='Path the model parameters saved.')
    parser.add_argument("--model-load-path", "-T",
                        type=str, default=model_save_path,
                        help='Path the model parameters loaded.')
    parser.add_argument("--net", "-n", type=str,
                        default='cifar10_binary_connect_resnet23_prediction',
                        help="Neural network architecture type (used only in classification*.py)"
                        "classification.py: "
                        "'cifar10_resnet23_prediction'\n"
                        "'cifar10_binary_net_resnet23_prediction'\n"
                        "'cifar10_binary_weight_resnet23_prediction'\n"
                        "'cifar10_fp_connect_resnet23_prediction'\n"
                        "'cifar10_fp_net_resnet23_prediction'\n"
                        "'cifar10_pow2_connect_resnet23_prediction'\n"
                        "'cifar10_pow2_net_resnet23_prediction'\n"
                        "'cifar10_inq_resnet23_prediction\n'"
                        "'cifar10_min_max_resnet23_prediction\n'"
                        )
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension modules. ex) 'cpu', 'cudnn'.")
    parser.add_argument("--bit-width",
                        type=int, default=8,
                        help='Bitwidth used, corresponding to n.')
    parser.add_argument("--ql-min",
                        type=int, default=0,
                        help='Mininum quantization level of the min-max quantization.')
    parser.add_argument("--ql-max",
                        type=int, default=255,
                        help='Maximum quantization level of the min-max quantization.')
    parser.add_argument("--p-min-max", action="store_true",
                        help='Use the input min and max for the min-max quantization for weights and bias.')
    parser.add_argument("--a-min-max", action="store_true",
                        help='Use the input min and max for the min-max quantization for activation.')
    parser.add_argument("--a-ema", action="store_true",
                        help='Use the exponential moving average for the min-max quantization for activation.')
    parser.add_argument("--upper-bound",
                        type=int, default=1,
                        help='Upper bound for pow-of-2 quantization, corresponding to m')
    parser.add_argument("--delta",
                        type=float, default=2**-4,
                        help='Step size for fixed-point quantization, corresponding to delta')
    parser.add_argument("--ste-fine-grained",
                        type=bool, default=True,
                        help='Use the fine-grained STE for the fixed-point, pow2, and min-max quantization.')
    args = parser.parse_args()
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
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
