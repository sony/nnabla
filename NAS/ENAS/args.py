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


def get_macro_args():
    """
    Get command line arguments for macro search.

    Arguments set the default values of command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser()

    # General setting
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension path. ex) cpu, cudnn.")
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--recommended-arch", type=str, default="macro_arch.npy",
                        help='name of the npy file which contains the recommended architecture by the trained controller.')

    # Controller-related
    parser.add_argument("--max-search-iter", "-i", type=int, default=350)
    parser.add_argument("--num-candidate", "-C", type=int, default=10)
    parser.add_argument("--early-stop-over", type=float, default=1.0,
                        help='If valid accuracy is more than this value, architecture search finishes.')
    parser.add_argument("--lstm-size", type=int, default=32)
    parser.add_argument("--state-size", type=int, default=32)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--skip-prob", type=float, default=0.8)
    parser.add_argument("--skip-weight", type=float, default=1.5)
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--tanh-constant", type=float, default=1.5)
    parser.add_argument("--baseline-decay", type=float, default=0.999)
    parser.add_argument("--num-ops", type=int, default=6,
                        help='change this value only when you add a operation to make the controller choose.')
    parser.add_argument("--control-lr", type=float, default=0.001)
    parser.add_argument("--select-strategy", type=str, choices=["best", "last"], default="best",
                        help='Architecture selection strategy, either "best" or "last".')
    parser.add_argument("--use-variance-reduction", type=bool, default=False)
    parser.add_argument("--sampling-only", type=bool, default=False)
    parser.add_argument("--num-sampling", type=int, default=5)

    # CNN-related
    # basic config. mainly for CNN training during architecture search.
    parser.add_argument("--use-sparse", type=bool, default=True,
                        help='Only for test. If True, no skip connections are made.')
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--output-filter", type=int, default=36,
                        help='Number of output filters of CNN (used during architecture search), must be even number.')
    parser.add_argument("--epoch-per-search", "-e", type=int, default=2,
                        help='Number of epochs used for CNN training during architecture search,')

    parser.add_argument("--additional_filters_on_retrain", "-f", type=int, default=60,
                        help='Number of additional output filters of CNN (used when CNN retraining), must be even number.')
    parser.add_argument("--epoch-on-retrain", "-r", type=int, default=350,
                        help='Number of epochs used for CNN retraining after architecture search.')

    # gradient clip
    parser.add_argument("--with-grad-clip-on-search", type=bool, default=False)
    parser.add_argument("--with-grad-clip-on-retrain", type=bool, default=True)
    parser.add_argument("--grad-clip-value", "-g", type=float, default=5.0)

    # weight_decay
    parser.add_argument("--weight-decay", "-w", type=float, default=0.00025,
                        help='Weight decay rate. Weight decay is executed by default. Set it 0 to virtually disable it.')

    # learning rate and its control
    parser.add_argument("--child-lr", "-clr", type=float, default=0.1)
    parser.add_argument("--lr-control-on-search", type=bool, default=False,
                        help='whether or not use learning rate controller on CNN training when architecture search.')
    parser.add_argument("--lr-control-on-retrain", type=bool, default=True,
                        help='whether or not use learning rate controller on CNN training when retraining.')

    # misc
    parser.add_argument("--val-iter", "-j", type=int,
                        default=100, help='number of the validation.')
    parser.add_argument("--monitor-path", "-m", type=str,
                        default='tmp.macro_monitor')
    parser.add_argument("--model-save-interval", "-s", type=int, default=1000)
    parser.add_argument("--model-save-path", "-o",
                        type=str, default='tmp.macro_monitor')

    return parser.parse_args()


def get_micro_args():
    """
    Get command line arguments for micro search.

    Arguments set the default values of command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser()
    # General setting
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension path. ex) cpu, cudnn.")
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--recommended-arch", type=str, default="micro_arch.npy",
                        help='name of the npy file which contains the recommended architecture by the trained controller.')

    # Controller-related
    parser.add_argument("--max-search-iter", "-i", type=int, default=350)
    parser.add_argument("--num-candidate", "-C", type=int, default=10)
    parser.add_argument("--early-stop-over", type=float, default=1.0,
                        help='If valid accuracy is more than this value, architecture search finishes.')
    parser.add_argument("--lstm-size", type=int, default=32)
    parser.add_argument("--state-size", type=int, default=32)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--skip-prob", type=float, default=0.8)
    parser.add_argument("--skip-weight", type=float, default=1.5)
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--tanh-constant", type=float, default=1.5)
    parser.add_argument("--op-tanh-reduce", type=float, default=1.0)
    parser.add_argument("--baseline-decay", type=float, default=0.999)
    parser.add_argument("--num-ops", type=int, default=5,
                        help='change this value only when you add a operation to make the controller choose.')
    parser.add_argument("--control-lr", type=float, default=0.001)
    parser.add_argument("--select-strategy", type=str, choices=["best", "last"], default="best",
                        help='Architecture selection strategy, either "best" or "last".')
    parser.add_argument("--use-variance-reduction", type=bool, default=False)
    parser.add_argument("--sampling-only", type=bool, default=False)
    parser.add_argument("--num-sampling", type=int, default=5)

    # CNN-related
    # basic config. mainly for CNN training during architecture search.
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--num-cells", type=int, default=6)
    parser.add_argument("--num-nodes", type=int, default=7,
                        help='Number of nodes per cell, must be more than 2.')
    parser.add_argument("--output-filter", type=int, default=20,
                        help='Number of output filters of CNN (used during architecture search), must be even number.')
    parser.add_argument("--epoch-per-search", "-e", type=int, default=2,
                        help='Number of epochs used for CNN training during architecture search,')

    parser.add_argument("--additional_filters_on_retrain", "-f", type=int, default=60,
                        help='Number of additional output filters of CNN (used when CNN retraining), must be even number.')
    parser.add_argument("--epoch-on-retrain", "-r", type=int, default=350,
                        help='Number of epochs used for CNN retraining after architecture search.')

    # gradient clip
    parser.add_argument("--with-grad-clip-on-search", type=bool, default=False)
    parser.add_argument("--with-grad-clip-on-retrain", type=bool, default=True)
    parser.add_argument("--grad-clip-value", "-g", type=float, default=5.0)

    # weight_decay
    parser.add_argument("--weight-decay", "-w", type=float, default=0.00025,
                        help='Weight decay rate. Weight decay is executed by default. Set it 0 to virtually disable it.')

    # learning rate and its control
    parser.add_argument("--child-lr", "-clr", type=float, default=0.1)
    parser.add_argument("--lr-control-on-search", type=bool, default=False,
                        help='whether or not use learning rate controller on CNN training when architecture search.')
    parser.add_argument("--lr-control-on-retrain", type=bool, default=True,
                        help='whether or not use learning rate controller on CNN training when retraining.')

    # misc
    parser.add_argument("--val-iter", "-j", type=int,
                        default=100, help='number of the validation.')
    parser.add_argument("--monitor-path", "-m", type=str,
                        default='tmp.micro_monitor')
    parser.add_argument("--model-save-interval", "-s", type=int, default=1000)
    parser.add_argument("--model-save-path", "-o",
                        type=str, default='tmp.micro_monitor')

    return parser.parse_args()
