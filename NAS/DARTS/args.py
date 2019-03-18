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


def get_args():
    """
    Get command line arguments for architecture search and evaluation.
    """

    import argparse
    parser = argparse.ArgumentParser()
    # General settings
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        help="Extension path. ex) cpu, cudnn.")
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. \
                        This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--epoch", "-e", type=int, default=50)

    # architecture-parameter-related
    parser.add_argument("--second-order", type=bool, default=False,
                        help='whether or not using second order derivative \
                        (approximation) when updating architecture parameters.')
    parser.add_argument("--arch-lr", type=float, default=3e-4)
    parser.add_argument("--eta", type=float, default=0.025,
                        help='coefficient for second order derivative.')

    # model-parameter-related
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--num-cells", type=int, default=8)
    parser.add_argument("--num-nodes", type=int, default=7,
                        help='Number of nodes per cell, must be more than 2.')
    parser.add_argument("--output-filter", type=int, default=16,
                        help='Number of output filters of CNN, must be even number.')
    parser.add_argument("--additional-filters-on-retrain", "-f", type=int, default=20,
                        help='Number of additional output filters of CNN \
                        (used when CNN retraining), must be even number.')
    parser.add_argument("--dropout-rate", type=float, default=0.2)

    # learning rate and its control
    parser.add_argument("--model-lr", type=float, default=0.025)
    parser.add_argument("--lr-control-model", type=bool, default=True,)

    # gradient clip model
    parser.add_argument("--with-grad-clip-model", type=bool, default=True)
    parser.add_argument("--grad-clip-value-model", type=float, default=5.0)

    # gradient clip arch
    parser.add_argument("--with-grad-clip-archs", type=bool, default=False)
    parser.add_argument("--grad-clip-value-archs", type=float, default=5.0)

    # auxiliary tower
    parser.add_argument('--auxiliary', default=True,
                        help='Whether to use auxiliary tower or not. \
                        Used when evaluation.')
    parser.add_argument('--auxiliary-weight', type=float, default=0.4,
                        help='Weight for auxiliary loss')

    # cutout
    parser.add_argument("--cutout", type=bool, default=False,
                        help='Whether to use cutout or not in evaluation process.')
    parser.add_argument("--cutout-length", type=int, default=16,
                        help='height and width of a mask used for cutout.')

    # weight_decay
    parser.add_argument("--weight-decay-model", type=float, default=3e-4,
                        help='Weight decay rate. Weight decay is executed by default. \
                        Set it 0 to virtually disable it.')
    parser.add_argument("--weight-decay-archs", type=float, default=1e-3,
                        help='Weight decay rate. Weight decay is executed by default. \
                        Set it 0 to virtually disable it.')
    # misc
    parser.add_argument("--monitor-path", "-m",
                        type=str, default='tmp.monitor')
    parser.add_argument("--model-save-interval", "-s", type=int, default=1000)
    parser.add_argument("--model-save-path", "-o",
                        type=str, default='tmp.monitor')
    parser.add_argument("--model-arch-name", type=str,
                        default='DARTS_arch.json')

    return parser.parse_args()
