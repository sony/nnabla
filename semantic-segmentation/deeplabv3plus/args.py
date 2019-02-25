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


def get_args(monitor_path='tmp.monitor', max_iter=10000, model_save_path=None, learning_rate=1e-3, batch_size=128, weight_decay=1e-4, description=None):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    import os
    if model_save_path is None:
        model_save_path = monitor_path
    if description is None:
        description = "Examples on data iterator examples. The following help shared among examples in this folder. Some arguments are valid or invalid in some examples."
    parser = argparse.ArgumentParser(description)
    parser.add_argument('--fine-tune', action='store_true',
                        default=False, help="Whether to fine tune model or not; False by default")
    parser.add_argument('--distributed', action='store_true',
                        default=False, help="Whether to use distributed/single gpu training; False by default")
    parser.add_argument('--compute-acc', action='store_true',
                        default=False, help="Whether to compute the accuracy mean IOU value during training/validation; False by default")
    parser.add_argument("--input-ckpt-file", type=str)
    parser.add_argument("--output-nnabla-file", type=str,
                        default='deeplab_nnabla.h5')
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size)
    parser.add_argument("--label-path", type=str)
    parser.add_argument("--data-dir", type=str, help='Path to VOC datset.')
    parser.add_argument("--train-file", type=str,
                        help='VOC train split text file')
    parser.add_argument("--val-file",
                        type=str, help='VOC val split text file')
    parser.add_argument("--train-dir", "-t",
                        type=str, default=model_save_path,
                        help='Path to training data.')
    parser.add_argument("--val-dir", "-v",
                        type=str, default=model_save_path,
                        help='Path to validation data.')
    parser.add_argument("--train-label-dir",
                        type=str, default=model_save_path,
                        help='Path to training data-labels.')
    parser.add_argument("--val-label-dir",
                        type=str, default=model_save_path,
                        help='Path to validation data-labels.')
    parser.add_argument("--learning-rate", "-l",
                        type=float, default=learning_rate)
    parser.add_argument("--output-stride",
                        type=int, default=16)
    parser.add_argument("--monitor-path", "-m",
                        type=str, default=monitor_path,
                        help='Path monitoring logs saved.')
    parser.add_argument("--max-iter", "-i", type=int, default=max_iter,
                        help='Max iteration of training.')
    parser.add_argument("--val-interval", type=int, default=100,
                        help='Validation interval.')
    parser.add_argument("--val-iter", "-j", type=int, default=10,
                        help='Each validation runs `val_iter mini-batch iteration.')
    parser.add_argument("--accum-grad",
                        type=int, default=32,
                        help='Weight decay factor of SGD update.')
    parser.add_argument("--weight-decay", "-w",
                        type=float, default=weight_decay,
                        help='Weight decay factor of SGD update.')
    parser.add_argument("--warmup-epoch", type=int, default=5)
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cuda.cudnn`.')
    parser.add_argument("--type-config", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--model-save-interval", "-s", type=int, default=1000,
                        help='The interval of saving model parameters.')
    parser.add_argument("--model-save-path", "-o",
                        type=str, default=model_save_path,
                        help='Path the model parameters saved.')
    parser.add_argument("--pretrained-model-path",
                        type=str, default=model_save_path,
                        help='Path the pretrained model parameters saved.')
    parser.add_argument("--net", "-n", type=str,
                        default='lenet',
                        help="Neural network architecure type (used only in classification*.py).\n  classification.py: ('lenet'|'resnet'),  classification_bnn.py: ('bincon'|'binnet'|'bwn'|'bwn'|'bincon_resnet'|'binnet_resnet'|'bwn_resnet')")
    parser.add_argument('--context', '-c', type=str,
                        default='cpu', help="Extension modules. ex) 'cpu', 'cudnn'.")
    parser.add_argument('--augment-train', action='store_true',
                        default=False, help="Enable data augmentation of training data.")
    parser.add_argument('--augment-test', action='store_true',
                        default=False, help="Enable data augmentation of testing data.")
    parser.add_argument('--channel', default=1, type=int)
    parser.add_argument('--image-width', default=28, type=int)
    parser.add_argument('--image-height', default=28, type=int)
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument("--model-load-path", "-T",
                        type=str, default=model_save_path,
                        help='Path the model parameters loaded.')
    parser.add_argument('--label-file-path', type=str)
    parser.add_argument('--test-image-file', type=str)
    parser.add_argument('--num-class', default=10, type=int)
    parser.add_argument('--train-samples', default=10, type=int)
    parser.add_argument('--val-samples', default=10, type=int)
    parser.add_argument("--sync-weight-every-itr",
                        type=int, default=100,
                        help="Sync weights every specified iteration. NCCL uses\
 the ring all reduce, so gradients in each device are not exactly same. When it\
 is accumulated in the weights, the weight values in each device diverge.")

    args = parser.parse_args()
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    return args
