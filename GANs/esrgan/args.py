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
    Get command line arguments.
    Arguments set the default values of command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description='ESRGAN')
    parser.add_argument('--gt_train', default='/Div2k_subset/mod/HR/x4',
                        help='train ground truth (HQ) image path')
    parser.add_argument('--lq_train', default='/Div2k_subset/mod/LR/x4',
                        help='train low quality (LQ) image path')
    parser.add_argument('--gt_val', default='/Set14_LR/HR/x4',
                        help='val ground truth (HQ) image path')
    parser.add_argument('--lq_val', default='/Set14_LR/LR/x4',
                        help='val generated images')
    parser.add_argument('--save_results', default='/ESRGANS/Nnabla/Set14_LR',
                        help='path for saving validation results.')
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--batch_size_train', type=int, default=16,
                        help='batch_size_train')
    parser.add_argument('--batch_size_val', type=int, default=1,
                        help='batch_size_val')
    parser.add_argument('--scale', type=int, default=4,
                        help='batch_size')
    parser.add_argument('--n_epochs', type=int, default=497,
                        help='no. of epochs 497 for psnr oriented rrdb and 199')
    parser.add_argument('--gt_size', type=int, default=128,
                        help='GT size')
    parser.add_argument('--savemodel', default='./output_esrgan',
                        help='path to save the trained weights')
    parser.add_argument('-d', '--device_id', type=int,
                        help='GPU IDs to be used', default=0)
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension modules. ex) 'cpu', 'cudnn'.")
    parser.add_argument('--eta_pixel_loss', type=float, default=1e-2,
                        help='pixel loss co-efficient')
    parser.add_argument('--feature_loss_weight', type=float, default=1,
                        help='feature loss co-efficient')
    parser.add_argument('--lambda_gan_loss', type=float, default=5e-3,
                        help='gan loss co-efficient')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='factor for lr reducyion in multistep learning')
    parser.add_argument('--eta_max', type=float, default=2e-4,
                        help='max learning rate for psnr oriented training')
    parser.add_argument('--eta_min', type=float, default=1e-7,
                        help='initial learning rate for psnr oriented training')
    parser.add_argument('--distributed', type=bool,
                        default=False, help="Distributed or single gpu training; False by default")
    parser.add_argument('--vgg_pre_trained_weights', type=str,
                        default='./vgg19.h5',
                        help="Path to VGG19 weights")
    parser.add_argument('--psnr_rrdb_pretrained', type=str,
                        default='/1000000.h5',
                        help="path to psnr rrdb pretrained file")
    args = parser.parse_args()
    return args
