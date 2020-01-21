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
    Get command line arguments to run FAN inference.
    Arguments set the default values of command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-size", type=int, default=4)
    parser.add_argument("--reference-scale", type=int, default=195)
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the Inference run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension modules('cpu', 'cudnn')")
    parser.add_argument("--model", "-m",
                        type=str, default="./2DFAN4_NNabla_model.h5",
                        help='Path to converted FAN-model weight file.')
    parser.add_argument("--cnn-face-detction-model",
                        type=str, default="./mmod_human_face_detector.dat",
                        help='Path to downloded cnn face-detection model.')
    parser.add_argument("--resnet-depth-model",
                        type=str, default="./Resnet_Depth_NNabla_model.h5",
                        help='Path to converted ResNetDepth weight file.')
    parser.add_argument('--test-image', type=str, default='./test-image.jpg',
                        help='Path to the image file.')
    parser.add_argument('--output', type=str, default='output.png',
                        help='Path to save the output image.')
    parser.add_argument("--landmarks-type-3D", help="To run 3D-FAN network. If it is True, you need to pass 3D-FAN pre-trained model path to --model", default=False,
                        action="store_true")
    args = parser.parse_args()
    return args
