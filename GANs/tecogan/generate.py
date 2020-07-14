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

import os
import argparse
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla.ext_utils import get_extension_context
from utils.utils import save_img, space_to_depth, upscale_four, warp_by_flow
from data_loader import inference_data_loader
from models import generator, flow_estimator

parser = argparse.ArgumentParser(description='TecoGAN')
parser.add_argument('--model', default='./TecoGAN_NNabla_model.h5',
                    help='path to converted NNabla weight file')
parser.add_argument('--input-dir-lr', default='./LR/calendar',
                    help='Path to input LR directory')
parser.add_argument('--output-dir', default='./results/calendar',
                    help='Path to save the output HR frames')
parser.add_argument('--output-name', default='output',
                    help='The pre name of the outputs')
parser.add_argument('--output-ext', default='png',
                    help='The format of the output')
parser.add_argument('--num_resblock', default=16, type=int,
                    help='No of residual blocks in generator. (16 for TecoGAN and 10 FRVSR)')
args = parser.parse_args()

ctx = get_extension_context('cudnn')
nn.set_default_context(ctx)


def main():
    """
        Inference function to generate SR images.
    """
    nn.load_parameters(args.model)
    # Inference data loader
    inference_data = inference_data_loader(args.input_dir_lr)
    input_shape = [1, ] + list(inference_data.inputs[0].shape)
    output_shape = [1, input_shape[1]*4, input_shape[2]*4, 3]
    oh = input_shape[1] - input_shape[1]//8 * 8
    ow = input_shape[2] - input_shape[2]//8 * 8

    # Build the computation graph
    inputs_raw = nn.Variable(input_shape)
    pre_inputs = nn.Variable(input_shape)
    pre_gen = nn.Variable(output_shape)
    pre_warp = nn.Variable(output_shape)

    transposed_pre_warp = space_to_depth(pre_warp)
    inputs_all = F.concatenate(inputs_raw, transposed_pre_warp)
    with nn.parameter_scope("generator"):
        gen_output = generator(inputs_all, 3, args.num_resblock)
    outputs = (gen_output + 1) / 2
    inputs_frames = F.concatenate(pre_inputs, inputs_raw)
    with nn.parameter_scope("fnet"):
        flow_lr = flow_estimator(inputs_frames)
    flow_lr = F.pad(flow_lr, (0, 0, 0, oh, 0, ow, 0, 0), "reflect")
    flow_hr = upscale_four(flow_lr*4.0)
    pre_gen_warp = warp_by_flow(pre_gen, flow_hr)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    max_iter = len(inference_data.inputs)
    print('Frame evaluation starts!!')
    pre_inputs.d, pre_gen.d, pre_warp.d = 0, 0, 0
    for i in range(max_iter):
        inputs_raw.d = np.array([inference_data.inputs[i]]).astype(np.float32)
        if i != 0:
            pre_gen_warp.forward()
            pre_warp.data.copy_from(pre_gen_warp.data)
        outputs.forward()
        output_frame = outputs.d

        if i >= 5:
            name, _ = os.path.splitext(
                os.path.basename(str(inference_data.paths_lr[i])))
            filename = args.output_name+'_'+name
            print('saving image %s' % filename)
            out_path = os.path.join(args.output_dir, "%s.%s" %
                                    (filename, args.output_ext))
            save_img(out_path, output_frame[0])
        else:  # First 5 is a hard-coded symmetric frame padding, ignored but time added!
            print("Warming up %d" % (5-i))

        pre_inputs.data.copy_from(inputs_raw.data)
        pre_gen.data.copy_from(outputs.data)


if __name__ == '__main__':
    main()
