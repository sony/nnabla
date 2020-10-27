# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

from ops import upsample_2d, lerp, convert_images_to_uint8
from networks import mapping_network, conv_block
from nnabla.ext_utils import get_extension_context
from nnabla.utils.image_utils import imsave


def synthesis(w, constant_bc, noise_seed, mix_after):
    """
        given latent vector w, constant input and noise seed,
        synthesis the image.
    """
    # resolution 4 x 4
    rnd = np.random.RandomState(noise_seed)
    noise = nn.Variable.from_numpy_array(rnd.randn(1, 1, 4, 4))
    h = conv_block(constant_bc, w[0], noise, res=4,
                   outmaps=512, namescope="Conv")
    torgb = conv_block(h, w[1], noise=None, res=4, outmaps=3, inmaps=512,
                       kernel_size=1, pad_size=0, demodulate=False, namescope="ToRGB", act=F.identity)

    # initial feature maps
    outmaps = 512
    inmaps = 512

    # resolution 8 x 8 - 1024 x 1024
    for i in range(1, 9):
        
        w_1 = w[0] if (2+i)*2-5 <= mix_after else w[1] 
        w_2 = w[0] if (2+i)*2-4 <= mix_after else w[1] 
        w_3 = w[0] if (2+i)*2-3 <= mix_after else w[1] 
        
        if i > 4:
            outmaps = outmaps // 2
        curr_shape = (1, 1, 2 ** (i + 2), 2 ** (i + 2))
        noise = nn.Variable.from_numpy_array(rnd.randn(*curr_shape))
        h = conv_block(h, w_1, noise, res=2 ** (i + 2), outmaps=outmaps, inmaps=inmaps,
                       kernel_size=3, up=True, namescope="Conv0_up")

        if i > 4:
            inmaps = inmaps // 2
        noise = nn.Variable.from_numpy_array(rnd.randn(*curr_shape))
        h = conv_block(h, w_2, noise, res=2 ** (i + 2), outmaps=outmaps, inmaps=inmaps,
                       kernel_size=3, pad_size=1, namescope="Conv1")

        # toRGB blocks
        prev_torgb = upsample_2d(torgb, k=[1, 3, 3, 1])
        curr_torgb = conv_block(h, w_3, noise=None, res=2 ** (i + 2), outmaps=3, inmaps=inmaps,
                                kernel_size=1, pad_size=0, demodulate=False, namescope="ToRGB", act=F.identity)

        torgb = curr_torgb + prev_torgb

    return torgb


def generate(batch_size, style_noises, noise_seed, mix_after, truncation_psi=0.5):
    """
        given style noises, noise seed and truncation value, generate an image.
    """
    # normalize noise inputs
    style_noises_normalized = []
    for style_noise in style_noises:
        noise_std = (F.mean(style_noise ** 2., axis=1,
                            keepdims=True)+1e-8) ** 0.5
        style_noise_normalized = F.div2(style_noise, noise_std)
        style_noises_normalized.append(style_noise_normalized)

    # get latent code
    w = [mapping_network(_, outmaps=512) for _ in style_noises_normalized]

    # truncation trick
    dlatent_avg = nn.parameter.get_parameter_or_create(
        name="dlatent_avg", shape=(1, 512))
    w = [lerp(dlatent_avg, _, truncation_psi) for _ in w]

    constant = nn.parameter.get_parameter_or_create(
                    name="G_synthesis/4x4/Const/const",
                    shape=(1, 512, 4, 4))
    constant_bc = F.broadcast(constant, (batch_size,) + constant.shape[1:])
    rgb_output = synthesis(w, constant_bc, noise_seed, mix_after)
    return rgb_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-filename', '-o', type=str, default=None,
                        help="name of an output image file.")
    parser.add_argument('--output-dir', '-d', type=str, default="results",
                        help="directory where the generated image is saved.")

    parser.add_argument('--seed', type=int, required=True,
                        help="seed for primal style noise.")
    parser.add_argument('--stochastic-seed', type=int, default=1,
                        help="seed for noises added to intermediate features.")

    parser.add_argument('--truncation-psi', default=0.5, type=float,
                        help="value for truncation trick.")
    
    parser.add_argument('--batch-size', type=int, default=1,
                        help="Number of images to generate.")

    parser.add_argument('--mixing', action='store_true',
                        help="if specified, apply style mixing with additional seed.")
    parser.add_argument('--seed-mix', type=int, default=None,
                        help="seed for another / secondary style noise.")
    parser.add_argument('--mix-after', type=int, default=7,
                        help="after this layer, style mixing is applied.")

    parser.add_argument('--context', '-c', type=str, default="cudnn",
                        help="context. cudnn is recommended.")

    args = parser.parse_args()

    assert 0 < args.mix_after < 17, "specify --mix-after from 1 to 16."

    if not os.path.isfile("styleGAN2_G_params.h5"):
        print("Downloading the pretrained weight. Please wait...")
        url = "https://nnabla.org/pretrained-models/nnabla-examples/GANs/stylegan2/styleGAN2_G_params.h5"
        from nnabla.utils.data_source_loader import download
        download(url, url.split('/')[-1], False)

    ctx = get_extension_context(args.context)
    nn.set_default_context(ctx)

    batch_size = args.batch_size
    num_layers = 18

    rnd = np.random.RandomState(args.seed)
    z = rnd.randn(batch_size, 512)

    print("Generation started...")
    print(f"truncation value: {args.truncation_psi}")
    print(f"seed for additional noise: {args.stochastic_seed}")
    
    # Inference via nn.NdArray utilizes significantly less memory
    
    if args.mixing:
        # apply style mixing
        assert args.seed_mix
        print(f"using style noise seed {args.seed} for layers 0-{args.mix_after - 1}")
        print(f"using style noise seed {args.seed_mix} for layers {args.mix_after}-{num_layers}.")
        rnd = np.random.RandomState(args.seed_mix)
        z2 = rnd.randn(batch_size, 512)
        style_noises = [nn.NdArray.from_numpy_array(z)]
        style_noises += [nn.NdArray.from_numpy_array(z2)]
    else:
        # no style mixing (single noise / style is used)
        print(f"using style noise seed {args.seed} for entire layers.")
        style_noises = [nn.NdArray.from_numpy_array(z) for _ in range(2)]

    nn.set_auto_forward(True)
    nn.load_parameters("styleGAN2_G_params.h5")
    rgb_output = generate(batch_size, style_noises,
                          args.stochastic_seed, args.mix_after, args.truncation_psi)

    # convert to uint8 to save an image file
    image = convert_images_to_uint8(rgb_output, drange=[-1, 1])
    if args.output_filename is None:
        if not args.mixing:
            filename = f"seed{args.seed}"
        else:
            filename = f"seed{args.seed}_{args.seed_mix}"
    else:
        filename = args.output_filename

    os.makedirs(args.output_dir, exist_ok=True)
    
    for i in range(batch_size):
        filepath = os.path.join(args.output_dir, f'{filename}_{i}.png')
        imsave(filepath, image[i], channel_first=True)
        print(f"Genetation completed. Saved {filepath}.")


if __name__ == '__main__':
    main()