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


import os
import argparse
import nnabla as nn
from nnabla.ext_utils import get_extension_context
import numpy as np
import json
import glob
import model
from nnabla.utils.image_utils import imread, imsave, imresize
import functools


def saveimage(path, img):
    img = (img * 0.5) + 0.5
    imsave(path, img, channel_first=True)


def save_results(i, args, used_config, img_trg, lbl_trg):
    target_attr_flags = lbl_trg.d[0].reshape(lbl_trg.d[0].size)
    target_domain = "_".join([attr for idx, attr in zip(
        target_attr_flags, used_config["selected_attrs"]) if bool(idx) is True])
    result_x = img_trg.d[0]
    filename = os.path.join(args.result_save_path,
                            "generated_{}_{}.png".format(i, target_domain))
    saveimage(filename, result_x)
    print("Saved {}.".format(filename))
    return


def img_preprocess(img_paths, used_config):

    image_size = used_config["image_size"]
    images = list()
    image_names = list()

    for img_path in img_paths:
        # Load (and resize) image and labels.
        image = imread(img_path, num_channels=3, channel_first=True)
        if image.dtype == np.uint8:
            # Clip image's value from [0, 255] -> [0.0, 1.0]
            image = image / 255.0
        image = (image - 0.5) / 0.5  # Normalize
        image = imresize(image, (image_size, image_size),
                         interpolate='bilinear', channel_first=True)
        images.append(image)
        image_names.append(img_path.split("/")[-1])

    return np.asarray(images), np.asarray(image_names)


def get_user_input(used_config):
    label = [0 for _ in range(used_config["c_dim"])]
    choice = used_config["selected_attrs"]
    for i, c in enumerate(choice):
        print("Use '{}'?".format(c))
        while 1:
            ans = input("type yes or no: ")
            if ans in ["yes", "no"]:
                label[i] = 1 if ans == "yes" else 0
                break
            else:
                print("type 'yes' or 'no'.")
        #label[i] = int(bool(input("if yes, type 1, if not, just press enter:")))
    return np.array(label)


def generate(args):

    # Load the config data used for training.
    with open(args.config, "r") as f:
        used_config = json.load(f)

    paramfile = args.pretrained_params
    img_paths = glob.glob(os.path.join(args.test_image_path, "*.png"))
    assert os.path.isfile(paramfile) and paramfile.split(
        "/")[-1] == used_config["pretrained_params"], "Corresponding parameter file not found."

    print("Learned attributes choice: {}".format(
        used_config["selected_attrs"]))

    # Prepare Generator and Discriminator based on user config.
    generator = functools.partial(
        model.generator, conv_dim=used_config["g_conv_dim"], c_dim=used_config["c_dim"], repeat_num=used_config["g_repeat_num"])

    x_real = nn.Variable(
        [1, 3, used_config["image_size"], used_config["image_size"]])
    label_trg = nn.Variable([1, used_config["c_dim"], 1, 1])
    with nn.parameter_scope("gen"):
        x_fake = generator(x_real, label_trg)
    x_fake.persistent = True

    nn.load_parameters(paramfile)  # load learned parameters.

    images, image_names = img_preprocess(img_paths, used_config)

    for i, (image, image_name) in enumerate(zip(images, image_names)):
        # Get real images.
        print("Source image: {}".format(image_name))
        x_real.d = image

        # Generate target domain based on user input.
        label_trg.d = np.reshape(get_user_input(used_config), label_trg.shape)

        # Execute image translation.
        x_fake.forward(clear_no_need_grad=True)

        save_results(i, args, used_config, x_fake, label_trg)


def get_args():

    parser = argparse.ArgumentParser()

    # Generation
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension path. ex) cpu, cudnn.")
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument('--test-image-path', type=str,
                        help='a directory containing images used for image translation')
    parser.add_argument('--result-save-path', type=str,
                        default="tmp.results", help='a directory to save generated images')
    parser.add_argument('--pretrained-params', type=str, required=True,
                        help='path to the parameters used for generation.')
    parser.add_argument('--config', type=str, required=True,
                        help='path to the config file used for generation.')

    args = parser.parse_args()
    if not os.path.isdir(args.result_save_path):
        os.makedirs(args.result_save_path)

    return args


def main():
    args = get_args()
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)
    generate(args)


if __name__ == '__main__':
    main()
