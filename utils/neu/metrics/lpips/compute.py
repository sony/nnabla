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
import glob
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

from tqdm import tqdm
from .lpips import LPIPS
from nnabla.utils.image_utils import imread
from nnabla.ext_utils import get_extension_context


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, nargs=2,
                        help='path to images to measure LPIPS, \
                              alternative;y or the directory containing real/fake images, \
                              or the text file listing real/fake images, \
                              or .npz file containing statistics of real/fake images. \
                              Expects the former is the real, and the latter is the fake.')
    parser.add_argument('--params-dir',
                        default=os.path.dirname(os.path.abspath(__file__)), type=str,
                        help='path to the directory containing weight files (.h5).')
    parser.add_argument('--model', type=str, choices=["alex", "vgg"], default="alex",
                        help='network architecture to use as a feature extractor')
    parser.add_argument('--outfile', '-o', type=str,
                        help='path to the output file the scores is recorded.')
    parser.add_argument('--device-id', '-d', default=0, type=int,
                        help='Device ID.')
    parser.add_argument('--context', '-c', default="cudnn", type=str,
                        help='Backend name.')
    return parser.parse_args()


def compute_lpips_of_paired_images(lpips, img0_path, img1_path, params_dir, model):
    img0 = imread(img0_path, channel_first=True)
    # normalize. value range should be in [-1., +1.].
    img0 = (img0 / (255. / 2)) - 1
    img0 = F.reshape(nn.Variable.from_numpy_array(img0), (1,)+img0.shape)

    img1 = imread(img1_path, channel_first=True)
    # normalize. value range should be in [-1., +1.].
    img1 = (img1 / (255. / 2)) - 1
    img1 = F.reshape(nn.Variable.from_numpy_array(img1), (1,)+img1.shape)

    lpips_val = lpips(img0, img1, mean_batch=True)
    lpips_val.forward()

    return lpips_val


def process_with_image_lists(img0_paths, img1_paths, outfile, params_dir, model):

    lpips = LPIPS(model=model, params_dir=params_dir)

    if outfile:
        print(f"All the computed LPIPS scores are recorded to {outfile}.")
        with open(outfile, "w", encoding="utf-8") as fo:
            print("LPIPS", file=fo)
        pbar = tqdm(total=len(img0_paths))

    for img0_path, img1_path in zip(img0_paths, img1_paths):
        lpips_val = compute_lpips_of_paired_images(
            lpips, img0_path, img1_path, params_dir, model)
        if outfile:
            with open(outfile, 'a', encoding="utf-8") as fo:
                print(f"{lpips_val.d.sum():.3f}: {img0_path} - {img1_path}", file=fo)
            pbar.update(1)
        else:
            print(f"{lpips_val.d.sum():.3f}: {img0_path} - {img1_path}")


def handle_textfiles(path0, path1, outfile, params_dir, model):
    assert os.path.isfile(path0), f"{path0} is not found."
    assert os.path.isfile(path1), f"{path1} is not found."

    with open(path0, "r") as fi0:
        img0_paths = [_.rstrip("\n") for _ in f.readlines()]
    with open(path1, "r") as fi1:
        img1_paths = [_.rstrip("\n") for _ in f.readlines()]
    assert len(img0_paths) == len(
        img1_paths), "number of images does not match."

    process_with_image_lists(img0_paths, img1_paths,
                             outfile, params_dir, model)


def handle_directories(path0, path1, outfile, params_dir, model):
    assert os.path.isdir(path0), f"specified directory {path0} is not found."
    assert os.path.isdir(path1), f"specified directory {path1} is not found."

    img0_paths = sorted(glob.glob(f"{path0}/*"))
    img1_paths = sorted(glob.glob(f"{path1}/*"))
    assert len(img0_paths) == len(
        img1_paths), "number of images does not match."

    process_with_image_lists(img0_paths, img1_paths,
                             outfile, params_dir, model)


def main():
    """
        an example usage.
    """
    args = get_args()
    # Context
    ctx = get_extension_context(args.context, device_id=args.device_id)
    nn.set_default_context(ctx)
    params_dir = args.params_dir
    model = args.model
    outfile = args.outfile

    paths = args.path
    ext0, ext1 = [os.path.splitext(path)[-1] for path in paths]
    assert ext0 == ext1, "given inputs are not the same filetype."

    if ext0 == ".txt":
        # assume image lists are given
        handle_textfiles(paths[0], paths[1], outfile, params_dir, model)

    elif ext0 == "":
        # assume directoriess are given
        handle_directories(paths[0], paths[1], outfile, params_dir, model)

    elif ext0 in [".png", "jpg"]:
        assert os.path.isfile(paths[0]), f"specified file {paths[0]} is not found."
        assert os.path.isfile(paths[1]), f"specified file {paths[1]} is not found."
        lpips = LPIPS(model=model, params_dir=params_dir)
        lpips_val = compute_lpips_of_paired_images(
            lpips, paths[0], paths[1], params_dir, model)
        print(f"LPIPS: {lpips_val.d.sum():.3f}")

    else:
        raise RuntimeError(f"Invalid input file {ext0}.")


if __name__ == '__main__':
    main()
