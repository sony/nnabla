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
import glob
import numpy as np

from tqdm import tqdm
from utils import get_points_list, get_bod_map, get_bod_img
from data import CelebVNonRefDatahandler
from nnabla.utils.image_utils import imread, imsave, imresize


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='CelebV',
                        choices=['CelebV', "WFLW"],
                        help='path to images root directory')
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', "test"],
                        help='test or train (valid only for WFLW)')
    parser.add_argument('--src-dir',
                        type=str,
                        default='./datasets',
                        help='path to images root directory')
    parser.add_argument('--celeb-name-list',
                        nargs="*",
                        default=None,
                        help='names of the celebs (valid only for CelebV)')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./datasets/celebv_heatmaps_GT/',
                        help='path to images root directory')
    parser.add_argument('--resize-size',
                        type=tuple,
                        default=(64, 64),
                        help='size of the boundary images after resized')
    parser.add_argument('--line-thickness',
                        type=int,
                        default=3,
                        help='thickness of the line between points')
    parser.add_argument('--gaussian-kernel',
                        type=tuple,
                        default=(5, 5),
                        help='size of the gaussian kernel')
    parser.add_argument('--gaussian-sigma',
                        type=int,
                        default=3,
                        help='sigma used for gaussian kernel')
    parser.add_argument('--save-boundary-image',
                        action='store_true',
                        default=False,
                        help='if specified, save boundary image for visualization.')
    args = parser.parse_args()

    return args


def get_square_corners(y1, x1, y2, x2):
    h = y2 - y1
    w = x2 - x1

    if h == w:
        return y1, x1, y2, x2

    diff = np.abs(h - w)
    if diff % 2:
        diff += 1

    if h > w:
        x1 = int(x1 - (diff / 2))
        x2 = int(x2 + (diff / 2))
    else:
        y1 = int(y1 - (diff / 2))
        y2 = int(y2 + (diff / 2))

    return y1, x1, y2, x2


def get_croped_image(annotation, data_dir, margin=np.random.uniform(0, 0.15)):
    NUM_POINTS = 98
    img_name = annotation[-1].rsplit(os.linesep)[0]
    landmarks = [float(_) for _ in annotation[:NUM_POINTS*2]]
    y1, x1, y2, x2 = [int(_) for _ in annotation[NUM_POINTS*2:-7]]
    y_list = [int(float(_)) for _ in landmarks[0::2]]
    x_list = [int(float(_)) for _ in landmarks[1::2]]
    y_center = y_list[54]
    x_center = x_list[54]

    y_diff = max(y2 - y_center, y_center - y1)
    x_diff = max(x2 - x_center, x_center - x1)

    y1 = y_center - int((1 + margin)*y_diff)
    x1 = x_center - int((1 + margin)*x_diff)
    y2 = y_center + int((1 + margin)*y_diff)
    x2 = x_center + int((1 + margin)*x_diff)

    y1, x1, y2, x2 = get_square_corners(y1, x1, y2, x2)

    img = imread(os.path.join(data_dir, img_name), channel_first=True)
    H, W = img.shape[1:]

    # just in case that the corner lies outside the image, apply padding.
    if x1 < 0:
        img = np.concatenate([img[:, ::-1, :], img], axis=1)
        x1 += H
        x2 += H
        x_list = [_ + H for _ in x_list]
        H += H

    if y1 < 0:
        img = np.concatenate([img[:, :, ::-1], img], axis=2)
        y1 += W
        y2 += W
        y_list = [_ + W for _ in y_list]
        W += W

    if x2 > H:
        img = np.concatenate([img, img[:, ::-1, :]], axis=1)

    if y2 > W:
        img = np.concatenate([img, img[:, :, ::-1]], axis=2)

    img = img[:, x1:x2, y1:y2]
    y_list = [_ - y1 for _ in y_list]
    x_list = [_ - x1 for _ in x_list]
    return img_name, img, y_list, x_list


class Preprocesser(CelebVNonRefDatahandler):
    """docstring for Preprocesser"""

    def __init__(self, imgs_root_path, resize_size, line_thickness, gaussian_kernel, gaussian_sigma):
        super(Preprocesser, self).__init__(resize_size,
                                           line_thickness, gaussian_kernel, gaussian_sigma)
        self.imgs_root_path = imgs_root_path
        self.resize_size = resize_size
        self.line_thickness = line_thickness
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma

    def get_img_path(self, imgs_root_path, img_name):
        img_path = os.path.join(imgs_root_path, 'Image', img_name)
        return img_path

    def get_img_name(self, ant):
        name = ant.split(' ')[-1].split('\n')[0]
        return name

    def get_img(self, img_path):
        img = imread(img_path, num_channels=3, channel_first=True)
        return img  # (3, 256, 256)

    def get_ant_and_size(self, imgs_root_path, img_dirname="Image", txt_dirname="", test=False):
        # override
        print('loading: {}'.format(imgs_root_path))
        # get the annotation txt file path
        txt_path = sorted(glob.glob(os.path.join(
            imgs_root_path, txt_dirname, '*.txt')))
        if test:
            txt_path = txt_path[0]
        else:
            txt_path = txt_path[-1]

        with open(txt_path, "r", encoding="utf-8") as f:
            ant = f.readlines()  # read the annotation data from the txt
        size = len(ant)  # the number of training images
        print(f'the number of training images: {size}')
        return ant, size

    def __call__(self, ant, stop_idx=-1):
        # load image
        img_name = self.get_img_name(ant)
        img_path = self.get_img_path(self.imgs_root_path, img_name)
        img = self.get_img(img_path)

        # len(x_list)=98, len(y_list)=98
        x_list, y_list = get_points_list(ant, stop_idx)

        bod_map = get_bod_map(img, x_list, y_list,
                              resize_size=self.resize_size,
                              line_thickness=self.line_thickness,
                              gaussian_kernel=self.gaussian_kernel,
                              gaussian_sigma=self.gaussian_sigma)

        bod_map_resized = get_bod_map(img, x_list, y_list,
                                      resize_size=(256, 256),
                                      line_thickness=self.line_thickness,
                                      gaussian_kernel=self.gaussian_kernel,
                                      gaussian_sigma=self.gaussian_sigma)

        # For Visualization. by default  (3, 64, 64)
        bod_img = get_bod_img(img, x_list, y_list,
                              resize_size=self.resize_size,
                              line_thickness=self.line_thickness,
                              gaussian_kernel=self.gaussian_kernel,
                              gaussian_sigma=self.gaussian_sigma)
        return img, bod_map, bod_img, img_name, bod_map_resized


def denormalize_img(img):
    # [-1., +1.] -> [0, 255]
    img = (img + 1.) / 2.0
    return img


def preprocess_WFLW(args):
    import csv
    print("preprocessing WFLW dataset...")

    src_dir = args.src_dir
    assert os.path.isdir(src_dir)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    resize_size = args.resize_size
    line_thickness = args.line_thickness
    gaussian_kernel = args.gaussian_kernel
    gaussian_sigma = args.gaussian_sigma

    imgs_root_path = src_dir
    assert os.path.exists(imgs_root_path), f"specified path {imgs_root_path} not found."

    out_csv = [["saved_name", "real_name"]]

    mode = args.mode
    textname = f"WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_{mode}.txt"
    with open(os.path.join(src_dir, textname)) as f:
        annotations = f.readlines()
        annotations = [_.split(" ") for _ in annotations]

    prep = Preprocesser(imgs_root_path, resize_size,
                        line_thickness, gaussian_kernel, gaussian_sigma)

    tmp_hm_dict = dict()
    tmp_img_dict = dict()

    if args.save_boundary_image:
        os.makedirs(os.path.join(
            out_dir, "WFLW_landmark_images", mode), exist_ok=True)
        os.makedirs(os.path.join(
            out_dir, "WFLW_cropped_images", mode), exist_ok=True)

    idx = 0
    for annotation in tqdm(annotations):
        img_name, img, y_list, x_list = get_croped_image(
            annotation, os.path.join(src_dir, "WFLW_images"))
        scale_ratio = 256. / img.shape[-1]
        x_list_scaled = [int(_ * scale_ratio) for _ in x_list]
        y_list_scaled = [int(_ * scale_ratio) for _ in y_list]
        img_resized = imresize(img, (256, 256), channel_first=True)
        bod_img = get_bod_img(img_resized, y_list_scaled, x_list_scaled,
                              resize_size, line_thickness, gaussian_kernel, gaussian_sigma)
        bod_map = get_bod_map(img_resized, y_list_scaled, x_list_scaled,
                              resize_size, line_thickness, gaussian_kernel, gaussian_sigma)
        saved_name = f"{mode}_{idx}.png"
        tmp_img_dict[saved_name] = img_resized
        tmp_hm_dict[saved_name] = bod_map  # uint8
        out_csv.append([saved_name, img_name])

        if args.save_boundary_image:
            save_path_bod = os.path.join(
                out_dir, "WFLW_landmark_images", mode, saved_name)
            save_path_cropped = os.path.join(
                out_dir, "WFLW_cropped_images", mode, saved_name)
            imsave(save_path_bod, bod_img, channel_first=True)
            imsave(save_path_cropped, img_resized, channel_first=True)
        idx += 1

    np.savez_compressed(os.path.join(out_dir, f'WFLW_cropped_image_{mode}'), **tmp_img_dict)
    np.savez_compressed(os.path.join(out_dir, f'WFLW_heatmap_{mode}'), **tmp_hm_dict)
    with open(os.path.join(out_dir, f"{mode}_data.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(out_csv)


def preprocess_celebV(args):
    """
        save .npz files containing images(as uint8), boundary images(as uint8),
        and boundary heatmaps (as float).
    """
    print("preprocessing celebV dataset...")

    src_dir = args.src_dir
    assert os.path.isdir(src_dir)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    resize_size = args.resize_size
    line_thickness = args.line_thickness
    gaussian_kernel = args.gaussian_kernel
    gaussian_sigma = args.gaussian_sigma

    if args.celeb_name_list is None:
        celeb_name_list = ['Donald_Trump', 'Emmanuel_Macron',
                           'Jack_Ma', 'Kathleen', 'Theresa_May']
    else:
        celeb_name_list = args.celeb_name_list

    for celeb_name in celeb_name_list:
        imgs_root_path = os.path.join(src_dir, "CelebV", celeb_name)
        if not os.path.exists(imgs_root_path):
            raise ValueError(f"specified path {imgs_root_path} not found.")

        prep = Preprocesser(imgs_root_path, resize_size,
                            line_thickness, gaussian_kernel, gaussian_sigma)
        annotations, _ = prep.get_ant_and_size(imgs_root_path)

        tmp_image_dict = dict()
        tmp_hm_dict = dict()
        tmp_hm_resized_dict = dict()

        if args.save_boundary_image:
            os.makedirs(os.path.join(
                out_dir, celeb_name, "Image"), exist_ok=True)

        for annotation in tqdm(annotations):
            img, bod_map, bod_img, img_name, bod_map_resized = prep(annotation)
            tmp_image_dict[img_name] = img  # uint8
            tmp_hm_dict[img_name] = bod_map  # uint8
            tmp_hm_resized_dict[img_name] = bod_map_resized  # uint8

            if args.save_boundary_image:
                save_path = os.path.join(
                    out_dir, celeb_name, "Image", img_name)
                imsave(save_path, bod_img, channel_first=True)

        np.savez_compressed(os.path.join(
            out_dir, celeb_name + '_image'), **tmp_image_dict)
        np.savez_compressed(os.path.join(
            out_dir, celeb_name + '_heatmap'), **tmp_hm_dict)
        np.savez_compressed(os.path.join(
            out_dir, celeb_name + '_resized_heatmap'), **tmp_hm_resized_dict)

        del tmp_image_dict
        del tmp_hm_dict
        del tmp_hm_resized_dict


def main():

    args = get_args()
    if args.dataset == "CelebV":
        preprocess_celebV(args)
    else:
        preprocess_WFLW(args)


if __name__ == '__main__':
    main()
