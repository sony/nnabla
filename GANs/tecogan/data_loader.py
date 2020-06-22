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
import collections
import numpy as np
import cv2 as cv
from PIL import Image
from numpy import asarray
from scipy import signal
import nnabla as nn
import nnabla.functions as F
from nnabla.utils.data_iterator import data_iterator_simple


def inference_data_loader(filedir):
    """
    Read and prepare the inference data from specified directory
    filedir: inference data directory name
    return: list of inference image names and images to be enhanced
    """

    image_list_lr_temp = os.listdir(filedir)
    image_list_lr_temp = [_ for _ in image_list_lr_temp if _.endswith(".png")]
    # first sort according to abc, then sort according to 123
    image_list_lr_temp = sorted(image_list_lr_temp)
    image_list_lr_temp.sort(key=lambda f: int(
        ''.join(list(filter(str.isdigit, f))) or -1))
    image_list_lr = [os.path.join(filedir, _) for _ in image_list_lr_temp]

    # Read in and preprocess the images
    def preprocess_test(name):
        ip_img = cv.imread(name, 3).astype(np.float32)[:, :, ::-1]
        max_divided_img = ip_img / 255.0  # equivalent to np.max(ip_img)
        return max_divided_img

    image_lr = [preprocess_test(_) for _ in image_list_lr]

    # a hard-coded symmetric padding
    image_list_lr = image_list_lr[5:0:-1] + image_list_lr
    image_lr = image_lr[5:0:-1] + image_lr

    Data = collections.namedtuple('Data', 'paths_lr, inputs')
    return Data(
        paths_lr=image_list_lr,
        inputs=image_lr
    )


def gaussian_2dkernel(size=5, sig=1.0):
    """
    Returns a 2D Gaussian kernel array with side length size and a sigma of sig
    """
    g_kern1d = signal.gaussian(size, std=sig).reshape(size, 1)
    g_kern2d = np.outer(g_kern1d, g_kern1d)
    return g_kern2d/g_kern2d.sum()


def nn_data_gauss_down_quad(hr_data, sigma=1.5):
    """
    2D down-scaling by 4 with Gaussian blur
    sigma: the sigma used for Gaussian blur
    return: down-scaled data
    """

    k_w = 1 + 2 * int(sigma * 3.0)
    gau_k = gaussian_2dkernel(k_w, sigma)
    gau_0 = np.zeros_like(gau_k)
    gau_wei = np.float32([
        [gau_k, gau_0, gau_0],
        [gau_0, gau_k, gau_0],
        [gau_0, gau_0, gau_k]])  # only works for RGB images!
    gau_wei = np.transpose(gau_wei, [0, 2, 3, 1])
    gau_wei = nn.Variable.from_numpy_array(gau_wei)
    down_sampled_data = F.convolution(
        hr_data, weight=gau_wei, stride=(4, 4), channel_last=True)
    return down_sampled_data


def get_sample_name_grid(conf):
    """
    Return a conf.train.rnn_n number of image names list
    for ex., for 120 frame video, return a list of 10, each with 110 * sequences file names
    conf: configuration information
    return: grid of file names for TecoGAN training
    """

    # Check the input directory
    if conf.data.input_video_dir == '':
        raise ValueError(
            'Video input directory input_video_dir is not provided')

    if not os.path.exists(conf.data.input_video_dir):
        raise ValueError('Video input directory not found')

    image_grid_hr_r = [[] for _ in range(conf.train.rnn_n)]  # all empty lists

    for dir_i in range(conf.data.str_dir, conf.data.end_dir+1):
        ip_dir = os.path.join(conf.data.input_video_dir,
                              '%s_%04d' % (conf.data.input_video_pre, dir_i))
        if os.path.exists(ip_dir):  # the following names are hard coded
            if not os.path.exists(os.path.join(ip_dir, 'col_high_%04d.png' % conf.data.max_frm)):
                print("Skip %s, since foler doesn't contain enough frames!" % ip_dir)
                continue
            for f_i in range(conf.train.rnn_n):
                image_grid_hr_r[f_i] += [os.path.join(ip_dir, 'col_high_%04d.png' % frame_i)
                                         for frame_i in
                                         range(f_i, conf.data.max_frm - conf.train.rnn_n + f_i + 1)]

    return image_grid_hr_r


# moving decision
def moving_decision(conf):
    """
    Decide based on random sampling whether to randomly move first frame for data augmentation
    conf: configuration information
    return: whether to randomly move first frame, also movement offset parameters
    """

    offset_xy = np.floor(
        np.random.uniform(-3.5, 4.5, [conf.train.rnn_n, 2])).astype(int)

    # [FLAGS.RNN_N , 2], shifts
    pos_xy_tmp = np.cumsum(offset_xy, axis=0)
    pos_xy = np.zeros([conf.train.rnn_n, 2])
    pos_xy[1:, :] = pos_xy_tmp[:-1, :]

    min_pos = np.min(pos_xy, axis=0)
    range_pos = np.max(pos_xy, axis=0) - min_pos  # [ shrink x, shrink y ]
    lefttop_pos = pos_xy - min_pos  # crop point
    is_move = np.random.uniform(0, 1, []).astype(float)

    return (lefttop_pos).astype(int), (range_pos).astype(int), is_move


def preprocess(image):
    """
    modify image range from [0,1] to [-1, 1]
    return: range modified image
    """
    return image * 2 - 1


def data_iterator_sr(conf, num_samples, sample_names, tar_size, shuffle, rng=None):
    """
    Data iterator for TecoGAN training
    return: makes provision for low res & high res frames in RNN segments for specified batch_size
    """

    def populate_hr_data(i):

        hr_data = []  # high res rgb, in range 0-1, shape any

        # moving first frame -> data augmentation
        # our data augmentation, moving first frame to mimic camera motion
        if conf.train.movingFirstFrame:
            lefttop_pos, range_pos, is_move = moving_decision(conf)

        for f_i in range(conf.train.rnn_n):
            img_name = sample_names[f_i][i]
            image = Image.open(img_name)
            img_data = asarray(image).astype(float)
            img_data = img_data/255

            if conf.train.movingFirstFrame:
                if f_i == 0:
                    img_data_0 = img_data
                    target_size = img_data.shape

                img_data_1 = img_data_0[lefttop_pos[f_i][1]:target_size[0]
                                        - range_pos[1] + lefttop_pos[f_i][1],
                                        lefttop_pos[f_i][0]:target_size[1]
                                        - range_pos[0] + lefttop_pos[f_i][0], :]

                # random data augmentation -> move first frame only with 70% probability
                img_data = img_data if is_move < 0.7 else img_data_1

            hr_data.append(img_data)

        return hr_data

    def dataset_load_func(i):

        hr_data = populate_hr_data(i)

        # random crop each batch entry separately
        # Check whether perform crop
        if conf.train.random_crop is True:
            cur_size = hr_data[0].shape
            offset_h = np.floor(np.random.uniform(
                0, cur_size[0] - tar_size, [])).astype(int)
            offset_w = np.floor(np.random.uniform(
                0, cur_size[1] - tar_size, [])).astype(int)
            for frame_t in range(conf.train.rnn_n):
                hr_data[frame_t] = hr_data[frame_t][offset_h:offset_h +
                                                    tar_size, offset_w:offset_w + tar_size, :]

        # random flip:
        if conf.train.flip is True:
            # Produce the decision of random flip
            flip_decision = np.random.uniform(0, 1, []).astype(float)
            for frame_t in range(conf.train.rnn_n):
                if flip_decision < 0.5:
                    np.fliplr(hr_data[frame_t])

        hr_frames = hr_data
        target_frames = []

        k_w_border = int(1.5 * 3.0)
        for rnn_inst in range(conf.train.rnn_n):
            # crop out desired data
            cropped_data = hr_data[rnn_inst][k_w_border:k_w_border+conf.train.crop_size*4,
                                             k_w_border:k_w_border+conf.train.crop_size*4, :]
            pre_processed_data = preprocess(cropped_data)
            target_frames.append(pre_processed_data)

        return hr_frames, target_frames

    return data_iterator_simple(dataset_load_func, num_samples, conf.train.batch_size,
                                shuffle=shuffle, rng=rng, with_file_cache=False,
                                with_memory_cache=False)
