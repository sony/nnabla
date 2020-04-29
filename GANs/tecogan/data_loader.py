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


import nnabla as nn
import nnabla.functions as F
from nnabla.utils.data_iterator import data_iterator_simple
import numpy as np
import collections
import os
import cv2 as cv
from PIL import Image
from numpy import asarray
from scipy import signal


def inference_data_loader(filedir):
    image_list_LR_temp = os.listdir(filedir)
    image_list_LR_temp = [_ for _ in image_list_LR_temp if _.endswith(".png")]
    # first sort according to abc, then sort according to 123
    image_list_LR_temp = sorted(image_list_LR_temp)
    image_list_LR_temp.sort(key=lambda f: int(
        ''.join(list(filter(str.isdigit, f))) or -1))
    image_list_LR = [os.path.join(filedir, _) for _ in image_list_LR_temp]

    # Read in and preprocess the images
    def preprocess_test(name):
        im = cv.imread(name, 3).astype(np.float32)[:, :, ::-1]
        im = im / 255.0  # np.max(im)
        return im
    image_LR = [preprocess_test(_) for _ in image_list_LR]

    if True:  # a hard-coded symmetric padding
        image_list_LR = image_list_LR[5:0:-1] + image_list_LR
        image_LR = image_LR[5:0:-1] + image_LR

    Data = collections.namedtuple('Data', 'paths_LR, inputs')
    return Data(
        paths_LR=image_list_LR,
        inputs=image_LR
    )


def gaussian_2dkernel(size=5, sig=1.):
    """
    Returns a 2D Gaussian kernel array with side length size and a sigma of sig
    """
    gkern1d = signal.gaussian(size, std=sig).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d/gkern2d.sum()


def nn_data_gaussDownby4(HRdata, sigma=1.5):
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
    y = F.convolution(HRdata, gau_wei, stride=(4, 4), channel_last=True)
    return y


# return a args.RNN_N number of image names list
# for example -> return a list of 10, each with 110*sequences file names
def getSampleNameGrid(conf):
    # Check the input directory
    if (conf.data.input_video_dir == ''):
        raise ValueError(
            'Video input directory input_video_dir is not provided')

    if (not os.path.exists(conf.data.input_video_dir)):
        raise ValueError('Video input directory not found')

    image_grid_HR_r = [[] for _ in range(conf.train.rnn_n)]  # all empty lists

    for dir_i in range(conf.data.str_dir, conf.data.end_dir+1):
        inputDir = os.path.join(conf.data.input_video_dir,
                                '%s_%04d' % (conf.data.input_video_pre, dir_i))
        if (os.path.exists(inputDir)):  # the following names are hard coded
            if not os.path.exists(os.path.join(inputDir, 'col_high_%04d.png' % conf.data.max_frm)):
                print("Skip %s, since foler doesn't contain enough frames!" % inputDir)
                continue
            for fi in range(conf.train.rnn_n):
                image_grid_HR_r[fi] += [os.path.join(inputDir, 'col_high_%04d.png' % frame_i)
                                        for frame_i in range(fi, conf.data.max_frm - conf.train.rnn_n + fi + 1)]

    return image_grid_HR_r


# moving decision
def movingDecision(conf):

    offset_xy = np.floor(
        np.random.uniform(-3.5, 4.5, [conf.train.rnn_n, 2])).astype(int)

    # [FLAGS.RNN_N , 2], shifts
    pos_xy_tmp = np.cumsum(offset_xy, axis=0)
    pos_xy = np.zeros([conf.train.rnn_n, 2])
    pos_xy[1:, :] = pos_xy_tmp[:-1, :]

    min_pos = np.min(pos_xy, axis=0)
    range_pos = np.max(pos_xy, axis=0) - min_pos  # [ shrink x, shrink y ]
    lefttop_pos = pos_xy - min_pos  # crop point
    moving_decision = np.random.uniform(0, 1, []).astype(float)

    return (lefttop_pos).astype(int), (range_pos).astype(int), moving_decision


def preprocess(image):
    # [0, 1] => [-1, 1]
    return image * 2 - 1


def data_iterator_sr(conf, num_samples, sample_names, tar_size, shuffle, rng=None):

    def dataset_load_func(i):

        # moving first frame -> data augmentation
        # our data augmentation, moving first frame to mimic camera motion
        if conf.train.movingFirstFrame :
            lefttop_pos, range_pos, moving_decision = movingDecision(conf)

        HR_data = []  # high res rgb, in range 0-1, shape any

        for fi in range(conf.train.rnn_n):
            img_name = sample_names[fi][i]
            image = Image.open(img_name)
            img_data = asarray(image).astype(float)
            img_data = img_data/255

            if(conf.train.movingFirstFrame):
                if(fi == 0):
                    img_data_0 = img_data
                    target_size = img_data.shape

                img_data_1 = img_data_0[lefttop_pos[fi][1]:target_size[0] - range_pos[1] + lefttop_pos[fi]
                                        [1], lefttop_pos[fi][0]:target_size[1] - range_pos[0] + lefttop_pos[fi][0], :]
                # random data augmentation -> move first frame only with 70% probability
                img_data = img_data if moving_decision < 0.7 else img_data_1

            HR_data.append(img_data)

        # random crop each batch entry separately
        # Check whether perform crop
        if (conf.train.random_crop is True):
            cur_size = HR_data[0].shape
            offset_h = np.floor(np.random.uniform(
                0, cur_size[0] - tar_size, [])).astype(int)
            offset_w = np.floor(np.random.uniform(
                0, cur_size[1] - tar_size, [])).astype(int)
            for frame_t in range(conf.train.rnn_n):
                HR_data[frame_t] = HR_data[frame_t][offset_h:offset_h +
                                                    tar_size, offset_w:offset_w + tar_size, :]

        # random flip:
        if (conf.train.flip is True):
            # Produce the decision of random flip
            flip_decision = np.random.uniform(0, 1, []).astype(float)
            for frame_t in range(conf.train.rnn_n):
                if (flip_decision < 0.5):
                    np.fliplr(HR_data[frame_t])

        HR_frames = HR_data
        target_frames = []

        k_w_border = int(1.5 * 3.0)
        for ft in range(conf.train.rnn_n):
            # crop out desired data
            croped_data = HR_data[ft][k_w_border:k_w_border+conf.train.crop_size*4,
                                      k_w_border:k_w_border+conf.train.crop_size*4, :]
            pre_procced_data = preprocess(croped_data)
            target_frames.append(pre_procced_data)

        return HR_frames, target_frames

    return data_iterator_simple(dataset_load_func, num_samples, conf.train.batch_size, shuffle=shuffle, rng=rng,
                                with_file_cache=False, with_memory_cache=False)
