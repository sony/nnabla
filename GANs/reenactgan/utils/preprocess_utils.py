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

import numpy as np
import cv2

# utils

######################################
# Definition of the parts pair index #
######################################
FACIAL_OUTER_CONTOUR_PAIR = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                             [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [
                                           15, 16], [16, 17], [17, 18], [18, 19], [19, 20],
                             [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [
                                           25, 26], [26, 27], [27, 28], [28, 29], [29, 30],
                             [30, 31], [31, 32]]
UPPER_SIDE_OF_LEFT_EYEBROW_PAIR = [[33, 34], [34, 35], [35, 36], [36, 37]]
LOWER_SIDE_OF_LEFT_EYEBROW_PAIR = [[38, 39], [39, 40], [40, 41]]
UPPER_SIDE_OF_RIGHT_EYEBROW_PAIR = [[42, 43], [43, 44], [44, 45], [45, 46]]
LOWER_SIDE_OF_RIGHT_EYEBROW_PAIR = [[47, 48], [48, 49], [49, 50]]
NOSE_BRIDGE_PAIR = [[51, 52], [52, 53], [53, 54]]
NOSE_BOUNDARY_PAIR = [[55, 56], [56, 57], [57, 58], [58, 59]]
LEFT_UPPER_EYELID_PAIR = [[60, 61], [61, 62], [62, 63], [63, 64]]
LEFT_LOWER_EYELID_PAIR = [[64, 65], [65, 66], [66, 67], [67, 60]]
RIGHT_UPPER_EYELID_PAIR = [[68, 69], [69, 70], [70, 71], [71, 72]]
RIGHT_LOWER_EYELID_PAIR = [[72, 73], [73, 74], [74, 75], [75, 68]]
UPPER_SIDE_OF_UPPER_LIP_PAIR = [
    [76, 77], [77, 78], [78, 79], [80, 81], [81, 82]]
LOWER_SIDE_OF_UPPER_LIP_PAIR = [[88, 89], [89, 90], [90, 91], [91, 92]]
UPPER_SIDE_OF_LOWER_LIP_PAIR = [[92, 93], [93, 94], [94, 95], [95, 88]]
LOWER_SIDE_OF_LOWER_LIP_PAIR = [[82, 83], [
    83, 84], [84, 85], [85, 86], [86, 87], [87, 76]]


def get_part_pair_index(id):
    PART_INDEX_PAIR_LIST = []
    if id == 0:
        PART_INDEX_PAIR_LIST = FACIAL_OUTER_CONTOUR_PAIR
    elif id == 1:
        PART_INDEX_PAIR_LIST = UPPER_SIDE_OF_LEFT_EYEBROW_PAIR
    elif id == 2:
        PART_INDEX_PAIR_LIST = LOWER_SIDE_OF_LEFT_EYEBROW_PAIR
    elif id == 3:
        PART_INDEX_PAIR_LIST = UPPER_SIDE_OF_RIGHT_EYEBROW_PAIR
    elif id == 4:
        PART_INDEX_PAIR_LIST = LOWER_SIDE_OF_RIGHT_EYEBROW_PAIR
    elif id == 5:
        PART_INDEX_PAIR_LIST = NOSE_BRIDGE_PAIR
    elif id == 6:
        PART_INDEX_PAIR_LIST = NOSE_BOUNDARY_PAIR
    elif id == 7:
        PART_INDEX_PAIR_LIST = LEFT_UPPER_EYELID_PAIR
    elif id == 8:
        PART_INDEX_PAIR_LIST = LEFT_LOWER_EYELID_PAIR
    elif id == 9:
        PART_INDEX_PAIR_LIST = RIGHT_UPPER_EYELID_PAIR
    elif id == 10:
        PART_INDEX_PAIR_LIST = RIGHT_LOWER_EYELID_PAIR
    elif id == 11:
        PART_INDEX_PAIR_LIST = UPPER_SIDE_OF_UPPER_LIP_PAIR
    elif id == 12:
        PART_INDEX_PAIR_LIST = LOWER_SIDE_OF_UPPER_LIP_PAIR
    elif id == 13:
        PART_INDEX_PAIR_LIST = UPPER_SIDE_OF_LOWER_LIP_PAIR
    elif id == 14:
        PART_INDEX_PAIR_LIST = LOWER_SIDE_OF_LOWER_LIP_PAIR
    return PART_INDEX_PAIR_LIST


def get_points_list(ant, stop_idx=-1):
    points_list = ant.split(' ')[:stop_idx]
    x_list = points_list[0::2]
    y_list = points_list[1::2]
    x_float_list = [float(x) for x in x_list]
    y_float_list = [float(y) for y in y_list]
    x_int_list = [int(x) for x in x_float_list]
    y_int_list = [int(y) for y in y_float_list]
    return x_int_list, y_int_list


def get_bod_map(img, x_list, y_list, resize_size=(64, 64), line_thickness=3, gaussian_kernel=(5, 5), gaussian_sigma=3):
    if not isinstance(resize_size, tuple):
        resize_size = 2 * (resize_size, )
    if not isinstance(gaussian_kernel, tuple):
        gaussian_kernel = 2 * (gaussian_kernel, )

    img = np.zeros((img[0].shape), np.uint8)
    img_size = max(img.shape[0], img.shape[1])
    line_thickness = int((line_thickness/256)*img_size)
    bod_list = []
    for i in range(15):
        bod_part = np.zeros((img.shape), np.uint8)
        for pair in get_part_pair_index(i):
            bod_part = cv2.line(bod_part, (x_list[pair[0]], y_list[pair[0]]),
                                (x_list[pair[1]], y_list[pair[1]]), (255), line_thickness)
        bod_part = cv2.GaussianBlur(bod_part, gaussian_kernel, gaussian_sigma)
        bod_part = cv2.resize(bod_part, (resize_size))
        bod_list.append(bod_part)
    bod_map = np.stack(bod_list, 0)
    return bod_map


def get_bod_img(img, x_list, y_list, resize_size=(64, 64), line_thickness=3, gaussian_kernel=(5, 5), gaussian_sigma=3):
    if not isinstance(resize_size, tuple):
        resize_size = 2 * (resize_size, )
    if not isinstance(gaussian_kernel, tuple):
        gaussian_kernel = 2 * (gaussian_kernel, )

    img = np.zeros((img[0].shape), np.uint8)
    img_size = max(img.shape[0], img.shape[1])
    line_thickness = int((line_thickness/256)*img_size)
    bod_img = np.zeros((img.shape), np.uint8)
    for i in range(15):
        for pair in get_part_pair_index(i):
            bod_img = cv2.line(bod_img, (x_list[pair[0]], y_list[pair[0]]),
                               (x_list[pair[1]], y_list[pair[1]]), (255, 255, 255), line_thickness)
        bod_img = cv2.GaussianBlur(bod_img, gaussian_kernel, gaussian_sigma)
    bod_img = cv2.resize(bod_img, (resize_size))
    bod_img = cv2.cvtColor(bod_img, cv2.COLOR_GRAY2BGR)
    bod_img = bod_img.transpose((2, 0, 1))
    return bod_img
