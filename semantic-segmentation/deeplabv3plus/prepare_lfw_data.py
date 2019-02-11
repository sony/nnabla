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


import numpy as np
import os
from scipy.misc import imread
from args import get_args
import nnabla.utils.image_utils as utils


def get_prefix(word):
    '''
    Get the prefix to be used to search for image/label
    '''
    if len(str(int(word))) == 1:
        prefix = '_000'
    elif len(str(int(word))) == 2:
        prefix = '_00'
    else:
        prefix = '_0'
    return prefix


def encode_label(label):
    '''
    Converting pixel values to corresponding class numbers. Assuming that the input label in 2-dim(h,w)
    '''

    h, w, _ = label.shape
    new_label = np.zeros((h, w, 1), dtype=np.int32)

    for c in range(3):
        new_label[label[..., c] != 0] = c

    return new_label


def encode_and_write_to_path_files(filename, data_dir, ti, tl):
    '''
    Calling encode_label for each label and writing image and label paths to path files
    '''
    train_f = open(filename, 'r')
    label_path = data_dir+'parts_lfw_funneled_gt_images/'
    image_path = data_dir+'lfw_funneled/'
    for line in train_f:
        words = line.split(' ')
        prefix = get_prefix(words[1])

        if os.path.isdir(image_path + words[0] + '/'):
            ti.write(image_path + words[0] + '/' + words[0] +
                     prefix+str(int(words[1]))+'.jpg' + '\n')
            assert (os.path.isfile(label_path + words[0]+prefix+str(int(words[1]))+'.ppm')
                    ), "No matching label file for image : " + words[0]+prefix+str(int(words[1])) + '.jpg'
            label = utils.imread(
                label_path + words[0]+prefix+str(int(words[1]))+'.ppm')
            label = encode_label(label)
            np.save(label_path + 'encoded/' +
                    words[0]+prefix+str(int(words[1])) + '.npy', label)
            tl.write(label_path + 'encoded/' +
                     words[0]+prefix+str(int(words[1])) + '.npy' + '\n')


# this method should generate lfw_train_image.txt, lfw_train_label.txt, lfw_val_image.txt, lfw_val_label.txt
def generate_path_files(data_dir, train_file, val_file):

    ti = open('lfw_train_image.txt', 'w')
    tl = open('lfw_train_label.txt', 'w')
    vi = open('lfw_val_image.txt', 'w')
    vl = open('lfw_val_label.txt', 'w')

    if not os.path.exists(data_dir+'parts_lfw_funneled_gt_images/' + 'encoded/'):
        os.makedirs(data_dir+'parts_lfw_funneled_gt_images/' + 'encoded/')

    encode_and_write_to_path_files(train_file, data_dir, ti, tl)

    encode_and_write_to_path_files(val_file, data_dir, vi, vl)

    ti.close()
    tl.close()
    vi.close()
    vl.close()


def main():
    '''
    Arguments:
    train-file = txt file containing randomly selected image filenames to be taken as training set.
    val-file = txt file containing randomly selected image filenames to be taken as validation set.
    data-dir = dataset directory
    Usage: python dataset_utils.py --train-file="" --val-file="" --data_dir=""
    '''

    args = get_args()
    data_dir = args.data_dir

    generate_path_files(data_dir, args.train_file, args.val_file)


if __name__ == '__main__':
    main()
