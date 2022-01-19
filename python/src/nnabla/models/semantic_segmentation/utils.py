# Copyright 2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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
from nnabla.models.semantic_segmentation import image_preprocess
from nnabla.utils.image_utils import imsave, imresize


def get_color():
    # RGB format
    return np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [120, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128], [224, 224, 192], [0, 0, 0]])


def pre_process_image(image, target_h, target_w):
    processed_image = image_preprocess.preprocess_image_and_label(
        image, target_w, target_h)
    processed_image = np.transpose(processed_image, (2, 0, 1))
    processed_image = np.reshape(
        processed_image, (1, processed_image.shape[0], processed_image.shape[1], processed_image.shape[2]))
    return processed_image


def post_process_image(output, image, target_size):
    old_size = image.shape[:2]
    ratio = min(np.divide(target_size, old_size))
    new_size = (int(old_size[0]*ratio), int(old_size[1]*ratio))
    post_processed = output[0:new_size[0], 0:new_size[1]]
    post_processed = (imresize(
        post_processed, (old_size[1], old_size[0]), interpolate='nearest'))
    return (post_processed)


def save_image(image_path, label, color_map=None):
    h, w = label.shape
    vis = np.zeros((h, w, 3), dtype=np.int32)
    if color_map is None:
        color_map = get_color()
    for y in range(h):
        for x in range(w):
            vis[y][x] = (color_map[int(label[y][x])])

    vis = vis/(np.amax(vis))
    imsave(image_path, vis)


class ProcessImage(object):

    '''
    Input image is pre-processed before passing it to the network. This class applies the pre-processing to input image. It also
    applies post-processing to the output of the network and saves the output as an image.
    '''

    def __init__(self, image, target_h, target_w):
        self.image = image
        self.target_size = (target_h, target_w)
        self.target_h, self.target_w = target_h, target_w

    def pre_process(self):
        '''
        Before passing the input image to the network, it is resized to target shape, maintaining the aspect ratio.
        This is done in order to obtain better results if the input image shape is very small or very large than the
        the image shape on which the network was trained. The default shape for target image is (513,513).
        '''

        return pre_process_image(self.image, self.target_h, self.target_w)

    def post_process(self, output):
        '''
        The input image is resized from network output shape to original input shape maintaining the aspect ratio.

        Args:
                output :
                     Output from the network.

        '''

        self.post_proc = post_process_image(
            output, self.image, self.target_size)
        return self.post_proc

    def save_segmentation_image(self, path, color_map=None):
        '''
        The pre processed image is finally visulalized and saved. Each category of the label has been
        assigned a particular color and same color mapping is used during visualization. When the color_map
        is None, the number of categories by default is 20 and the color mapping is selected from a pre defined
        color mapping array. In case you want to save (say) 40 class output labels, color map may be defined as 
        `color_map=np.random.rand(40, 3)`. This is an example of color sampling from RGB space.

        Args:
                label(numpy.ndarray):
                     Post processed image.

        '''

        return save_image(path, self.post_proc, color_map)
