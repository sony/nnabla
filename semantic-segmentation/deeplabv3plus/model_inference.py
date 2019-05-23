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

import cv2
import numpy as np
import nnabla as nn
from nnabla.logger import logger
import model
import model as net
from args import get_args
import imageio
from PIL import Image
import image_preprocess
import dataset_utils
import matplotlib.pyplot as plt
import time


def visualize(label):
    h, w = label.shape
    vis = np.zeros((h, w, 3), dtype=np.int32)
    clr_map = dataset_utils.get_color()
    for y in range(h):
        for x in range(w):
            if label[y][x] == 22:
                vis[y][x] = 0
            else:
                vis[y][x] = clr_map[label[y][x]]

    #plt.imshow(vis, interpolation='none')
    # plt.show()
    imageio.imwrite('output.png', vis)


def post_process(output, old_size, target_size):
    ratio = min(np.divide(desired_size, old_size))
    new_size = (int(old_size[0]*ratio), int(old_size[1]*ratio))

    post_processed = output[0:new_size[0], 0:new_size[1]]

    post_processed = cv2.resize(
        post_processed, (old_size[1], old_size[0]), interpolation=cv2.INTER_NEAREST)

    return post_processed


def main():
    args = get_args()
    rng = np.random.RandomState(1223)

    # Get context
    from nnabla.ext_utils import get_extension_context, import_extension_module
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)
    ext = import_extension_module(args.context)

    # read label file
    f = open(args.label_file_path, "r")
    labels_dict = f.readlines()

    # Load parameters
    _ = nn.load_parameters(args.model_load_path)

    # Build a Deeplab v3+ network
    x = nn.Variable(
        (1, 3, args.image_height, args.image_width), need_grad=False)
    y = net.deeplabv3plus_model(
        x, args.output_stride, args.num_class, test=True)

    # preprocess image
    image = imageio.imread(args.test_image_file, as_gray=False, pilmode="RGB")
    #image = imread(args.test_image_file).astype('float32')
    orig_h, orig_w, orig_c = image.shape
    old_size = (orig_h, orig_w)

    input_array = image_preprocess.preprocess_image_and_label(
        image, label=None, target_width=args.image_width, target_height=args.image_height, train=False)
    print('Input', input_array.shape)
    input_array = np.transpose(input_array, (2, 0, 1))
    input_array = np.reshape(
        input_array, (1, input_array.shape[0], input_array.shape[1], input_array.shape[2]))

    # Compute inference and inference time
    t = time.time()

    x.d = input_array
    y.forward(clear_buffer=True)
    print ("done")
    available_devices = ext.get_devices()
    ext.device_synchronize(available_devices[0])
    ext.clear_memory_cache()

    elapsed = time.time() - t
    print('Inference time : %s seconds' % (elapsed))

    output = np.argmax(y.d, axis=1)  # (batch,h,w)

    # Apply post processing
    post_processed = post_process(
        output[0], old_size, (args.image_height, args.image_width))

    # Get the classes predicted
    predicted_classes = np.unique(post_processed)
    for i in range(predicted_classes.shape[0]):
        print('Classes Segmented: ', labels_dict[predicted_classes[i]])

    # Visualize inference result
    visualize(post_processed)


if __name__ == '__main__':
    '''
    Usage : python model_inference.py --model-load-path=/path to trained .h5 file --image-width=target width for input image --test-image-file=image file for inference --num-class=no. of categories --label-file-path=txt file having categories --output-stride=8 or 16
    '''

    main()
