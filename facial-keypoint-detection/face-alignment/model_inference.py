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
import os
import dlib
import bz2
import nnabla as nn
import nnabla.functions as F
from nnabla.logger import logger
from skimage import io, color
from args import get_args
from model import fan, resnet_depth
from utils import *
from external_utils import *
import numpy as np


def main():
    args = get_args()
    # Get context
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)

    image = io.imread(args.test_image)
    if image.ndim == 2:
        image = color.gray2rgb(image)
    elif image.ndim == 4:
        image = image[..., :3]

    if args.context == 'cudnn':
        if not os.path.isfile(args.cnn_face_detction_model):
            # Block of bellow code will download the cnn based face-detection model file provided by dlib for face detection
            # and will save it in the directory where this script is executed.
            print("Downloading the face detection CNN. Please wait...")
            url = "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
            from nnabla.utils.data_source_loader import download
            download(url, url.split('/')[-1], False)
            # get the decompressed data.
            data = bz2.BZ2File(url.split('/')[-1]).read()
            # write to dat file.
            open(url.split('/')[-1][:-4], 'wb').write(data)
        face_detector = dlib.cnn_face_detection_model_v1(
            args.cnn_face_detction_model)
        detected_faces = face_detector(cv2.cvtColor(
            image[..., ::-1].copy(), cv2.COLOR_BGR2GRAY))
        detected_faces = [[d.rect.left(), d.rect.top(
        ), d.rect.right(), d.rect.bottom()] for d in detected_faces]
    else:
        face_detector = dlib.get_frontal_face_detector()
        detected_faces = face_detector(cv2.cvtColor(
            image[..., ::-1].copy(), cv2.COLOR_BGR2GRAY))
        detected_faces = [[d.left(), d.top(), d.right(), d.bottom()]
                          for d in detected_faces]

    if len(detected_faces) == 0:
        print("Warning: No faces were detected.")
        return None

    # Load FAN weights
    with nn.parameter_scope("FAN"):
        print("Loading FAN weights...")
        nn.load_parameters(args.model)

    # Load ResNetDepth weights
    if args.landmarks_type_3D:
        with nn.parameter_scope("ResNetDepth"):
            print("Loading ResNetDepth weights...")
            nn.load_parameters(args.resnet_depth_model)

    landmarks = []
    for i, d in enumerate(detected_faces):
        center = [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0]
        center[1] = center[1] - (d[3] - d[1]) * 0.12
        scale = (d[2] - d[0] + d[3] - d[1]) / args.reference_scale
        inp = crop(image, center, scale)
        inp = nn.Variable.from_numpy_array(inp.transpose((2, 0, 1)))
        inp = F.reshape(F.mul_scalar(inp, 1 / 255.0), (1,) + inp.shape)
        with nn.parameter_scope("FAN"):
            out = fan(inp, args.network_size)[-1]
        pts, pts_img = get_preds_fromhm(out, center, scale)
        pts, pts_img = F.reshape(pts, (68, 2)) * \
            4, F.reshape(pts_img, (68, 2))

        if args.landmarks_type_3D:
            heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
            for i in range(68):
                if pts.d[i, 0] > 0:
                    heatmaps[i] = draw_gaussian(
                        heatmaps[i], pts.d[i], 2)
            heatmaps = nn.Variable.from_numpy_array(heatmaps)
            heatmaps = F.reshape(heatmaps, (1,) + heatmaps.shape)
            with nn.parameter_scope("ResNetDepth"):
                depth_pred = F.reshape(resnet_depth(
                    F.concatenate(inp, heatmaps, axis=1)), (68, 1))
            pts_img = F.concatenate(
                pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale))), axis=1)

        landmarks.append(pts_img.d)
    visualize(landmarks, image, args.output)


if __name__ == '__main__':
    '''
    Usage : python model_inference.py --model=/path to pre-trained .h5 file --test-image=image file for inference 
    '''
    main()
