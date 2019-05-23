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

import train
import model_inference

import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as M
import numpy as np

from args import get_args
from segmentation_data import data_iterator_segmentation


def validate(args):

   # load trained param file
    _ = nn.load_parameters(args.model_load_path)

    # get data iterator
    vdata = data_iterator_segmentation(args.val_samples, args.batch_size, args.val_dir,
                                       args.val_label_dir, target_width=args.image_width, target_height=args.image_height)

    # get deeplabv3plus model
    v_model = train.get_model(args, test=True)
    v_model.pred.persistent = True  # Not clearing buffer of pred in forward
    v_pred2 = v_model.pred.unlinked()

    # Create monitor
    monitor = M.Monitor(args.monitor_path)
    monitor_miou = M.MonitorSeries("mean IOU", monitor, interval=1)

    l = 0.0
    e = 0.0
    vmiou = 0.
    # Evaluation loop
    for j in range(args.val_samples // args.batch_size):
        images, labels, masks = vdata.next()
        v_model.image.d = images
        v_model.label.d = labels
        v_model.mask.d = masks
        v_model.pred.forward(clear_buffer=True)
        miou = train.compute_miou(
            args.num_class, labels, np.argmax(v_model.pred.d, axis=1), masks)
        vmiou += miou
        print(j, miou)

    monitor_miou.add(0, vmiou / (args.val_samples / args.batch_size))
    return vmiou / args.val_samples


def main():
    args = get_args()
    rng = np.random.RandomState(1223)

    # Get context
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    miou = validate(args)


if __name__ == '__main__':
    '''
    Usage : python eval.py --model-load-path=/path to trained .h5 file --val-samples=no. of validation examples --batch-size=1 --val-dir=file containing paths to val images --val-label-dir=file containing paths to val labels --num-class=no. of categories
    '''

    main()
