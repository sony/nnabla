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

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from cifar10_data import data_iterator_cifar10
from macro_CNN import CNN_run, get_data_stats, show_arch
from args import get_macro_args


def main():
    args = get_macro_args()

    if args.recommended_arch:
        filename = args.recommended_arch

    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)
    ext = nn.ext_utils.import_extension_module(args.context)

    data_iterator = data_iterator_cifar10
    tdata = data_iterator(args.batch_size, True)
    vdata = data_iterator(args.batch_size, False)
    mean_val_train, std_val_train, channel, img_height, img_width, num_class = get_data_stats(
        tdata)
    mean_val_valid, std_val_valid, _, _, _, _ = get_data_stats(vdata)

    data_dict = {"train_data": (tdata, mean_val_train, std_val_train),
                 "valid_data": (vdata, mean_val_valid, std_val_valid),
                 "basic_info": (channel, img_height, img_width, num_class)}

    check_arch = np.load(filename)
    print("Train the model whose architecture is:")
    show_arch(check_arch)

    val_acc = CNN_run(args, check_arch.tolist(), data_dict,
                      with_train=True, after_search=True)


if __name__ == '__main__':
    main()
