# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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


# Multi-gpu training command for 1024 x 2048 generator on cityscapes dataset

N_GPU=4
LOAD_PATH="./result/cityscapes_512_1024/param_final.h5"

mpirun -N ${N_GPU} python train.py --type-config half --fix-global-epoch 20 --d-n-scales 3 --g-n-scales 2 --save-path ./result/cityscapes_1024_2048 --load-path ${LOAD_PATH}
