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

# Generation command for 512 x 1024 generator on cityscapes dataset

MODEL_PATH="./result/cityscapes_512_1024"

python generate.py --g-n-scales 1 --save-path ${MODEL_PATH} --load-path "${MODEL_PATH}/param_final.h5"
