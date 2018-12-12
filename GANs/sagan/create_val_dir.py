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


import os
import tarfile
import argparse
import tqdm
import shutil

parser = argparse.ArgumentParser()
parser.add_argument(
    '-t', '--tarfile', help='train tar file of imagenet', required=True)
parser.add_argument(
    '-o', '--outdir', help='output directory', required=True)
args = parser.parse_args()

source_tar_file = args.tarfile
dst_dir = args.outdir

with tarfile.open(source_tar_file) as tf:
    v_tmp_dir = dst_dir + '/' + 'tmpdir'
    tf.extractall(v_tmp_dir)
with open(os.path.join(os.path.dirname(__file__), "category_list.txt")) as f:
    categories = f.readlines()
with open(os.path.join(os.path.dirname(__file__), "val_data_category_list.txt")) as f:
    v_data_categories = f.readlines()

for category in categories:
    os.mkdir(dst_dir + '/' + category.rstrip("\n"))

v_data_files = sorted(os.listdir(v_tmp_dir))

for v_data_file, v_data_category in zip(v_data_files, v_data_categories):
    shutil.move(v_tmp_dir + '/' + v_data_file, dst_dir + '/' +
                v_data_category.rstrip("\n") + '/' + v_data_file)

shutil.rmtree(v_tmp_dir)
