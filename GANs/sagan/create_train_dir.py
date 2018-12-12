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
    for tar_file_info in tqdm.tqdm(tf.getmembers()):
        fullname = tar_file_info.name
        name, ext = os.path.splitext(os.path.basename(fullname))
        category_dir = dst_dir + '/' + name
        os.mkdir(category_dir)
        fileobj = tf.extractfile(tar_file_info)
        with tarfile.open(fileobj=fileobj) as tf_class:
            tf_class.extractall(category_dir)
