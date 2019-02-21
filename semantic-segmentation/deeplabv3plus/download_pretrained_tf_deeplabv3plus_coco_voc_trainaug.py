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

import tarfile
import nnabla


def download(url):
    from nnabla.utils.data_source_loader import download as dl
    dl(url, url.split('/')[-1], False)


def main():
    model_url = 'http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz'

    print('Downloading Xception COCO VOC trainaug pretrained model ...')
    download(model_url)

    # untar the downloaded compressed file
    compressed_file = tarfile.open(model_url.split('/')[-1], "r:gz")
    compressed_file.extractall()
    compressed_file.close()


if __name__ == '__main__':
    main()
