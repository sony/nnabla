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


def download(url):
    from nnabla.utils.data_source_loader import download as dl
    dl(url, url.split('/')[-1], False)


def main():
    categories = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    weights = 'https://pjreddie.com/media/files/yolov2.weights'
    example_image = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg'
    print('Downloading MS COCO category names ...')
    download(categories)
    print('Downloading Darknet YOLO weights ...')
    download(weights)
    print('Downloading an example image ...')
    download(example_image)


if __name__ == '__main__':
    main()
