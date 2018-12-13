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


from nnabla.utils.data_iterator import data_iterator_cache, data_iterator_simple
from PIL import Image
import glob
import numpy as np


# def data_iterator_imagenet(batch_size, cache_dir, rng=None):
#     return data_iterator_cache(cache_dir, batch_size, shuffle=True, normalize=False, rng=rng)

def create_dirname_label_maps(dirname_to_label_path):
    dirname_to_label = {}
    label_to_dirname = {}
    with open(dirname_to_label_path) as fp:
        for l in fp:
            d, l = l.rstrip().split(" ")
            dirname_to_label[d] = int(l)
            label_to_dirname[int(l)] = d
    return dirname_to_label, label_to_dirname


class dummy_iterator_imagenet():
    def __init__(self, batch_size, n_classes=1000):
        self.batch_size = batch_size
        self.n_classes = n_classes
    def next(self, ):
        x = np.random.rand(self.batch_size, 3, 128, 128)
        y = np.random.choice(np.arange(self.n_classes), self.batch_size)
        return x, y
    

def data_iterator_imagenet(img_path, dirname_to_label_path,
                           batch_size=16, ih=128, iw=128, n_classes=1000, 
                           class_id=-1,
                           noise=True, 
                           normalize=lambda x: x / 128.0 - 1.0, 
                           train=True, shuffle=True, rng=None):
    # ------
    # Valid
    # ------
    if not train:
        # Classes (but this tmpdir in ImageNet case)
        dir_paths = glob.glob("{}/*".format(img_path))
        dir_paths.sort()
        dir_paths = dir_paths[0:n_classes]

        # Images
        imgs = []
        for dir_path in dir_paths:
            imgs += glob.glob("{}/*.JPEG".format(dir_path))

        def load_func(i):
            # image
            img = Image.open(imgs[i]).resize((iw, ih), Image.BILINEAR).convert("RGB")
            img = np.asarray(img)
            img = img.transpose((2, 0, 1))
            img = img / 128.0 - 1.0
            return img, None
        di = data_iterator_simple(
            load_func, len(imgs), batch_size, shuffle=shuffle, rng=rng, with_file_cache=False)
        return di

    # ------
    # Train
    # ------
    # Classes
    dir_paths = glob.glob("{}/*".format(img_path))
    dir_paths.sort()
    dir_paths = dir_paths[0:n_classes]
    
    # Images
    imgs = []
    for dir_path in dir_paths:
        imgs += glob.glob("{}/*.JPEG".format(dir_path))
    #np.random.shuffle(imgs)
    
    # Dirname to Label map
    dirname_to_label, label_to_dirname = create_dirname_label_maps(dirname_to_label_path)

    # Filter by class_id
    if class_id != -1:
        dirname = label_to_dirname[class_id]
        imgs = list(filter(lambda img: dirname in img, imgs))

    def load_func(i):
        # image
        img = Image.open(imgs[i]).resize((iw, ih), Image.BILINEAR).convert("RGB")
        img = np.asarray(img)
        img = img.transpose((2, 0, 1))
        img = img / 128.0 - 1.0
        if noise:
            img += np.random.uniform(size=img.shape, low=0.0, high=1.0 / 128)
        # label
        elms = imgs[i].rstrip().split("/")
        dname = elms[-2]
        label = dirname_to_label[dname]
        return img, label
        
    di = data_iterator_simple(
        load_func, len(imgs), batch_size, shuffle=shuffle, rng=rng, with_file_cache=False)
    return di


def main():
    img_path = "/home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan"
    dirname_to_label_path = "/home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label.txt"
    di = data_iterator_imagenet(img_path, dirname_to_label_path)
    itr = 1620
    for i in range(itr):
        x, y = di.next()
        print(i, x.shape)
        print(i, y.shape)
        if x.shape != (16, 3, 128, 128):
            for i, u in enumerate(x):
                print(i, u.shape)
            break

if __name__ == '__main__':
    main()
