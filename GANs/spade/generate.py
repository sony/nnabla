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
import argparse
import numpy as np
from tqdm import trange

import nnabla as nn
import nnabla.functions as F
from nnabla.logger import logger
from nnabla.utils.image_utils import imsave

from models import SpadeGenerator, encode_inputs
from utils import *


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_params", "-L", required=True, type=str,
                        help="A path for parameter to load. "
                             "model config file (config.yaml) is automatically detected on the same directory."
                             "If you want to change model settings, you can edit this config.yaml manually.")
    args, subargs = parser.parse_known_args()

    conf_path = os.path.join(os.path.dirname(args.load_params), "config.yaml")
    conf = read_yaml(conf_path)
    conf.save_path = os.path.dirname(args.load_params)

    conf.update(args.__dict__)

    return conf


class Generator(object):
    def __init__(self, conf, use_inst):
        self.conf = conf
        self.use_inst = use_inst

        self.ist_mask = nn.Variable(
            shape=(conf.batch_size, ) + conf.image_shape)
        self.obj_mask = nn.Variable(
            shape=(conf.batch_size, ) + conf.image_shape)

        self.fake = self.define_network()

    def define_network(self):

        if self.use_inst:
            obj_onehot, bm = encode_inputs(self.ist_mask, self.obj_mask,
                                           n_ids=self.conf.n_class)

            mask = F.concatenate(obj_onehot, bm, axis=1)
        else:
            om = self.obj_mask
            if len(om.shape) == 3:
                om = F.reshape(om, om.shape + (1,))
            obj_onehot = F.one_hot(om, shape=(self.conf.n_class, ))
            mask = F.transpose(obj_onehot, (0, 3, 1, 2))

        generator = SpadeGenerator(
            self.conf.g_ndf, image_shape=self.conf.image_shape)
        z = F.randn(shape=(self.conf.batch_size, self.conf.z_dim))
        fake = generator(z, mask)

        # Pixel intensities of fake are [-1, 1]. Rescale it to [0, 1]
        fake = (fake + 1) / 2

        return fake

    @staticmethod
    def _check_ndarray(x):
        if not isinstance(x, np.ndarray):
            raise ValueError("image must be np.ndarray.")

    def __call__(self, ist_label, obj_label):
        if self.use_inst and ist_label is not None:
            self._check_ndarray(ist_label)
            self.ist_mask.d = ist_label

        self._check_ndarray(obj_label)
        self.obj_mask.d = obj_label

        self.fake.forward(clear_buffer=True)

        return self.fake.d


def generate():
    rng = np.random.RandomState(803)

    conf = get_config()

    # set context
    comm = init_nnabla(conf)

    # find all test data
    if conf.dataset == "cityscapes":
        data_list = get_cityscape_datalist(
            conf.cityscapes, data_type="val", save_file=comm.rank == 0)
        conf.n_class = conf.cityscapes.n_label_ids
        use_inst = True

        data_iter = create_cityscapes_iterator(conf.batch_size, data_list, comm=comm,
                                               image_shape=conf.image_shape, rng=rng,
                                               flip=False)
    elif conf.dataset == "ade20k":
        data_list = get_ade20k_datalist(
            conf.ade20k, data_type="val", save_file=comm.rank == 0)
        conf.n_class = conf.ade20k.n_label_ids + 1  # class id + unknown
        use_inst = False

        load_shape = tuple(
            x + 30 for x in conf.image_shape) if conf.use_crop else conf.image_shape
        data_iter = create_ade20k_iterator(conf.batch_size, data_list, comm=comm,
                                           load_shape=load_shape, crop_shape=conf.image_shape,
                                           rng=rng, flip=False)
    else:
        raise NotImplementedError(
            "Currently dataset {} is not supported.".format(conf.dataset))

    # define generator
    generator = Generator(conf, use_inst)

    # load parameters
    if not os.path.exists(conf.load_params):
        logger.warn("Path to load params is not found."
                    " Loading params is skipped and generated result will be unreasonable. ({})".format(conf.load_params))

    else:
        print("load parameters from {}".format(conf.load_params))
        nn.load_parameters(conf.load_params)

    niter = get_iteration_per_epoch(
        data_iter._size, conf.batch_size, round="ceil")

    progress_iterator = trange(
        niter, desc="[Generating Images]", disable=comm.rank > 0)

    # for label2color
    label2color = Colorize(conf.n_class)

    save_path = os.path.join(conf.save_path, "generated")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    logger.info("Generated images will be saved on '{}'.".format(save_path))

    cnt = 0
    for i in progress_iterator:
        if conf.dataset == "cityscapes":
            _, instance_id, object_id = data_iter.next()
        elif conf.dataset == "ade20k":
            _, object_id = data_iter.next()
            instance_id = None
        else:
            raise NotImplemented()

        gen = generator(instance_id, object_id)
        id_colorized = label2color(object_id).astype(np.uint8)

        valid = conf.batch_size
        if cnt > data_iter._size:
            valid = data_iter._size - conf.batch_size * (i - 1)

        for j in range(valid):
            gen_image_path = os.path.join(
                save_path, "res_{}_{}.png".format(comm.rank, cnt + j))
            input_image_path = os.path.join(
                save_path, "input_{}_{}.png".format(comm.rank, cnt + j))

            imsave(gen_image_path, gen[j], channel_first=True)
            imsave(input_image_path, id_colorized[j])

        cnt += conf.batch_size


if __name__ == '__main__':
    generate()
