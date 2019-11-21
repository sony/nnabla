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

from models import LocalGenerator, encode_inputs
from utils import *


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", "-L", required=True, type=str)
    args, subargs = parser.parse_known_args()

    conf_path = os.path.join(os.path.dirname(args.load_path), "config.yaml")
    conf = read_yaml(conf_path)

    conf.load_path = args.load_path

    return conf


def get_data_lists_for_each_process(data_list, n_procs):
    base_len = len(data_list) // n_procs
    remains = len(data_list) % n_procs

    res = []
    left = 0
    for i in range(n_procs):
        right = left + base_len + int(remains > i)
        res.append(data_list[left:right])
        left = right + 1

    return res


class Generator(object):
    def __init__(self, image_shape, mconf, use_encoder=False):
        self.image_shape = image_shape
        self.model_conf = mconf
        self.use_encoder = False  # Currently encoder is not supported.

        self.inst_label = nn.Variable(shape=image_shape)
        self.id_label = nn.Variable(shape=image_shape)

        self.fake = self.define_network()

    def define_network(self):
        id_onehot, bm = encode_inputs(self.inst_label, self.id_label,
                                      n_ids=self.model_conf.n_label_ids,
                                      use_encoder=self.use_encoder)

        x = F.concatenate(id_onehot, bm, axis=1)

        generator = LocalGenerator()
        fake, _ = generator(x,
                            lg_channels=self.model_conf.lg_channels,
                            gg_channels=self.model_conf.gg_channels,
                            n_scales=self.model_conf.g_n_scales,
                            lg_n_residual_layers=self.model_conf.lg_num_residual_loop,
                            gg_n_residual_layers=self.model_conf.gg_num_residual_loop)

        return fake

    @staticmethod
    def _check_ndarray(x):
        if not isinstance(x, np.ndarray):
            raise ValueError("image must be np.ndarray.")

    def __call__(self, inst_label, object_id):
        self._check_ndarray(inst_label)
        self._check_ndarray(object_id)

        self.inst_label.d = inst_label
        self.id_label.d = object_id
        self.fake.forward(clear_buffer=True)

        return self.fake.d


def generate():
    conf = get_config()

    # batch_size is forced to be 1
    conf.train.batch_size = 1

    image_shape = (conf.train.batch_size,) + \
        tuple(x * conf.model.g_n_scales for x in [512, 1024])

    # set context
    comm = init_nnabla(conf.nnabla_context)

    # find all test data
    if conf.train.data_set == "cityscapes":
        data_list = get_cityscape_datalist(
            conf.cityscapes, data_type="val", save_file=comm.rank == 0)
        conf.model.n_label_ids = conf.cityscapes.n_label_ids
    else:
        raise NotImplementedError(
            "Currently dataset {} is not supported.".format(conf.dataset))

    if comm.n_procs > 1:
        data_list = get_data_lists_for_each_process(
            data_list, comm.n_procs)[comm.rank]

    # define generator
    generator = Generator(image_shape=image_shape, mconf=conf.model)

    # load parameters
    if not os.path.exists(conf.load_path):
        logger.warn("Path to load params is not found."
                    " Loading params is skipped and generated result will be unreasonable. ({})".format(conf.load_path))

    nn.load_parameters(conf.load_path)

    progress_iterator = trange(len(data_list) // conf.train.batch_size,
                               desc="[Generating Images]", disable=comm.rank > 0)

    # for label2color
    label2color = Colorize(conf.model.n_label_ids)

    save_path = os.path.join(conf.train.save_path, "generated")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    output_str = []
    for i in progress_iterator:
        paths = data_list[i]
        image, instance_id, object_id = cityscapes_load_function(
            paths[0], paths[1], paths[2], image_shape[1:])
        gen = generator(instance_id, object_id)
        gen = (gen - gen.min()) / (gen.max() - gen.min())
        id_colorized = label2color(object_id).astype(np.uint8)

        gen_image_path = os.path.join(
            save_path, "res{}_{}.png".format(comm.rank, i))
        input_image_path = os.path.join(
            save_path, "input_{}_{}.png".format(comm.rank, i))

        imsave(gen_image_path, gen[0], channel_first=True)
        imsave(input_image_path, id_colorized)
        output_str.append(
            " ".join([x for x in paths + [gen_image_path, input_image_path]]))

    if comm.rank == 0:
        with open(os.path.join(save_path, "in_out_pairs.txt"), "w") as f:
            f.write("\n".join(output_str))


if __name__ == '__main__':
    generate()
