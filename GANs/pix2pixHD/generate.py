import os
import numpy as np
from tqdm import trange

import nnabla as nn
import nnabla.functions as F
from nnabla.logger import logger
from nnabla.utils.image_utils import imsave

from models import LocalGenerator, encode_inputs
from data_iterator.data_loader import load_function
from utils import CommunicatorWrapper, Colorize, get_cityscape_datalist as get_datalist
from args import get_generation_args as get_args


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
    def __init__(self, batch_size, image_shape, g_n_scales, n_label_ids, use_encoder, load_path):
        self.image_shape = (batch_size, ) + image_shape
        self.g_n_scales = g_n_scales
        self.n_label_ids = n_label_ids
        self.use_encoder = use_encoder

        self.inst_label = nn.Variable(shape=self.image_shape)
        self.id_label = nn.Variable(shape=self.image_shape)

        self.fake = self.define_network()

        # load parameters
        if not os.path.exists(load_path):
            logger.warn("Path to load params is not found."
                        " Loading params is skipped and generated result will be unreasonable. ({})".format(load_path))

        nn.load_parameters(load_path)

    def define_network(self):
        id_onehot, bm = encode_inputs(self.inst_label, self.id_label,
                                      n_ids=self.n_label_ids, use_encoder=self.use_encoder)

        x = F.concatenate(id_onehot, bm, axis=1)

        generator = LocalGenerator()
        fake, _ = generator(x, self.g_n_scales)

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
    args = get_args()

    # batch_size is forced to be 1
    args.batch_size = 1

    image_shape = tuple(x * args.g_n_scales for x in [512, 1024])

    # set context
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)

    # init communicator
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(comm.ctx)

    # disable outputs from logger except rank==0
    if comm.rank > 0:
        from nnabla import logger
        import logging

        logger.setLevel(logging.ERROR)

    # find all test data
    data_list = get_datalist(args, data_type="val", save_file=comm.rank == 0)
    if comm.n_procs > 1:
        data_list = get_data_lists_for_each_process(
            data_list, comm.n_procs)[comm.rank]

    # define generator
    generator = Generator(args.batch_size, image_shape, args.g_n_scales,
                          args.n_label_ids, args.use_enc, args.load_path)

    progress_iterator = trange(len(data_list) // args.batch_size,
                               desc="[Generating Images]", disable=comm.rank > 0)

    # for label2color
    label2color = Colorize(args.n_label_ids)

    save_path = os.path.join(args.save_path, "generated")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    output_str = []
    for i in progress_iterator:
        paths = data_list[i]
        image, instance_id, object_id = load_function(
            paths[0], paths[1], paths[2], image_shape)
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
