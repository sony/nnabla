# Copyright (c) 2017-2020 Sony Corporation. All Rights Reserved.
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

from __future__ import division

import os
import tqdm
import numpy as np
import open3d as o3d

import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as M
from nnabla.ext_utils import get_extension_context

from args import get_args
from models import MLP
import utils


def main(args):
    # Context
    ctx = get_extension_context("cudnn", device_id=args.device_id)
    nn.set_default_context(ctx)

    # Network
    nn.load_parameters(args.model_load_path)
    model = MLP(args.dims, args.ldims, test=True)

    # Compute points and values
    pts, vol = utils.compute_pts_vol(model, args.grid_size, args.volume_factor,
                                     args.sub_batch_size)
    mesh = utils.create_mesh_from_volume(vol, args.gradient_direction)

    # Save as mesh
    dirname, pname = args.model_load_path.split("/")
    fname = os.path.splitext(pname)[0]
    fpath = "{}/{}_mesh.ply".format(dirname, fname)
    logger.info("Saving the mesh file to {}".format(fpath))
    utils.write_mesh(fpath, mesh)


if __name__ == '__main__':
    args = get_args()
    main(args)
