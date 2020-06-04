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
from skimage import measure
import open3d as o3d

import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as M
from nnabla.ext_utils import get_extension_context

from args import get_args
from models import MLP
from datasets import point_cloud_data_source, point_cloud_data_iterator
import utils


def main(args):
    # Context
    ctx = get_extension_context("cudnn", device_id=args.device_id)
    nn.set_default_context(ctx)

    # Dataset (input is normalized in [-1, 1])
    ds = point_cloud_data_source(args.fpath, knn=-1, test=True)
    pts_true = ds.points

    # Sample from mesh (unnormalized)
    mesh = utils.read_mesh(args.mesh_data_path)
    pcd = mesh.sample_points_poisson_disk(ds.size, seed=412)
    pts_pred = np.asarray(pcd.points)
    pts_pred = utils.normalize(pts_pred)

    # Pair-wise distance
    cd0, cd1, cd, hd0, hd1, hd = utils.chamfer_hausdorff_dists(
        pts_pred, pts_true)

    # Chamfer distance
    print("----- Chamfer distance -----")
    log = """
    One-sided Chamfer distance (Pred, True):   {}
    One-sided Chamfer distance (True, Pred):   {}
    Chamfer distance:                          {}
    """.format(cd0, cd1, cd)
    print(log)

    # Hausdorff distance
    print("----- Hausdorff distance -----")
    log = """
    One-sided Hausdorff distance (Pred, True): {}
    One-sided Hausdorff distance (True, Pred): {}
    Hausdorff distance:                        {}
    """.format(hd0, hd1, hd)
    print(log)


if __name__ == '__main__':
    args = get_args()
    main(args)
