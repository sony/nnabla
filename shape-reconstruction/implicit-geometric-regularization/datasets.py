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

import glob
import os
import numpy as np
from scipy import spatial
import open3d as o3d

import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as M
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.utils.data_iterator import data_iterator, data_iterator_simple
from nnabla.utils.data_source import DataSource
import utils

from args import get_args


class PointCloudDataSource(DataSource):

    def __init__(self, fpath, knn=50, test_rate=0.25, test=False, shuffle=True, rng=None):
        super(PointCloudDataSource, self).__init__(shuffle=shuffle)
        self.knn = knn
        self.test_rate = 0.25
        self.rng = np.random.RandomState(313) if rng is None else rng

        # Split info
        pcd = self._read_dataset(fpath)
        total_size = len(pcd.points)
        test_size = int(total_size * test_rate)
        indices = self.rng.permutation(total_size)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        indices = test_indices if test else train_indices
        self._size = test_size if test else total_size - test_size
        # Points
        points = np.asarray(pcd.points)
        self._points = self._preprocess(points)[indices]
        # Normals
        normals = np.asarray(pcd.normals)
        self._normals = normals[indices] if self.has_normals(
            normals) else normals
        # Radius
        self._radius = self._compute_radius(self._points, self.knn)
        self._variables = ('points', 'normals', 'radius')
        self.reset()

        logger.info("Data size = {}".format(self._size))

    def has_normals(self, normals):
        return False if normals.shape[0] == 0 else True

    def _preprocess(self, points):
        return utils.normalize(points)

    def _compute_radius(self, points, knn):
        if knn < 0:
            logger.info("Radius is not computed.")
            return
        # KDTree
        logger.info(
            "Constructing KDTree and querying {}-nearest neighbors".format(self.knn))
        tree = spatial.cKDTree(points, compact_nodes=True)
        # exclude self by adding 1
        dists, indices = tree.query(points, k=knn + 1)
        return dists[:, -1].reshape(dists.shape[0], 1)

    def _read_dataset(self, fpath):
        pcd = utils.read_pcd(fpath)
        return pcd

    def _get_data(self, position):
        points = self._points[self._indices[position]]
        normals = self._normals[self._indices[position]
                                ] if self.has_normals(self._normals) else [0.0]
        radius = self._radius[self._indices[position]]
        return points, normals, radius

    @property
    def points(self):
        return self._points

    @property
    def normals(self):
        return self._normals

    @property
    def radius(self):
        return self._radius

    def reset(self):
        self._indices = self.rng.permutation(self._size) \
            if self._shuffle else np.arange(self._size)
        return super(PointCloudDataSource, self).reset()


def point_cloud_data_source(fpath, knn=50, test_rate=0.25, test=False, shuffle=True, rng=None):
    return PointCloudDataSource(fpath, knn, test_rate, test, shuffle, rng)


def point_cloud_data_iterator(data_source, batch_size):
    return data_iterator(data_source, batch_size=batch_size,
                         with_memory_cache=False,
                         with_file_cache=False)


def create_pcd_dataset_from_mesh(fpath):
    mesh = utils.read_mesh(fpath)
    pcd = mesh.sample_points_poisson_disk(len(mesh.vertices),
                                          use_triangle_normal=True,
                                          seed=412)
    dpath = "/".join(fpath.split("/")[:-1])
    fname = fpath.split("/")[-1]
    fname = "{}_pcd.ply".format(os.path.splitext(fname)[0])
    fpath = os.path.join(dpath, fname)
    logger.info("PointCloud data ({}) is being created.".format(fpath))
    utils.write_pcd(fpath, pcd)


def main():
    args = get_args()
    if args.command == "create":
        create_pcd_dataset_from_mesh(args.mesh_data_path)


if __name__ == '__main__':
    main()
