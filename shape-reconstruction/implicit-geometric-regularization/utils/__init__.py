# Copyright (c) 2019-2020 Sony Corporation. All Rights Reserved.
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

import sys
import os
import tqdm

import open3d as o3d
from skimage import measure
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import numpy as np
import nnabla as nn
import nnabla.functions as F

# Set path to neu
common_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utils'))
sys.path.append(common_utils_path)

from neu.save_args import save_args


def normalize(pts):
    """
    Normalize to [-1, 1]
    """
    size = pts.max(axis=0) - pts.min(axis=0)
    pts = 2 * pts / size.max()
    pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2
    return pts


def read_pcd(fpath):
    return o3d.io.read_point_cloud(fpath)


def write_pcd(fpath, pcd):
    return o3d.io.write_point_cloud(fpath, pcd)


def write_points(fpath, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(fpath, pcd)


def read_mesh(fpath):
    return o3d.io.read_triangle_mesh(fpath)


def write_mesh(fpath, mesh):
    return o3d.io.write_triangle_mesh(fpath, mesh)


def create_mesh_from_volume(volume, gradient_direction="ascent"):
    verts, faces, normals, values = measure.marching_cubes_lewiner(volume,
                                                                   0.0,
                                                                   spacing=(
                                                                       1.0, -1.0, 1.0),
                                                                   gradient_direction=gradient_direction)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)
    return mesh


def compute_pts_vol(model, grid_size, volume_factor, sub_batch_size=512):
    x = np.linspace(-volume_factor, volume_factor,
                    grid_size).astype(np.float32)
    y = np.linspace(-volume_factor, volume_factor,
                    grid_size).astype(np.float32)
    z = np.linspace(-volume_factor, volume_factor,
                    grid_size).astype(np.float32)
    X, Y, Z = np.meshgrid(x, y, z)

    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    pts = np.stack((X, Y, Z), axis=1)
    pts = pts.reshape(-1, sub_batch_size, 3)

    vol = []
    for p in tqdm.tqdm(pts, desc="compute-volume"):
        v = model(nn.NdArray.from_numpy_array(p))
        v = v.data.copy().reshape(-1)
        vol.append(v)
    pts = pts.reshape((-1, 3))
    vol = np.concatenate(vol).reshape(grid_size, grid_size, grid_size)
    return pts, vol


def ref_chamfer_dists(X0, X1):
    """
    Reference for Chamfer distances. This only works small point clouds.
    """
    pwd = cdist(pts_pred, pts_true)

    cd0 = np.mean(np.min(pwd, axis=1))
    cd1 = np.mean(np.min(pwd, axis=0))
    cd = (cd0 + cd1) / 2
    return cd0, cd1, cd


def ref_hausdorff_dists(X0, X1):
    """
    Reference for Hasdorff distances. This only works small point clouds.
    """
    pwd = cdist(pts_pred, pts_true)

    hd0 = np.max(np.min(pwd, axis=1))
    hd1 = np.max(np.min(pwd, axis=0))
    hd = max(hd0, hd1)
    return hd0, hd1, hd


def chamfer_hausdorff_dists_block_wise(X0, X1, sub_batch_size=10000):
    """
    Compute one-sided Chamfer and Hausedorf distances and Chamfer and Hausedorf distances in the block-wise manner.
    """
    def chamfer_hausdorff_oneside_dists(X0, X1):
        b0 = X0.shape[0]
        b1 = X1.shape[0]

        sum_ = 0
        max_ = nn.NdArray.from_numpy_array(np.array(-np.inf))
        n = 0
        for i in tqdm.tqdm(range(0, b0, sub_batch_size), desc="cdist-outer-loop"):
            x0 = nn.NdArray.from_numpy_array(X0[i:i+sub_batch_size])
            norm_x0 = F.sum(x0 ** 2.0, axis=1, keepdims=True)
            min_ = nn.NdArray.from_numpy_array(np.ones(x0.shape[0]) * np.inf)
            for j in tqdm.tqdm(range(0, b1, sub_batch_size), desc="cdist-inner-loop"):
                x1 = nn.NdArray.from_numpy_array(X1[j:j+sub_batch_size])
                # block pwd
                norm_x1 = F.transpose(
                    F.sum(x1 ** 2.0, axis=1, keepdims=True), (1, 0))
                x1_T = F.transpose(x1, (1, 0))
                x01 = F.affine(x0, x1_T)
                bpwd = (norm_x0 + norm_x1 - 2.0 * x01) ** 0.5
                # block min
                min_ = F.minimum2(min_, F.min(bpwd, axis=1))
            # sum/max over cols
            sum_ += F.sum(min_)
            n += bpwd.shape[0]
            max_ = F.maximum2(max_, F.max(min_))
        ocd = sum_.data / n
        ohd = max_.data
        return ocd, ohd
    cd0, hd0 = chamfer_hausdorff_oneside_dists(X0, X1)
    cd1, hd1 = chamfer_hausdorff_oneside_dists(X1, X0)
    cd = (cd0 + cd1) / 2
    hd = max(hd0, hd1)
    return cd0, cd1, cd, hd0, hd1, hd


def chamfer_hausdorff_dists(X0, X1):
    """
    Compute one-sided Chamfer and Hausedorf distances and Chamfer and Hausedorf distances by cKDTree.
    """

    def chamfer_hausdorff_oneside_dists(X0, X1):
        tree = cKDTree(X1)
        dd, ii = tree.query(X0)
        cd = np.mean(dd)
        hd = np.max(dd)
        return cd, hd

    cd0, hd0 = chamfer_hausdorff_oneside_dists(X0, X1)
    cd1, hd1 = chamfer_hausdorff_oneside_dists(X1, X0)
    cd = (cd0 + cd1) / 2.0
    hd = max(hd0, hd1)

    return cd0, cd1, cd, hd0, hd1, hd
