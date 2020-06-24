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

import tqdm
import os
import numpy as np

import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as M
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context

from args import get_args
from models import MLP
from datasets import point_cloud_data_source, point_cloud_data_iterator

import utils


def points_loss(y):
    return F.sum(y ** 2.0).apply(persistent=True)


def normals_loss(y, x, n, with_normals):
    loss_normals = 0.0
    if args.with_normals:
        grads = nn.grad([y], [x])
        g = grads[0]
        loss_normals = F.sum((g - n) ** 2.0, axis=[1]) ** 0.5
        loss_normals = F.sum(loss_normals).apply(persistent=True)
    return loss_normals


def eikonal_reg(x, r, model, volume_factor):
    # volumetric space and neighbor points
    xu = (2 * args.volume_factor) * F.rand(shape=x.shape) - args.volume_factor
    xn = x + r * F.randn(shape=x.shape)
    xa = F.concatenate(*[xu, xn], axis=0).apply(need_grad=True)
    ya = model(xa)
    grads = nn.grad([ya], [xa])
    norms = [F.sum(g ** 2.0, axis=[1]) ** 0.5 for g in grads]
    gp = sum([F.mean((n - 1.0) ** 2.0) for n in norms])
    return gp.apply(persistent=True)


def feed_data(inputs, di, with_normals):
    x, n, r = inputs
    pts, nrm, rad = di.next()
    x.d = pts
    r.d = rad
    if with_normals:
        n.d = nrm


def evaluate(model, pts_true, grid_size, volume_factor, monitor_distances,
             i, save_interval_epoch=1):
    if i % save_interval_epoch != 0:
        return
    pts, vol = utils.compute_pts_vol(model, grid_size, volume_factor)
    mesh = utils.create_mesh_from_volume(vol)
    pcd = mesh.sample_points_poisson_disk(len(pts_true), seed=412)
    pts_pred = np.asarray(pcd.points)
    pts_pred = utils.normalize(pts_pred)
    # Pair-wise distance
    cd0, cd1, cd, hd0, hd1, hd = utils.chamfer_hausdorff_dists(
        pts_pred, pts_true)
    for m, d in zip(monitor_distances, [cd0, cd1, cd, hd0, hd1, hd]):
        m.add(i, d)


def save_params(monitor_path, i, save_interval_epoch=1):
    if i % save_interval_epoch != 0:
        return
    nn.save_parameters(os.path.join(monitor_path, "param_{:05d}.h5".format(i)))


def main(args):
    # Context
    ctx = get_extension_context("cudnn", device_id=args.device_id)
    nn.set_default_context(ctx)

    # Network
    model = MLP(args.dims, args.ldims)
    # points loss
    x = nn.Variable([args.batch_size, 3]).apply(need_grad=True)
    y = model(x)
    loss_points = points_loss(y)
    # normals loss
    n = nn.Variable([args.batch_size, 3])
    loss_normals = normals_loss(y, x, n, args.with_normals)
    # eikonal regularization
    r = nn.Variable([args.batch_size, 1])
    reg_eikonal = eikonal_reg(x, r, model, args.volume_factor)
    # total loss
    loss = loss_points + args.tau * loss_normals + args.lam * reg_eikonal

    # Dataset (input is normalized in [-1, 1])
    ds = point_cloud_data_source(
        args.fpath, args.knn, args.test_rate, test=False)
    di = point_cloud_data_iterator(ds, args.batch_size)
    ds = point_cloud_data_source(args.fpath, -1, test=True)
    pts_true = ds.points

    # Solver
    solver = S.Adam(args.learning_rate, args.beta0, args.beta1)
    solver.set_parameters(nn.get_parameters())

    # Monitor
    monitor = M.Monitor(args.monitor_path)
    monitor_time = M.MonitorTimeElapsed(
        "Training time", monitor, interval=100, verbose=False)
    monitor_points_loss = M.MonitorSeries(
        "Points loss", monitor, interval=100, verbose=False)
    monitor_normals_loss = M.MonitorSeries(
        "Normals loss", monitor, interval=100, verbose=False)
    monitor_eikonal_reg = M.MonitorSeries(
        "Eikonal reg", monitor, interval=100, verbose=False)
    monitor_total_loss = M.MonitorSeries(
        "Total loss", monitor, interval=100, verbose=False)

    monitor_distances = [
                         M.MonitorSeries("Chamfer Pred True",
                                         monitor, interval=1, verbose=False),
                         M.MonitorSeries("Chamfer True Pred",
                                         monitor, interval=1, verbose=False),
                         M.MonitorSeries("Chamfer", monitor,
                                         interval=1, verbose=False),
                         M.MonitorSeries("Hausdorff Pred True",
                                         monitor, interval=1, verbose=False),
                         M.MonitorSeries("Hausdorff True Pred",
                                         monitor, interval=1, verbose=False),
                         M.MonitorSeries("Hausdorff", monitor, interval=1, verbose=False)]

    # Train
    iter_per_epoch = di.size // args.batch_size
    for i in tqdm.tqdm(range(args.train_epoch), desc="train-loop"):
        # evaluate
        evaluate(model, pts_true, args.grid_size, args.volume_factor, monitor_distances,
                 i, args.save_interval_epoch)
        for j in range(iter_per_epoch):
            # feed data
            feed_data([x, n, r], di, args.with_normals)
            # zero_grad, forward, backward, update
            solver.zero_grad()
            loss.forward()
            loss.backward(clear_buffer=True)
            solver.update()
            # monitor
            monitor_points_loss.add(i * iter_per_epoch + j, loss_points.d)
            monitor_normals_loss.add(
                i * iter_per_epoch + j, loss_normals.d) if args.with_normals else None
            monitor_eikonal_reg.add(i * iter_per_epoch + j, reg_eikonal.d)
            monitor_total_loss.add(i * iter_per_epoch + j, loss.d)
            monitor_time.add(i)
        # save
        save_params(args.monitor_path, i, args.save_interval_epoch)
    save_params(args.monitor_path, args.train_epoch)
    evaluate(model, pts_true, args.grid_size, args.volume_factor, monitor_distances,
             args.train_epoch)


if __name__ == '__main__':
    args = get_args()
    utils.save_args(args)
    main(args)
