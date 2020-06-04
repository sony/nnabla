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

import shutil
import os
import numpy as np
import open3d as o3d
from args import get_args

from functools import partial


def save_as_image(fpath, geo, cpath):
    path, ext = os.path.splitext(args.fpath)
    fpath = "{}.jpeg".format(path)
    cpath = "{}.json".format(path)

    def capture_image(vis):
        ctr = vis.get_view_control()
        vis.capture_screen_image(fpath)
        print("{} was saved.".format(fpath))
        cp = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(cpath, cp)
        print("{} was saved.".format(cpath))
        return False

    def apply_camera_parameters(vis):
        if cpath != "":
            ctr = vis.get_view_control()
            cp = o3d.io.read_pinhole_camera_parameters(cpath)
            ctr.convert_from_pinhole_camera_parameters(cp)

    key_to_callback = {}
    key_to_callback[ord("S")] = capture_image
    key_to_callback[ord("A")] = apply_camera_parameters
    o3d.visualization.draw_geometries_with_key_callbacks(
        [geo], key_to_callback)


def save_as_images(dpath, mesh):
    np.random.seed(412)
    save_as_images.vis = o3d.visualization.Visualizer()
    if not os.path.exists(dpath):
        os.makedirs(dpath)
    else:
        shutil.rmtree(dpath)
        os.makedirs(dpath)
    save_as_images.n = 0
    save_as_images.i = 0
    save_as_images.a = np.random.randint(-15, 15)
    save_as_images.b = np.random.randint(-15, 15)

    def move_forward(vis):
        fpath = os.path.join(dpath, "{:05d}.jpeg".format(save_as_images.n))
        ctr = vis.get_view_control()
        ctr.rotate(save_as_images.a, save_as_images.b)
        vis.capture_screen_image(fpath)
        print("{} is saved.".format(fpath))
        save_as_images.i += 1
        save_as_images.n += 1
        if save_as_images.i % 200 == 0:
            save_as_images.a = np.random.randint(-15, 15)
            save_as_images.b = np.random.randint(-15, 15)
            save_as_images.i = 0
        return False

    vis = save_as_images.vis
    vis.create_window()
    vis.add_geometry(mesh)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


def draw_multiple_visualizations(mesh, rate):
    path, ext = os.path.splitext(args.fpath)

    def capture_image(vis, type):
        ctr = vis.get_view_control()
        fpath = "{}_{}.jpeg".format(path, type)
        vis.capture_screen_image(fpath)
        print("{} was saved.".format(fpath))
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord("S"), partial(capture_image, type="pcd"))
    vis.create_window(window_name='TopLeft', width=960,
                      height=540, left=0, top=0)
    n = len(mesh.vertices)
    pcd = mesh.sample_points_poisson_disk(int(n * rate))
    print("{} / {} points were sampled.".format(len(pcd.points), n))
    vis.add_geometry(pcd)

    vis2 = o3d.visualization.VisualizerWithKeyCallback()
    vis2.register_key_callback(ord("S"), partial(capture_image, type="mesh"))
    vis2.create_window(window_name='TopRight', width=960,
                       height=540, left=960, top=0)
    vis2.add_geometry(mesh)

    ctr = vis.get_view_control()
    ctr2 = vis2.get_view_control()

    while True:
        cp = ctr.convert_to_pinhole_camera_parameters()
        vis.update_geometry(pcd)
        if not vis.poll_events():
            break
        vis.update_renderer()

        ctr2.convert_from_pinhole_camera_parameters(cp)
        vis2.update_geometry(mesh)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

    vis.destroy_window()
    vis2.destroy_window()


def main(args):
    if args.command == "mesh":
        mesh = o3d.io.read_triangle_mesh(args.fpath)
        mesh.compute_vertex_normals()
        save_as_image(args.fpath, mesh, args.cpath)
    elif args.command == "pcd":
        pcd = o3d.io.read_point_cloud(args.fpath)
        save_as_image(args.fpath, pcd, args.cpath)
    elif args.command == "images":
        mesh = o3d.io.read_triangle_mesh(args.fpath)
        mesh.compute_vertex_normals()
        dpath, ext = os.path.splitext(args.fpath)
        save_as_images(dpath, mesh)
    elif args.command == "pcdmesh":
        mesh = o3d.io.read_triangle_mesh(args.fpath)
        mesh.compute_vertex_normals()
        draw_multiple_visualizations(mesh, args.rate)
    else:
        raise ValueError("Command={} is not supported".format(args.command))


if __name__ == '__main__':
    args = get_args()
    main(args)
