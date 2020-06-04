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

import numpy as np

import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as M
import nnabla.functions as F
import nnabla.parametric_functions as PF


def get_args(mode="train"):
    import argparse
    import os
    parser = argparse.ArgumentParser(
        description='''Implicit Geometric Regularization.
        ''')
    parser.add_argument("-f", "--fpath", type=str,
                        help='File path to a mesh or point cloud data.')
    parser.add_argument("-b", "--batch-size", type=int, default=512,
                        help='Batch size.')
    parser.add_argument("-sb", "--sub-batch-size", type=int, default=512,
                        help='Sub batch size for computing the pair-wise distance.')
    parser.add_argument("-d", "--device-id", type=str, default="0",
                        help='Device ID.')
    parser.add_argument("-k", "--knn", type=int, default=50,
                        help='K-Nearest Neighbors.')
    parser.add_argument("-vf", "--volume-factor", type=float, default=1.5,
                        help='Volume factors to a cubic space [-1, 1]^3.')
    parser.add_argument("-tr", "--test-rate", type=float, default=0.25,
                        help='Test sample rate for the entire dataset.')
    parser.add_argument("-te", "--train-epoch", type=int, default=1500,
                        help='Training Epoch.')
    parser.add_argument("-vi", "--valid-iter", type=int, default=100,
                        help='Validation Epoch.')
    parser.add_argument("-sie", "--save-interval-epoch", type=int, default=100,
                        help='Save interval for epoch. Model is saved at every this epochs.')
    parser.add_argument("-m", "--monitor-path", type=str, default="tmp.monitor",
                        help='Monitor path.')
    parser.add_argument("-mlp", "--model-load-path", type=str, default="",
                        help='MOdel load path.')
    parser.add_argument("-mdp", "--mesh-data-path", type=str, default="",
                        help='Mesh data path.')
    parser.add_argument("--lam", type=float, default=0.1,
                        help='Coefficient to the impcilit geometric regulartization.')
    parser.add_argument("--tau", type=float, default=1.0,
                        help='Coefficient to the normals loss.')
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument("--beta0", type=float, default=0.9,
                        help='First-order momentum of Adam.')
    parser.add_argument("--beta1", type=float, default=0.999,
                        help='Second-order momentum of Adam.')
    parser.add_argument("--dims", type=int, default=512,
                        help='Dimensions of MLP.')
    parser.add_argument("--ldims", type=int, default=0,
                        help='Latent dimensions.')
    parser.add_argument("--grid-size", type=int, default=256,
                        help='Grid size for computing the volume, SDF is computed for the grid-size^3'
                        'The grid is first normalized in [-1, 1], then the volume factor is multiplied.')
    parser.add_argument('--mesh-colors', type=float, nargs='+',
                        help='Mesh colors')
    parser.add_argument('--with-normals', action="store_true",
                        help='Use the normals loss.')
    parser.add_argument('-gd', '--gradient-direction', type=str, default="ascent",
                        choices=["ascent", "descent"],
                        help='Gradient direction for the marching_cubes_lewiner.')

    # Subcommands
    subparsers = parser.add_subparsers(dest="command")
    parser_mesh = subparsers.add_parser("mesh",
                                        help='')
    parser_mesh.add_argument("-f", "--fpath", type=str,
                             help='File path to a mesh data.')
    parser_mesh.add_argument("-c", "--cpath", type=str,
                             help='Path to a camera parameters.')
    parser_pcd = subparsers.add_parser("pcd",
                                       help='')
    parser_pcd.add_argument("-f", "--fpath", type=str,
                            help='File path to a point cloud data.')
    parser_pcd.add_argument("-c", "--cpath", type=str,
                            help='Path to a camera parameters.')
    parser_images = subparsers.add_parser("images",
                                          help='Create a video based on the mesh file in the same directory.')
    parser_images.add_argument("-f", "--fpath", type=str,
                               help='File path to a mesh data.')
    parser_pcdmesh = subparsers.add_parser("pcdmesh",
                                           help='Synchronized two views of a pcd sampled from a mesh and the mesh.')
    parser_pcdmesh.add_argument("-f", "--fpath", type=str,
                                help='File path to a mesh data.')
    parser_pcdmesh.add_argument(
        "-r", "--rate", help="Sampling rate.", type=float, default=0.1)

    parser_create = subparsers.add_parser("create",
                                          help='Create dataset.')
    parser_create.add_argument("-f", "--mesh-data-path", type=str,
                               help='File path to a mesh data.')

    args = parser.parse_args()

    if not os.path.exists(args.monitor_path):
        os.makedirs(args.monitor_path)
    return args
