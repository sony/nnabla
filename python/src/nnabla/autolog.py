# Copyright 2023 Sony Group Corporation.
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


import nnabla as nn
import mlflow
import unittest.mock as mock
import time
import os

from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from nnabla.monitor import MonitorImage, MonitorImageTile


nnabla_mlflow_patches = []
original_add_series = getattr(MonitorSeries, 'add')
original_add_time_elapsed = getattr(MonitorTimeElapsed, 'add')
original_add_image = getattr(MonitorImage, 'add')
original_add_image_tile = getattr(MonitorImageTile, 'add')
original_save_parameters = getattr(nn, 'save_parameters')


def _check_interval(index, flush_at, interval):
    return (index - flush_at) >= interval


def _check_interval_image(index, interval):
    return (index + 1) % interval == 0


def autolog(with_save_parameters=False):
    global nnabla_mlflow_patches

    def add_series(self, index, value):
        if _check_interval(index, self.flush_at, self.interval):
            value = sum(self.buf + [value]) / (len(self.buf) + 1)
            mlflow.log_metric(self.name, value, step=index)
        original_add_series(self, index, value)

    def add_time_elapsed(self, index):
        if _check_interval(index, self.flush_at, self.interval):
            now = time.time()
            elapsed = now - self.lap
            mlflow.log_metric(self.name, elapsed, step=index)
        original_add_time_elapsed(self, index)

    def add_image(self, index, value):
        original_add_image(self, index, value)
        if _check_interval_image(index, self.interval):
            image_name_tmpl = '{:06d}-{:03d}.png'
            run_id = mlflow.active_run().info.run_id
            for j in range(min(self.num_images, value.shape[0])):
                image_name = image_name_tmpl.format(index, j)
                local_path = os.path.join(self.save_dir, image_name)
                uri = 'runs_{}/{}'.format(run_id, self.name)
                mlflow.log_artifact(local_path, uri)

    def add_image_tile(self, index, value):
        original_add_image_tile(self, index, value)
        if _check_interval_image(index, self.interval):
            image_name = '{:06d}.png'.format(index)
            local_path = os.path.join(self.save_dir, image_name)
            run_id = mlflow.active_run().info.run_id
            uri = 'runs_{}/{}'.format(run_id, self.name)
            mlflow.log_artifact(local_path, uri)

    def save_parameters(path, params=None):
        original_save_parameters(path, params)
        run_id = mlflow.active_run().info.run_id
        uri = 'runs_{}/{}'.format(run_id, 'parameters')
        mlflow.log_artifact(path, uri)

    nnabla_mlflow_patches = [
        mock.patch.object(MonitorSeries, 'add', add_series),
        mock.patch.object(MonitorTimeElapsed, 'add', add_time_elapsed),
        mock.patch.object(MonitorImage, 'add', add_image),
        mock.patch.object(MonitorImageTile, 'add', add_image_tile),
    ]

    if with_save_parameters:
        patch = mock.patch.object(nn, 'save_parameters', save_parameters)
        nnabla_mlflow_patches.append(patch)

    for patch in nnabla_mlflow_patches:
        patch.__enter__()


def stop_autolog():
    global nnabla_mlflow_patches
    for patch in nnabla_mlflow_patches:
        patch.__exit__(None, None, None)
    nnabla_mlflow_patches = []
