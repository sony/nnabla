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

import pytest
import nnabla.autolog as autolog

from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed, MonitorImageTile, MonitorImage
import numpy as np
import mlflow
import tempfile
import os
import shutil
import platform

if platform.system() == 'Windows':
    prefix_scheme = "file:/"
else:
    prefix_scheme = "file://"


@pytest.fixture(scope="module")
def prepare_dir():
    cwd = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "..", "..", "..")
    test_work_path = tempfile.mkdtemp(prefix='tmp_autolog', dir=cwd)
    prepare_dir = os.path.join(test_work_path, "mlruns")
    os.chdir(test_work_path)

    return prepare_dir


@pytest.mark.parametrize("experiment", ["image-classification-mnist"])
@pytest.mark.parametrize("with_save_parameters", [True, False])
@pytest.mark.parametrize("interval_0, interval_1", [(1, 5), (2, 4), (4, 2), (3, 5)])
def test_autolog(experiment, with_save_parameters, interval_0, interval_1, prepare_dir):
    path = tempfile.mkdtemp()
    mlflow.set_tracking_uri(prefix_scheme + prepare_dir)

    # Create experiment
    assert mlflow.set_experiment(experiment)
    experiment_id = mlflow.get_experiment_by_name(experiment).experiment_id

    # Start autolog
    autolog.autolog(with_save_parameters)

    # Create monitors
    monitor = Monitor(path)
    assert monitor
    ms = MonitorSeries("MS", monitor, interval_0)
    assert ms
    mte = MonitorTimeElapsed("MTE", monitor, interval_1)
    assert mte
    mi = MonitorImage("MI", monitor, interval_0, num_images=2)
    assert mi
    mit = MonitorImageTile("MIT", monitor, interval_0, num_images=4)
    assert mit

    # Main training loop
    with mlflow.start_run(experiment_id=experiment_id):
        for i in range(10):
            ms.add(i, i * 2)
            mte.add(i)
            mi.add(i, np.random.randn(10, 3, 8, 8))
            mit.add(i, np.random.randn(10, 3, 8, 8))

    # Stop autolog
    autolog.stop_autolog()
