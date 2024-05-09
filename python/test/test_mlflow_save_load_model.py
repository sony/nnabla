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

import os
import json
import nnabla as nn
import nnabla.mlflow_save_load_model
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.monitor import tile_images
from nnabla.utils.data_iterator import data_iterator_simple
import nnabla.utils.save as save

import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelSignature
from mlflow.utils.file_utils import TempDir
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_additional_pip_env, _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

import pytest
import tempfile
import platform
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# Scikit
try:
    from sklearn.datasets import load_digits  # Only for dataset
except ImportError:
    print("Require scikit-learn", file=sys.stderr)
    raise

# Matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Require matplotlib", file=sys.stderr)
    raise

from sklearn.datasets import load_digits

np.random.seed(0)
imshow_opt = dict(cmap='gray', interpolation='nearest')

PYTEST_MODEL_PATH = "pytest_model"

if platform.system() == 'Windows':
    prefix_scheme = "file:/"
else:
    prefix_scheme = "file://"


def save_nnp(input, output, batchsize):
    runtime_contents = {
        'networks': [
            {'name': 'Validation',
             'batch_size': batchsize,
             'outputs': output,
             'names': input}],
        'executors': [
            {'name': 'Runtime',
             'network': 'Validation',
             'data': [k for k, _ in input.items()],
             'output': [k for k, _ in output.items()]}]}
    return runtime_contents


def save_checkpoint(path, current_iter, solvers):
    if isinstance(solvers, nn.solver.Solver):
        solvers = {"": solvers}

    assert isinstance(solvers, dict), \
        "`solvers` must be either Solver object or dict of { `name`: Solver }."

    checkpoint_info = dict()
    save_paths = []

    for solvername, solver_obj in solvers.items():
        prefix = "{}_".format(solvername.replace(
            "/", "_")) if solvername else ""
        partial_info = dict()

        # save solver states.
        states_fname = prefix + 'states_{}.h5'.format(current_iter)
        states_path = os.path.join(path, states_fname)
        solver_obj.save_states(states_path)
        save_paths.append(states_path)
        # save relative path to support moving a saved directory
        partial_info["states_path"] = states_fname

        # save registered parameters' name. (just in case)
        params_names = [k for k in solver_obj.get_parameters().keys()]
        partial_info["params_names"] = params_names

        # save the number of solver update.
        num_update = getattr(solver_obj.get_states()[params_names[0]], "t")
        partial_info["num_update"] = num_update

        checkpoint_info[solvername] = partial_info

    params_fname = 'params_{}.h5'.format(current_iter)
    params_path = os.path.join(path, params_fname)
    nn.parameter.save_parameters(params_path)
    save_paths.append(params_path)

    # save relative path so to support moving a saved directory
    checkpoint_info["params_path"] = params_fname

    checkpoint_info["current_iter"] = current_iter

    # save checkpoint
    checkpoint_fname = 'checkpoint_{}.json'.format(current_iter)
    filename = os.path.join(path, checkpoint_fname)

    with open(filename, 'w') as f:
        json.dump(checkpoint_info, f)

    save_paths.append(filename)


@pytest.fixture(scope="module")
def data():
    # Preparing a Toy Dataset
    digits = load_digits(n_class=10)
    print("Num images:", digits.images.shape[0])
    print("Image shape:", digits.images.shape[1:])
    print("Labels:", digits.target[:10])
    plt.imshow(tile_images(digits.images[:64, None]), **imshow_opt)

    def load_func(index):
        """Loading an image and its label"""
        img = digits.images[index]
        label = digits.target[index]
        return img[None], np.array([label]).astype(np.int32)

    data = data_iterator_simple(
        load_func, digits.target.shape[0], 64, False, None, with_file_cache=False)
    return data


@pytest.fixture(scope="module")
def model(data):
    img, label = data.next()
    plt.imshow(tile_images(img), **imshow_opt)
    print("labels: {}".format(label.reshape(8, 8)))
    print("Label shape: {}".format(label.shape))

    # Preparing the Computation Graph
    x = nn.Variable(img.shape)  # Define an image variable
    with nn.parameter_scope("affine1"):
        y = PF.affine(x, 10)  # Output is 10 class
    # Building a loss graph
    t = nn.Variable(label.shape)  # Define an target variable
    # Softmax Xentropy fits multi-class classification problems
    loss = F.mean(F.softmax_cross_entropy(y, t))

    model_save_path = PYTEST_MODEL_PATH

    cwd = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "..", "..", "..")
    test_work_path = tempfile.mkdtemp(prefix='tmp_save_load', dir=cwd)
    mlflow_workdir = os.path.join(test_work_path, "mlruns")
    os.chdir(test_work_path)
    assert mlflow.set_experiment("save_load_model")

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
    model = {}
    checkpoint_list = []
    # Training
    learning_rate = 1e-3
    solver = S.Sgd(learning_rate)
    # Set parameter variables to be updated.
    solver.set_parameters(nn.get_parameters())
    for i in range(1000):
        x.d, t.d = data.next()
        loss.forward()
        solver.zero_grad()  # Initialize gradients of all parameters to zero.
        loss.backward()
        solver.weight_decay(1e-5)  # Applying weight decay as an regularization
        solver.update()
        if i % 100 == 0:  # Print for each 10 iterations
            # save checkpoint
            save_checkpoint(model_save_path, i, solver)
            if os.path.isfile(os.path.join(model_save_path, 'checkpoint_{}.json'.format(i))):
                checkpoint_list.append(os.path.join(
                    model_save_path, 'checkpoint_{}.json'.format(i)))
            if os.path.isfile(os.path.join(model_save_path, 'params_{}.h5'.format(i))):
                checkpoint_list.append(os.path.join(
                    model_save_path, 'params_{}.h5'.format(i)))
            if os.path.isfile(os.path.join(model_save_path, 'states_{}.h5'.format(i))):
                checkpoint_list.append(os.path.join(
                    model_save_path, 'states_{}.h5'.format(i)))

    # save parameters
    parameter_file = os.path.join(model_save_path, 'pytest_params.h5')
    nn.save_parameters(parameter_file)

    # save nnp
    contents = save_nnp({'x': x}, {'y': loss}, 64)
    save.save(os.path.join(model_save_path, 'pytest_result.nnp'), contents)

    model['model'] = parameter_file
    model['nnp'] = os.path.join(model_save_path, 'pytest_result.nnp')
    model['checkpoint'] = checkpoint_list
    model['work_dir'] = mlflow_workdir
    return model


@pytest.fixture
def create_extra_files(tmp_path):
    fp1 = tmp_path.joinpath("extra1.txt")
    fp2 = tmp_path.joinpath("extra2.txt")
    fp1.write_text("1")
    fp2.write_text("2")
    return [str(fp1), str(fp2)], ["1", "2"]


@pytest.fixture
def create_config_file(tmp_path):
    fp = tmp_path.joinpath("config.txt")
    fp.write_text("batch_size: 64")
    return str(fp), "batch_size: 64"


@pytest.fixture
def nnabla_custom_env(tmp_path):
    conda_env = os.path.join(tmp_path, "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["nnabla", "pytest"])
    return conda_env


def test_model(model, data):
    assert model != {}
    assert model['model'] != None
    assert model['nnp'] != None
    assert model['checkpoint'] != None


def test_log_model(model, data):
    mlflow.set_tracking_uri(prefix_scheme + model['work_dir'])
    experiment_id = mlflow.get_experiment_by_name(
        "save_load_model").experiment_id
    try:
        with mlflow.start_run(run_name='nnabla.mlflow_save_load_model_log_model_test', experiment_id=experiment_id):
            nnabla_model = os.path.basename(model['model'])
            model_path = os.path.dirname(model['model'])
            model_info = nnabla.mlflow_save_load_model.log_model(nnabla_model,
                                                                 model_path,
                                                                 nnp=[
                                                                     model['nnp']],
                                                                 checkpoint=model['checkpoint'])
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_path}"
            assert model_info.model_uri == model_uri
    finally:
        mlflow.end_run()


def test_log_model_with_extra_files(model, create_extra_files):
    extra_files, contents_expected = create_extra_files
    mlflow.set_tracking_uri(prefix_scheme + model['work_dir'])
    experiment_id = mlflow.get_experiment_by_name(
        "save_load_model").experiment_id
    with mlflow.start_run(experiment_id=experiment_id):
        model_path = os.path.dirname(model['model'])
        nnabla.mlflow_save_load_model.log_model(os.path.basename(model['model']),
                                                os.path.dirname(
                                                    model['model']),
                                                extra_files=extra_files)

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_path}"

    with TempDir(remove_on_exit=True) as tmp:
        model_path = _download_artifact_from_uri(model_uri, tmp.path())
        model_config_path = os.path.join(model_path, "MLmodel")
        model_config = Model.load(model_config_path)
        flavor_config = model_config.flavors["nnabla"]

        assert "extra_files" in flavor_config
        loaded_extra_files = flavor_config["extra_files"]

        for loaded_extra_file, content_expected in zip(loaded_extra_files, contents_expected):
            assert "path" in loaded_extra_file
            extra_file_path = os.path.join(
                model_path, loaded_extra_file["path"])
            with open(extra_file_path) as fp:
                assert fp.read() == content_expected


def test_log_model_with_config_file(model, create_config_file):
    config_file, content_expected = create_config_file
    mlflow.set_tracking_uri(prefix_scheme + model['work_dir'])
    experiment_id = mlflow.get_experiment_by_name(
        "save_load_model").experiment_id
    with mlflow.start_run(experiment_id=experiment_id):
        model_path = os.path.dirname(model['model'])
        nnabla.mlflow_save_load_model.log_model(os.path.basename(model['model']),
                                                os.path.dirname(
                                                    model['model']),
                                                config_file=config_file)

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_path}"

    model_path = _download_artifact_from_uri(model_uri)
    config_file_path = os.path.join(model_path, "config.txt")
    with open(config_file_path) as fp:
        assert fp.read() == content_expected


def test_log_model_with_conda_env(model, nnabla_custom_env):
    mlflow.set_tracking_uri(prefix_scheme + model['work_dir'])
    experiment_id = mlflow.get_experiment_by_name(
        "save_load_model").experiment_id
    with mlflow.start_run(experiment_id=experiment_id):
        model_path = os.path.dirname(model['model'])
        nnabla.mlflow_save_load_model.log_model(os.path.basename(model['model']),
                                                os.path.dirname(
                                                    model['model']),
                                                conda_env=nnabla_custom_env,
                                                )
        model_uri = _download_artifact_from_uri(
            f"runs:/{mlflow.active_run().info.run_id}/{model_path}"
        )

    pyfunc_conf = _get_flavor_configuration(
        model_path=model_uri, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(
        model_uri, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != nnabla_custom_env

    with open(nnabla_custom_env) as f:
        nnabla_custom_env_text = f.read()
    with open(saved_conda_env_path) as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == nnabla_custom_env_text


@pytest.mark.parametrize('kwargs', [None, 'model', 'MLmodel', 'nnp', 'checkpoint'])
def test_load_model_with_kwargs(kwargs, model, data):
    mlflow.set_tracking_uri(prefix_scheme + model['work_dir'])
    experiment_id = mlflow.get_experiment_by_name(
        "save_load_model").experiment_id
    run_name = "nnabla.mlflow_save_load_model_log_model_test_{}".format(
        kwargs).rstrip("_None")
    nnp = None
    checkpoint = None

    if kwargs == 'nnp':
        nnp = [model['nnp']]

    if kwargs == 'checkpoint':
        checkpoint = model['checkpoint']

    with mlflow.start_run(run_name='nnabla.mlflow_save_load_model_log_model_test', experiment_id=experiment_id):
        nnabla_model = os.path.basename(model['model'])
        model_path = os.path.dirname(model['model'])
        model_info = nnabla.mlflow_save_load_model.log_model(nnabla_model,
                                                             model_path,
                                                             nnp=nnp,
                                                             checkpoint=checkpoint)
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_path}"

    loaded_model = nnabla.mlflow_save_load_model.load_model(
        model_uri, model=kwargs)
    if kwargs == None or kwargs == 'model':
        if kwargs == None:
            loaded_model = nnabla.mlflow_save_load_model.load_model(model_uri)
        model_path = model['model']
        assert loaded_model != model_path
        with open(loaded_model, encoding="utf8", errors='ignore') as f:
            loaded_model_text = f.read()
        with open(model_path, encoding="utf8", errors='ignore') as f:
            model_text = f.read()
        assert len(model_text) >= 1
        assert loaded_model_text == model_text
    elif kwargs == 'MLmodel':
        assert 'model' in loaded_model
    elif kwargs == 'nnp' or kwargs == 'checkpoint':
        for model in loaded_model:
            model_path = os.path.join(
                PYTEST_MODEL_PATH, os.path.basename(model))
            assert model != model_path
            with open(model, encoding="utf8", errors='ignore') as f:
                loaded_model_text = f.read()
            with open(model_path, encoding="utf8", errors='ignore') as f:
                model_text = f.read()
            assert len(model_text) >= 1
            assert loaded_model_text == model_text
    else:
        raise ValueError


def test_load_model_with_run_name(model, data):
    mlflow.set_tracking_uri(prefix_scheme + model['work_dir'])
    experiment_id = mlflow.get_experiment_by_name(
        "save_load_model").experiment_id
    with mlflow.start_run(run_name='test_load_model_with_run_name', experiment_id=experiment_id):
        nnabla_model = os.path.basename(model['model'])
        model_path = os.path.dirname(model['model'])
        model_info = nnabla.mlflow_save_load_model.log_model(nnabla_model,
                                                             model_path,
                                                             nnp=[
                                                                 model['nnp']],
                                                             checkpoint=model['checkpoint'])
        run_id = mlflow.active_run().info.run_id

    get_run_id = nnabla.mlflow_save_load_model.get_runid_by_name(
        "test_load_model_with_run_name")
    assert get_run_id == run_id

    model_uri = f"runs:/{get_run_id}/{model_path}"
    loaded_model = nnabla.mlflow_save_load_model.load_model(model_uri)
    model_path = model['model']
    assert loaded_model != model_path
    with open(loaded_model, encoding="utf8", errors='ignore') as f:
        loaded_model_text = f.read()
    with open(model_path, encoding="utf8", errors='ignore') as f:
        model_text = f.read()
    assert len(model_text) >= 1
    assert loaded_model_text == model_text


def test_load_as_pyfunc_model(model, data):
    mlflow.set_tracking_uri(prefix_scheme + model['work_dir'])
    experiment_id = mlflow.get_experiment_by_name(
        "save_load_model").experiment_id
    with mlflow.start_run(run_name='test_load_model_with_run_name', experiment_id=experiment_id):
        nnabla_model = os.path.basename(model['model'])
        model_path = os.path.dirname(model['model'])
        model_info = nnabla.mlflow_save_load_model.log_model(nnabla_model,
                                                             model_path,
                                                             nnp=[
                                                                 model['nnp']],
                                                             checkpoint=model['checkpoint'])
        run_id = mlflow.active_run().info.run_id

    get_run_id = nnabla.mlflow_save_load_model.get_runid_by_name(
        "test_load_model_with_run_name")
    assert get_run_id == run_id

    model_uri = f"runs:/{get_run_id}/{model_path}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    try:
        loaded_model.predict(None)
    except:
        raise AttributeError
