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

"""
The ``mlflow.nnabla`` module provides an API for logging and loading nnabla models.
This module exports nnabla models with the following flavors:

Nnabla (native) format
    This is the main flavor that can be loaded back into nnabla.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""
import atexit
import logging
import os
import posixpath
import re
import shutil
import json
import unittest.mock as mock
from typing import Any, Dict, NamedTuple, Optional

import numpy as np
import yaml
from packaging.version import Version

import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.types.schema import TensorSpec
from mlflow.utils import is_iterator
from mlflow import MlflowClient
from mlflow.utils.autologging_utils import (
    PatchFunction,
    autologging_integration,
    batch_metrics_logger,
    get_autologging_config,
    log_fn_args_as_params,
    picklable_exception_safe_function,
    resolve_input_example_and_signature,
    safe_patch,
)
from mlflow.utils.autologging_utils.metrics_queue import (
    add_to_metrics_queue,
    flush_metrics_queue,
)
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import (
    TempDir,
    write_to,
)
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.time import get_current_time_millis

FLAVOR_NAME = "nnabla"
_EXTRA_FILES_KEY = "extra_files"
_REQUIREMENTS_FILE_KEY = "requirements_file"
_NNP_FILES_KEY = "nnp_files"
_CHECKPOINT_FILES_KEY = "checkpoint_files"
_LOADER_MODULE = "nnabla.mlflow_save_load_model"

_logger = logging.getLogger(__name__)
logging.getLogger("mlflow").setLevel(logging.DEBUG)


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("nnabla")]

    return pip_deps


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    nnabla_model_name,
    path,
    nnp=None,
    checkpoint=None,
    config_file=None,
    extra_files=None,
    conda_env=None,
    code_paths=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Log a nnabla model in MLflow Model format.

    :param nnabla_model_name: The nnabla model name to be saved.
    :param path: Local path where the model is to be saved.
    :param nnp: A list containing the paths to corresponding .nnp file.
    :param checkpoint: A list containing the paths to checkpoint.
    :param config_file: the path to corresponding config file.
    :param extra_files: A list containing the paths to corresponding extra files. Remote URIs
                      are resolved to absolute filesystem paths.
                      For example, consider the following ``extra_files`` list -

                      extra_files = ["s3://my-bucket/path/to/my_file1",
                                    "s3://my-bucket/path/to/my_file2"]

                      In this case, the ``"my_file1 & my_file2"`` extra file is downloaded from S3.

                      If ``None``, no extra files are added to the model.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param signature: {{ signature }}
    :param input_example: {{ input_example }}
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """

    import nnabla
    import nnabla.mlflow_save_load_model
    return Model.log(
        artifact_path=path,
        flavor=nnabla.mlflow_save_load_model,
        nnabla_model_name=nnabla_model_name,
        nnp=nnp,
        checkpoint=checkpoint,
        config_file=config_file,
        extra_files=extra_files,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
    )


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    nnabla_model_name,
    path,
    nnp=None,
    checkpoint=None,
    config_file=None,
    extra_files=None,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Log a nnabla model in MLflow Model format.

    :param nnabla_model_name: The nnabla model name to be saved.
    :param path: Local path where the model is to be saved.
    :param nnp: A list containing the paths to corresponding .nnp file.
    :param checkpoint: A list containing the paths to checkpoint.
    :param config_file: the path to corresponding config file.
    :param extra_files: A list containing the paths to corresponding extra files. Remote URIs
                      are resolved to absolute filesystem paths.
                      For example, consider the following ``extra_files`` list -

                      extra_files = ["s3://my-bucket/path/to/my_file1",
                                    "s3://my-bucket/path/to/my_file2"]

                      In this case, the ``"my_file1 & my_file2"`` extra file is downloaded from S3.

                      If ``None``, no extra files are added to the model.
    :param mlflow_model: MLflow model configuration to which to add the ``nnabla`` flavor.
    :param signature: {{ signature }}
    :param input_example: {{ input_example }}
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    """

    import nnabla
    _validate_env_arguments(conda_env, pip_requirements,
                            extra_pip_requirements)
    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)

    if signature is None and input_example is not None:
        wrapped_model = _NnablaModelWrapper(nnabla_model)
        signature = _infer_signature_from_input_example(
            input_example, wrapped_model)
    elif signature is False:
        signature = None

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    # save code
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    # save model
    model_data_subpath = "model"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(model_data_path)
    nnabla_model = os.path.join(mlflow_model.artifact_path, nnabla_model_name)
    shutil.copy(nnabla_model, model_data_path)

    # save config_file
    if config_file:
        shutil.copy(config_file, path)

    nnabla_artifacts_config = {}
    # save checkpoint
    if checkpoint:
        nnabla_artifacts_config[_CHECKPOINT_FILES_KEY] = []
        if not isinstance(checkpoint, list):
            raise TypeError("checkpoint files argument should be a list")

        with TempDir() as tmp_checkpoint_files_dir:
            for checkpoint_file in checkpoint:
                _download_artifact_from_uri(
                    artifact_uri=checkpoint_file, output_path=tmp_checkpoint_files_dir.path()
                )
                rel_path = posixpath.join(
                    _CHECKPOINT_FILES_KEY, os.path.basename(checkpoint_file))
                nnabla_artifacts_config[_CHECKPOINT_FILES_KEY].append(
                    {"path": rel_path})
            shutil.move(
                tmp_checkpoint_files_dir.path(),
                posixpath.join(path, _CHECKPOINT_FILES_KEY),
            )

    # save nnp
    if nnp:
        nnabla_artifacts_config[_NNP_FILES_KEY] = []
        if not isinstance(nnp, list):
            raise TypeError("nnp files argument should be a list")

        with TempDir() as tmp_nnp_files_dir:
            for nnp_file in nnp:
                _download_artifact_from_uri(
                    artifact_uri=nnp_file, output_path=tmp_nnp_files_dir.path()
                )
                rel_path = posixpath.join(
                    _NNP_FILES_KEY, os.path.basename(nnp_file))
                nnabla_artifacts_config[_NNP_FILES_KEY].append(
                    {"path": rel_path})
            shutil.move(
                tmp_nnp_files_dir.path(),
                posixpath.join(path, _NNP_FILES_KEY),
            )
    # save extra_files
    if extra_files:
        nnabla_artifacts_config[_EXTRA_FILES_KEY] = []
        if not isinstance(extra_files, list):
            raise TypeError("Extra files argument should be a list")

        with TempDir() as tmp_extra_files_dir:
            for extra_file in extra_files:
                _download_artifact_from_uri(
                    artifact_uri=extra_file, output_path=tmp_extra_files_dir.path()
                )
                rel_path = posixpath.join(
                    _EXTRA_FILES_KEY, os.path.basename(extra_file))
                nnabla_artifacts_config[_EXTRA_FILES_KEY].append(
                    {"path": rel_path})
            shutil.move(
                tmp_extra_files_dir.path(),
                posixpath.join(path, _EXTRA_FILES_KEY),
            )

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        model_data=model_data_subpath,
        nnabla_version=str(nnabla.__version__),
        code=code_dir_subpath,
        **nnabla_artifacts_config,
    )
    pyfunc.add_to_model(
        mlflow_model,
        loader_module=_LOADER_MODULE,
        data=model_data_subpath,
        code=code_dir_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                model_data_path,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(
            conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME),
                 "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME),
             "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def _load_model(path, **kwargs):
    if os.path.isdir(path):
        if os.path.basename(path.rstrip("/")) in [_CHECKPOINT_FILES_KEY, _NNP_FILES_KEY]:
            models = []
            models_list = os.listdir(path)
            for model in models_list:
                models.append(os.path.join(path, model))
            return models
        else:
            model = os.path.join(path, os.listdir(path)[0])
            return model

    _logger.error(" No such path %s.", path)
    return None


def load_model(model_uri, dst_path=None, **kwargs):
    """
    Load an MLflow model that contains the nnabla flavor from the specified path.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :return: a nnabla model.
    """
    local_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path)
    nnabla_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, nnabla_conf)
    if "model" in kwargs:
        if kwargs['model'] == 'MLmodel':
            return local_model_path
        elif kwargs['model'] == 'model':
            model_path = os.path.join(
                local_model_path, nnabla_conf["model_data"])
        elif kwargs['model'] == 'nnp':
            model_path = os.path.join(local_model_path, _NNP_FILES_KEY)
        elif kwargs['model'] == 'checkpoint':
            model_path = os.path.join(local_model_path, _CHECKPOINT_FILES_KEY)
        else:
            _logger.error(" error parameter %s.", )
            return None
    else:
        # default to nnabla .h5 model
        model_path = os.path.join(local_model_path, nnabla_conf["model_data"])

    return _load_model(path=model_path, **kwargs)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``. This function loads an MLflow
    model with the TensorFlow flavor into a new nnabla graph and exposes it behind the
    ``pyfunc.predict`` interface.

    :param path: Local filesystem path to the MLflow Model with the ``nnabla`` flavor.
    """
    return _NnablaModelWrapper(_load_model(path, model='model'))


class _NnablaModelWrapper:
    def __init__(self, nnabla_model):
        self.nnabla_model = nnabla_model

    def predict(
        self, data, params: Optional[Dict[str, Any]] = None  # pylint: disable=unused-argument
    ):
        """
        :param data: Model input data.
        :param params: Additional parameters to pass to the model for inference.

                       .. Note:: Experimental: This parameter may change or be removed in a future
                                               release without warning.

        :return: Model predictions.
        """
        print("modle: %s" % self.nnabla_model)
        # To Do


def _get_all_run_names():
    run_names = []

    client = MlflowClient()
    runs = client.search_runs(_get_experiment_id())
    for run in runs:
        run_names.append(run.__dict__['_data'].tags['mlflow.runName'])

    return run_names


original_start_run = getattr(mlflow, 'start_run')


def unique_run_name():
    def new_start_run(
            run_id: Optional[str] = None,
            experiment_id: Optional[str] = None,
            run_name: Optional[str] = None,
            nested: bool = False,
            tags: Optional[Dict[str, Any]] = None,
            description: Optional[str] = None,
            log_system_metrics: Optional[bool] = None,
            ):
        if run_name != None:
            run_names = _get_all_run_names()
            if run_name in run_names:
                # change run_name
                index = 1
                while True:
                    if '{}-{}'.format(run_name, index) in run_names:
                        index += 1
                        continue
                    run_name = '{}-{}'.format(run_name, index)
                    _logger.info("Change run_name to %s.", run_name)
                    break

        return original_start_run(run_id, experiment_id, run_name, nested, tags, description, log_system_metrics)

    patch = mock.patch.object(mlflow, 'start_run', new_start_run)
    patch.__enter__()


def get_runid_by_name(run_name, experiment_name=None):
    client = MlflowClient()
    if experiment_name == None:
        experiment = client.get_experiment(_get_experiment_id())
    else:
        experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(experiment._experiment_id)

    for run in runs:
        if run.__dict__['_data'].tags['mlflow.runName'] == run_name:
            res = json.loads(runs[0].__dict__[
                             '_data'].tags['mlflow.log-model.history'])
            return res[0]['run_id']
            break

    return None
