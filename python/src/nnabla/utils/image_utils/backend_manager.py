# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

import glob
import importlib
import os

from .backend_events.image_utils_backend import ImageUtilsBackend


class ImageUtilsBackendManager(object):
    def __new__(cls, *args, **kwargs):
        path = os.path.dirname(__file__)
        for f in glob.glob(os.path.join(path, "backend_events/*_backend.py")):
            module = os.path.splitext(os.path.basename(f))[0]
            try:
                importlib.import_module(
                    "nnabla.utils.image_utils.backend_events.{}".format(module))
            except ImportError:
                pass
        return ImageUtilsBackend()


backend_manager = ImageUtilsBackendManager()
