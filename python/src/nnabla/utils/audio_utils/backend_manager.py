# Copyright 2020,2021 Sony Corporation.
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


import importlib
from collections import OrderedDict


class AudioUtilsBackendManager(object):
    _instance = None
    _loaded = False
    _backend = None

    def __new__(cls, *args, **kwargs):
        # as singleton
        if cls._instance is None:
            cls._instance = super(AudioUtilsBackendManager, cls).__new__(
                cls, *args, **kwargs)
            cls._loaded = False

        return cls._instance

    def __init__(self):
        if not self._loaded:
            load_modules = ["pydub"]
            self.backends = OrderedDict()

            for module in load_modules:
                self._import_backend(module)

            for backend, module in self.backends.items():
                if module is not None:
                    self.backend = backend
                    break
            else:
                try:
                    import scipy.io.wavfile
                except ImportError:
                    raise ImportError("No backend module is found. "
                                      "At least you must install scipy in your environment.")

            self._loaded = True

    def _import_backend(self, backend):
        try:
            module = importlib.import_module(
                "nnabla.utils.audio_utils.{}_utils".format(backend))
        except ImportError:
            # log backend status in _check_backend
            module = None

        self.backends[backend] = module

    def _check_backend(self, backend):
        if backend is None:
            raise ModuleNotFoundError("No available audio backends in"
                                      " your environment.")
        if self.backends.get(backend, None) is None:
            raise ValueError(
                "{} is not found in your environment.".format(backend))

    def get_available_backends(self):
        return [key for key, value in self.backends.items() if value is not None]

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        self._check_backend(backend)

        self._backend = backend

        # Too verbose for common user
        # logger.info(
        #     "use {} as the backend of audio utils".format(self._backend))

    @property
    def module(self):
        self._check_backend(self._backend)

        return self.backends[self._backend]


backend_manager = AudioUtilsBackendManager()
