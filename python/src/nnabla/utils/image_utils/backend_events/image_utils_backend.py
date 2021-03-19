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


from collections import OrderedDict


class ImageUtilsBackend(object):
    _subclasses = OrderedDict()
    _best_backend = {}
    _best_backend_name = {}
    _current_backend = None

    def __init__(self):
        for subclass in ImageUtilsBackend.__subclasses__():
            ImageUtilsBackend._subclasses[subclass.__name__] = subclass
        self.next = {}
        self._default_backend = None

    def accept(self):
        return "NG"

    def imread(self, path, grayscale=False, size=None, interpolate="bilinear",
               channel_first=False, as_uint16=False, num_channels=-1):
        backend = self.get_best_backend(path, 'load')
        if backend is None:
            raise ValueError("No available backend to load image.")
        return backend.imread(path, grayscale, size, interpolate, channel_first,
                              as_uint16, num_channels)

    def imsave(self, path, img, channel_first=False, as_uint16=False, auto_scale=True):
        backend = self.get_best_backend(path, 'save')
        if backend is None:
            raise ValueError("No available backend to save image.")
        return backend.imsave(path, img, channel_first, as_uint16, auto_scale)

    def imresize(self, img, size, interpolate="bilinear", channel_first=False):
        backend = self.get_best_backend('', 'resize')
        if backend is None:
            raise ValueError("No available backend to resize image.")
        return backend.imresize(img, size, interpolate, channel_first)

    def get_best_backend(self, path, operator):
        from nnabla.utils.data_source_loader import get_file_extension
        if self._default_backend:
            self._current_backend = self._default_backend
            return self._default_backend()
        else:
            ext = get_file_extension(path)
            if ext not in self._best_backend:
                self._best_backend[ext] = None
                self._best_backend_name[ext] = None
                for name, backend in self._subclasses.items():
                    state = backend().accept(path, ext, operator)
                    if state == "OK":
                        if self._best_backend[ext] is None:
                            self._best_backend_name[ext] = name
                            self._best_backend[ext] = backend()
                            self._best_backend[ext].next[ext] = None
                        else:
                            end = self._best_backend[ext]
                            while ext in end.next and end.next[ext] is not None:
                                end = self._best_backend[ext].next[ext]
                            end.next[ext] = backend()
                    elif state == "Recommended":
                        self._best_backend_name[ext] = name
                        if self._best_backend[ext] is not None:
                            head = self._best_backend[ext]
                            self._best_backend[ext] = backend()
                            self._best_backend[ext].next[ext] = head
                        else:
                            self._best_backend[ext] = backend()
            if self._best_backend[ext]:
                self._current_backend = self._best_backend[ext].__class__
                return self._best_backend[ext]
            else:
                raise ValueError("Currently, No backend available.")

    def next_available(self, path):
        from nnabla.utils.data_source_loader import get_file_extension
        ext = get_file_extension(path)
        if ext not in self.next:
            raise ValueError("Currently, No backend available.")
        else:
            if self.next[ext]:
                self._current_backend = self.next[ext].__class__
                return self.next[ext]
            else:
                raise ValueError("Currently, No backend available.")

    def get_backend(self):
        if self._default_backend:
            return self._default_backend.__name__
        else:
            return self._current_backend.__name__ if self._current_backend else ''

    def set_backend(self, name):
        if name in self._subclasses:
            self._default_backend = self._subclasses[name]
        else:
            self._default_backend = None

    def get_available_backends(self):
        return [key for key, value in self._subclasses.items() if value is not None]

    def get_backend_from_name(self, name):
        if name in self._subclasses:
            return self._subclasses[name]()
        else:
            return None
