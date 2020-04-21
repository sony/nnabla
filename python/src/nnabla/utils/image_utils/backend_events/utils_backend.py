import os
import binascii
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

    def imread(self):
        pass

    def imsave(self):
        pass

    def imresize(self):
        pass

    @staticmethod
    def get_file_extension(path):
        ext = ''
        file_signature = {
            '.bmp': ['424d'],
            '.dib': ['424d'],
            '.pgm': ['50350a'],
            '.jpeg': ['ffd8ffe0'],
            '.jpg': ['ffd8ffe0'],
            '.png': ['89504e470d0a1a0a'],
            '.tif': ['492049'],
            '.tiff': ['492049'],
            '.eps': ['c5d0d3c6'],
            '.gif': ['474946383761', '474946383961'],
            '.ico': ['00000100'],
        }
        if hasattr(path, "read"):
            data = binascii.hexlify(path.read()).decode('utf-8')
            path.seek(0)
            for extension, signature in file_signature.items():
                for s in signature:
                    if data.startswith(s):
                        ext = extension
        elif isinstance(path, str):
            ext = os.path.splitext(path)[1]
        return ext

    def get_best_backend(self, path, operator):
        if self._default_backend:
            self._current_backend = self._default_backend
            return self._default_backend()
        else:
            ext = self.get_file_extension(path)
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
                            while end.next[ext] is not None:
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
        ext = self.get_file_extension(path)
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
