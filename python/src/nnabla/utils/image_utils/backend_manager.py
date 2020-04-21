import os
import glob
import importlib

from .backend_events.utils_backend import ImageUtilsBackend


class ImageUtilsBackendManager(object):
    def __new__(cls, *args, **kwargs):
        path = os.path.dirname(__file__)
        for f in glob.glob(os.path.join(path, "backend_events/*_utils.py")):
            module = os.path.splitext(os.path.basename(f))[0]
            try:
                importlib.import_module(
                    "nnabla.utils.image_utils.backend_events.{}".format(module))
            except ImportError:
                pass
        return ImageUtilsBackend()


backend_manager = ImageUtilsBackendManager()
