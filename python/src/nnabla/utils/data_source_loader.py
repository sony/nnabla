# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

'''
Contents loader functions for DataSource.

'''

from __future__ import absolute_import

from six.moves import map
from shutil import rmtree
from six import BytesIO
from six import StringIO
from six.moves.urllib.parse import urljoin
import contextlib
import csv

# TODO temporary work around to suppress FutureWarning message.
import warnings
warnings.simplefilter('ignore', category=FutureWarning)
import h5py

import numpy
import os
import six.moves.urllib.request as request
import six
import tempfile

from nnabla.utils.image_utils import imresize, imread
from nnabla.logger import logger

# Expose for backward compatibility
from .download import download, get_data_home

pypng_available = False
try:
    import png
    pypng_available = True
except ImportError:
    pass
cv2_available = False
try:
    import cv2
    # TODO: Currently cv2 image reader doesn't work.
    # cv2_available = True
except ImportError:
    pass


class FileReader:
    '''FileReader

    Read dataset from several data sources.
    Supported data sources are,

    * Local file (file or directory name)
    * HTTP/HTTPS (URI)
    * S3         (URI with s3:// prefix)

    Currently HTTP/HTTPS source does not support CACHE input because
    there is no standard way to get directory entry with
    HTTP/HTTPS/protocol.


    To access S3 data, you must specify credentials with environment
    variable.

    For example,

    ::

        $ export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
        $ export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

    Or, you can specify PROFILE with following.

    ::

        $ export AWS_DEFAULT_PROFILE=my_profile
'''

    def __init__(self, base_uri):
        self._base_uri = base_uri
        if base_uri[0:5].lower() == 's3://':
            self._file_type = 's3'
            uri_header, uri_body = self._base_uri.split('://', 1)
            us = uri_body.split('/')
            bucketname = us.pop(0)
            self._s3_base_key = '/'.join(us)
            logger.info('Creating session for S3 bucket {}'.format(bucketname))

            import boto3
            self._s3_bucket = boto3.session.Session().resource('s3').Bucket(bucketname)

        elif base_uri[0:7].lower() == 'http://' or base_uri[0:8].lower() == 'https://':
            self._file_type = 'http'
        else:
            self._file_type = 'file'

    def read_s3_object(self, key):
        retry = 1
        result = ''
        while True:
            if retry > 10:
                logger.log(99, 'read_s3_object() retry count over give up.')
                raise
            try:
                result = self._s3_bucket.Object(key).get()['Body'].read()
                break
            except:
                logger.log(
                    99, 'read_s3_object() fails retrying count {}/10.'.format(retry))
                retry += 1

        return result

    @contextlib.contextmanager
    def open(self, filename=None, textmode=False):
        if filename is None:
            filename = self._base_uri
        else:
            if self._file_type == 's3':
                filename = urljoin(self._base_uri.replace(
                    's3://', 'http://'), filename.replace('\\', '/')).replace('http://', 's3://')
            elif self._file_type == 'http':
                filename = urljoin(self._base_uri, filename.replace('\\', '/'))
            else:
                filename = os.path.abspath(os.path.join(os.path.dirname(
                    self._base_uri.replace('\\', '/')), filename.replace('\\', '/')))
        f = None
        if self._file_type == 's3':
            uri_header, uri_body = filename.split('://', 1)
            us = uri_body.split('/')
            bucketname = us.pop(0)
            key = '/'.join(us)
            logger.info('Opening {}'.format(key))
            if textmode:
                f = StringIO(self.read_s3_object(key).decode('utf-8'))
            else:
                f = BytesIO(self.read_s3_object(key))
        elif self._file_type == 'http':
            f = request.urlopen(filename)
        else:
            if textmode:
                f = open(filename, 'rt')
            else:
                f = open(filename, 'rb')
        yield f
        f.close()

    @contextlib.contextmanager
    def open_cache(self, cache_name):
        if self._file_type == 's3':
            tmpdir = tempfile.mkdtemp()
            filename = urljoin((self._base_uri + '/').replace('s3://', 'http://'),
                               cache_name.replace('\\', '/')).replace('http://', 's3://')
            key = '/'.join(filename.split('/')[3:])
            fn = '{}/{}'.format(tmpdir, os.path.basename(filename))
            with open(fn, 'wb') as f:
                f.write(self.read_s3_object(key))
            with h5py.File(fn, 'r') as h5:
                yield h5
            rmtree(tmpdir, ignore_errors=True)
        elif self._file_type == 'http':
            pass
        else:
            filename = os.path.abspath(os.path.join(os.path.dirname(
                (self._base_uri + '/').replace('\\', '/')), cache_name.replace('\\', '/')))
            with h5py.File(filename, 'r') as h5:
                yield h5

    def listdir(self):
        if self._file_type == 's3':
            list = []
            for fn in self._s3_bucket.objects.filter(Prefix=self._s3_base_key + '/', Delimiter='/'):
                list.append(os.path.basename(fn.key))
            return sorted(list)
        elif self._file_type == 'http':
            return None
        return [f for f in sorted(os.listdir(self._base_uri)) if os.path.splitext(f)[1].lower() == ".h5"]


def load_image_imread(file, shape=None, max_range=1.0):
    '''
    Load image from file like object.

    :param file: Image contents
    :type file: file like object.
    :param shape: shape of output array
        e.g. (3, 128, 192) : n_color, height, width.
    :type shape: tuple of int
    :param float max_range: the value of return array ranges from 0 to `max_range`.

    :return: numpy array

    '''
    img255 = imread(
        file)  # return value is from zero to 255 (even if the image has 16-bitdepth.)

    if len(img255.shape) == 2:  # gray image
        height, width = img255.shape
        if shape is None:
            out_height, out_width, out_n_color = height, width, 1
        else:
            out_n_color, out_height, out_width = shape
        assert(out_n_color == 1)
        if out_height != height or out_width != width:
            # imresize returns 0 to 255 image.
            img255 = imresize(img255, (out_height, out_width))
        img255 = img255.reshape((out_n_color, out_height, out_width))
    elif len(img255.shape) == 3:  # RGB image
        height, width, n_color = img255.shape
        if shape is None:
            out_height, out_width, out_n_color = height, width, n_color
        else:
            out_n_color, out_height, out_width = shape
        assert(out_n_color == n_color)
        if out_height != height or out_width != width or out_n_color != n_color:
            # imresize returns 0 to 255 image.
            img255 = imresize(img255, (out_height, out_width, out_n_color))
        img255 = img255.transpose(2, 0, 1)

    if max_range < 0 or max_range == 255.0:
        return img255
    else:
        return img255 * (max_range / 255.0)


def load_image_pypng(file, shape=None, max_range=1.0):
    import png
    r = png.Reader(file=file)
    width, height, pixels, metadata = r.read()
    bitscale = 2 ** metadata['bitdepth'] - 1
    img = numpy.array(list(pixels), dtype=numpy.float32).reshape(
        (height, width, -1)) / bitscale  # (height, width, n_channel)
    if metadata['alpha'] and metadata['planes'] == 4:  # RGBA
        # TODO: this case is note tested well
        try:
            bg = numpy.array(metadata['background']) / bitscale
        except KeyError:
            bg = numpy.array([1.0, 1.0, 1.0])
        rgb = img[:, :, :3]
        alpha = img[:, :, 3]
        imshp = alpha.shape
        img = numpy.outer((1 - alpha), bg).reshape(imshp + (3,)) +\
            numpy.tile(alpha.reshape(imshp + (1,)), (1, 1, 3)) * rgb
        out_n_color = 3
    elif metadata['alpha'] and metadata['planes'] == 2:  # (gray, alpha)
        # TODO: this case is note tested well
        try:
            bg = numpy.array(metadata['background']) / bitscale
        except KeyError:
            bg = numpy.array([1.0])
        rgb = img[:, :, :1]
        alpha = img[:, :, 1]
        imshp = alpha.shape
        img = numpy.outer((1 - alpha), bg).reshape(imshp + (1,)
                                                   ) + alpha.reshape(imshp + (1,)) * rgb
        out_n_color = 1
    else:  # RGB or Gray
        out_n_color = metadata['planes']

    # Reshape image
    if max_range < 0:
        max_range = 255
    if shape is None:
        return img.transpose(2, 0, 1) * max_range
    else:
        out_n_color, out_height, out_width = shape
        return imresize(img, (out_height, out_width)).transpose((2, 0, 1)) * max_range / 255.0


def load_image_cv2(file, shape=None, max_range=1.0):
    img = cv2.imdecode(numpy.asarray(bytearray(file.read()),
                                     dtype=numpy.uint8), cv2.IMREAD_UNCHANGED)

    if len(img.shape) == 2:  # gray image
        height, width = img.shape
        img = img.reshape(1, height, width)

    elif len(img.shape) == 3:  # rgb image
        if img.shape[2] == 3:
            img = img[:, :, ::-1].copy()  # BGR to RGB
            img = img.transpose(2, 0, 1)
        elif img.shape[2] == 4:
            img = img.transpose(2, 0, 1)  # BGRA to RGBA
            img = numpy.array([img[2], img[1], img[0], img[3]])

    if max_range < 0:
        pass
    elif max_range == 255:
        if img.dtype == numpy.uint8:
            pass
        elif img.dtype == numpy.uint16:
            img = numpy.uint8(img / 256)
    elif max_range == 65535:
        if img.dtype == numpy.uint8:
            img = numpy.uint16(img * 256)
        elif img.dtype == numpy.uint16:
            pass
    else:
        if img.dtype == numpy.uint8:
            img = numpy.float32(img) * max_range / 255.0
        elif img.dtype == numpy.uint16:
            img = numpy.float32(img) * max_range / 65535.0
    return img


def load_image(file, shape=None, normalize=False):
    if normalize:
        max_range = 1.0
    else:
        max_range = -1
    global cv2_available
    global pypng_available

    if cv2_available:
        return load_image_cv2(file, shape, max_range)
    else:
        ext = None
        try:
            ext = os.path.splitext(file.name)[1].lower()
        except:
            pass
        if ext == '.png' and pypng_available:
            r = png.Reader(file=file)
            width, height, pixels, metadata = r.read()
            file.seek(0)
            if metadata['bitdepth'] > 8:  # if png with high bitdepth
                return load_image_pypng(file, shape, max_range)
        return load_image_imread(file, shape, max_range)


def load_csv(file, shape=None, normalize=False):
    """
    Load CSV file.

    :param file: CSV file.
    :type file: file like object
    :param shape : data array is reshape to this shape.
    :type shape: tuple of int

    :return: numpy array
    """
    value_list = []
    if six.PY2:
        for row in csv.reader(file):
            value_list.append(list(map(float, row)))
    elif six.PY34:
        for row in csv.reader([l.decode('utf-8') for l in file.readlines()]):
            value_list.append(list(map(float, row)))
    if shape is None:
        return numpy.array(value_list)
    else:
        return numpy.array(value_list).reshape(shape)


def load_npy(path, shape=None, normalize=False):
    if shape is None:
        return numpy.load(path)
    else:
        return numpy.load(path).reshape(shape)


_load_functions = {
    '.bmp': load_image,
    '.jpg': load_image,
    '.jpeg': load_image,
    '.png': load_image,
    '.gif': load_image,
    '.tif': load_image,
    '.tiff': load_image,
    '.csv': load_csv,
    '.npy': load_npy}


def register_load_function(ext, function):
    _load_functions[ext] = function


def load(ext):
    if ext in _load_functions:
        return _load_functions[ext]
    raise ValueError(
        'File format with extension "{}" is not supported.'.format(ext))


def _download_hook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner
