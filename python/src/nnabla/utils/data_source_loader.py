# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

'''
Contents loader functions for DataSource.

'''

from __future__ import absolute_import

import contextlib
import csv
# TODO temporary work around to suppress FutureWarning message.
import warnings

from six import BytesIO
from six import StringIO
from six.moves import map
from six.moves.urllib.parse import urljoin

warnings.simplefilter('ignore', category=FutureWarning)
import h5py

import numpy
import scipy.io.wavfile
import os
import urllib.request as request
import six
import tempfile
import binascii
from shutil import rmtree

from nnabla.utils.image_utils import imresize, imread
from nnabla.utils.audio_utils import auresize, auread
from nnabla.logger import logger


# Expose for backward compatibility
from .download import download, get_data_home

pydub_available = False
with warnings.catch_warnings():
    warnings.simplefilter('error', RuntimeWarning)
    try:
        from pydub import AudioSegment
        pydub_available = True
    except ImportError:
        pass
    except RuntimeWarning as w:
        if "Couldn't find ffmpeg or avconv" in w:
            pydub_available = True
    warnings.simplefilter('default', RuntimeWarning)


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
        self.base_ext = os.path.splitext(self._base_uri)[1].lower()
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
    def open(self, filename=None, textmode=False, encoding='utf-8-sig'):
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
                f = StringIO(self.read_s3_object(key).decode(encoding))
            else:
                f = BytesIO(self.read_s3_object(key))
        elif self._file_type == 'http':
            f = request.urlopen(filename)
        else:
            if textmode:
                f = open(filename, 'rt', encoding=encoding)
            else:
                f = open(filename, 'rb')
        f.ext = os.path.splitext(filename)[1].lower()
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


def get_file_extension(source):
    ext = ''
    file_signature = {
        '.bmp': (['424d'], 0x0),
        '.dib': (['424d'], 0x0),
        '.pgm': (['50350a'], 0x0),
        '.jpeg': (['ffd8ff'], 0x0),
        '.jpg': (['ffd8ff'], 0x0),
        '.png': (['89504e470d0a1a0a'], 0x0),
        '.tif': (['492049'], 0x0),
        '.tiff': (['492049'], 0x0),
        '.eps': (['c5d0d3c6'], 0x0),
        '.gif': (['474946383761', '474946383961'], 0x0),
        '.ico': (['00000100'], 0x0),
        '.dcm': (['4449434d'], 0x80),
        '.wav': (['52494646'], 0x0),
    }
    if hasattr(source, "read"):
        if hasattr(source, "name"):
            ext = os.path.splitext(source.name)[1].lower()
        else:
            for extension, (signature, offset) in file_signature.items():
                source.seek(offset)
                data = binascii.hexlify(source.read()).decode('utf-8')
                source.seek(0)
                for s in signature:
                    if data.startswith(s):
                        ext = extension
    elif isinstance(source, str):
        ext = os.path.splitext(source)[1].lower()
    return ext


class ResourceFileReader:
    '''
    arrange a file or BytesIO object with extension info appended

    '''

    def __init__(self, source):
        self._source = source
        self.handler = None
        if isinstance(self._source, str):
            self.handler = FileReader(self._source)
            self.ext = self.handler.base_ext
            if not self.ext:
                with self.handler.open() as f:
                    self.ext = get_file_extension(f)
        elif hasattr(self._source, "read") and self._accepted_source_type(self._source):
            self.handler = self._source
            self.ext = get_file_extension(self._source)
        else:
            raise ValueError(
                "ResourceFileReader only accept path str, binary file handler or BytesIO")

    def _accepted_source_type(self, source):
        if not hasattr(source, 'seek'):
            return False
        try:
            source.seek(0)
            c = source.read(1)
            source.seek(0)
        except Exception:
            return False
        if isinstance(c, str):
            return False
        return True

    @contextlib.contextmanager
    def open(self):
        if isinstance(self.handler, FileReader):
            with self.handler.open() as f:
                if not f.ext:
                    f.ext = self.ext
                yield f
        else:
            self.handler.seek(0)
            self.handler.ext = self.ext
            yield self.handler
            self.handler.seek(0)


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
    orig_img = imread(
        file)  # return value is from zero to 255 (even if the image has 16-bitdepth.)

    if len(orig_img.shape) == 2:  # gray image
        height, width = orig_img.shape
        if shape is None:
            out_height, out_width, out_n_color = height, width, 1
        else:
            out_n_color, out_height, out_width = shape
        assert (out_n_color == 1)
        if out_height != height or out_width != width:
            # imresize returns 0 to 255 image.
            orig_img = imresize(orig_img, (out_height, out_width))
        orig_img = orig_img.reshape((out_n_color, out_height, out_width))
    elif len(orig_img.shape) == 3:  # RGB image
        height, width, n_color = orig_img.shape
        if shape is None:
            out_height, out_width, out_n_color = height, width, n_color
        else:
            out_n_color, out_height, out_width = shape
        assert (out_n_color == n_color)
        if out_height != height or out_width != width or out_n_color != n_color:
            # imresize returns 0 to 255 image.
            orig_img = imresize(orig_img, (out_height, out_width, out_n_color))
        orig_img = orig_img.transpose(2, 0, 1)

    if max_range < 0:
        return orig_img
    else:
        # 16bit depth
        if orig_img.dtype == 'uint16':
            if max_range == 65535.0:
                return orig_img
            return orig_img * (max_range / 65535.0)
        # 8bit depth (default)
        else:
            if max_range == 255.0:
                return orig_img
            return orig_img * (max_range / 255.0)


def load_image(file, shape=None, normalize=False):
    if normalize:
        max_range = 1.0
    else:
        max_range = -1

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
    for row in csv.reader([l.decode('utf-8') for l in file.readlines()]):
        if len(row):
            value_list.append(list(map(float, row)))
    try:
        if shape is None:
            return numpy.array(value_list)
        else:
            return numpy.array(value_list).reshape(shape)
    except:
        logger.log(99, 'Failed to load array from "{}".'.format(file.name))
        raise


def load_npy(path, shape=None, normalize=False):
    if shape is None:
        return numpy.load(path, allow_pickle=True)
    else:
        return numpy.load(path, allow_pickle=True).reshape(shape)


def load_wav(path, shape=None, normalize=False):
    wav = scipy.io.wavfile.read(path)[1]
    if shape is None:
        if len(wav.shape) == 1:
            return wav.reshape(-1, 1)
        else:
            return wav
    else:
        return wav.reshape(shape)


def load_audio_pydub(path, shape=None, normalize=False):
    if shape:
        return auresize(auread(path), shape)
    return auread(path)


def load_audio(file, shape=None, normalize=False):
    global pydub_available
    if pydub_available:
        return load_audio_pydub(file, shape, normalize)
    else:
        return load_wav(file, shape, normalize)


_load_functions = {
    '.bmp': load_image,
    '.jpg': load_image,
    '.jpeg': load_image,
    '.png': load_image,
    '.gif': load_image,
    '.tif': load_image,
    '.tiff': load_image,
    '.dcm': load_image,
    '.csv': load_csv,
    '.npy': load_npy,
    '.wav': load_audio}


def load(ext):
    import nnabla.utils.callback as callback
    func = callback.get_load_image_func(ext)
    if func is not None:
        return func
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
