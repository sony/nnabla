# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

from __future__ import print_function

import numpy as np
import os
import time

from nnabla.logger import logger


class Monitor(object):

    """
    This class is created to setup the output directory of the monitoring logs.
    The created :class:`nnabla.monitor.Monitor` instance is passed to classes
    in the following :ref:`monitors`.
    """

    def __init__(self, save_path):
        self._save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    @property
    def save_path(self):
        return self._save_path


class MonitorSeries(object):
    """Logs a series of values.

    The values are displayed and/or output to the file
    ``<name>-series.txt``.

    Example:

    .. code-block:: python

        mons = MonitorSeries('mon', interval=2)
        for i in range(10):
            mons.add(i, i * 2)

    Args:
        name (str): Name of the monitor. Used in the log.
        monitor (~nnabla.monitor.Monitor): Monitor class instance.
        interval (int): Interval of flush the outputs. The values added by
            ``.add()`` are averaged during interval.
        verbose (bool): Output to screen.

    """

    def __init__(self, name, monitor=None, interval=1, verbose=True):
        self.name = name
        self.interval = interval
        self.verbose = verbose
        self.sp = None
        if monitor is not None:
            self.sp = os.path.join(
                monitor.save_path, name.replace(" ", "-")) + ".series.txt"
            # refresh output file
            with open(self.sp, "w") as f:
                pass
        self.flush_at = -1
        self.buf = []

    def add(self, index, value):
        """Add a value to the series.

        Args:
            index (int): Index.
            value (float): Value.

        """
        self.buf.append(value)
        if (index - self.flush_at) < self.interval:
            return
        value = np.mean(self.buf)
        if self.verbose:
            logger.info("iter={} {{{}}}={}".format(index, self.name, value))
        if self.sp is not None:
            with open(self.sp, 'a') as fd:
                print("{} {:g}".format(index, value), file=fd)
        self.flush_at = index
        self.buf = []


class MonitorTimeElapsed(object):

    """Logs the elapsed time.

    The values are displayed and/or output to the file
    ``<name>-timer.txt``.

    Example:

    .. code-block:: python

        import time
        mont = MonitorTimeElapsed("time", interval=2)
        for i in range(10):
            time.sleep(1)
            mont.add(i)

    Args:
        name (str): Name of the monitor. Used in the log.
        monitor (~nnabla.monitor.Monitor): Monitor class instance.
        interval (int): Interval of flush the outputs.
            The elapsed time is calculated within the interval.
        verbose (bool): Output to screen.

    """

    def __init__(self, name, monitor=None, interval=100, verbose=True):
        self.name = name
        self.interval = interval
        self.verbose = verbose
        self.sp = None
        if monitor is not None:
            self.sp = os.path.join(
                monitor.save_path, name.replace(" ", "-")) + ".timer.txt"
            # refresh output file
            with open(self.sp, "w") as f:
                pass
        self.flush_at = -1
        self.start = time.time()
        self.lap = self.start

    def add(self, index):
        """Calculate time elapsed from the point previously called
        this method or this object is created to this is called.

        Args:
            index (int): Index to be displayed, and be used to take intervals.

        """
        if (index - self.flush_at) < self.interval:
            return
        now = time.time()
        elapsed = now - self.lap
        elapsed_total = now - self.start
        it = index - self.flush_at
        self.lap = now
        if self.verbose:
            logger.info("iter={} {{{}}}={}[sec/{}iter] {}[sec]".format(
                index, self.name, elapsed, it, elapsed_total))
        if self.sp is not None:
            with open(self.sp, 'a') as fd:
                print("{} {} {} {}".format(index, elapsed,
                                           it, elapsed_total), file=fd)
        self.flush_at = index


class MonitorImage(object):

    """Saves a series of images.

    The `.add()` method takes a ``(N,..., C, H, W)`` array as an input,
    and ``num_images`` of ``[H, W, :min(3, C)]`` are saved into
    the monitor folder for each interval.

    The values are displayed and/or output to the file
    ``<name>/{iter}-{image index}.png``.

    Example:

    .. code-block:: python

        import numpy as np
        m = Monitor('tmp.monitor')
        mi = MonitorImage('noise', m, interval=2, num_images=2)
        x = np.random.randn(10, 3, 8, 8)
        for i in range(10):
            mi.add(i, x)

    Args:
        name (str): Name of the monitor. Used in the log.
        monitor (~nnabla.monitor.Monitor): Monitor class instance.
        interval (int): Interval of flush the outputs.
        num_images (int): Number of images to be saved in each iteration.
        normalize_method (function): A function that takes a NCHW format image minibatch
            as :obj:`numpy.ndarray`. The function should define a normalizer which
            map any inputs to a range of [0, 1]. The default normalizer normalizes
            the images into min-max normalization.

    """

    def __init__(self, name, monitor, interval=1000, verbose=True, num_images=16, normalize_method=None):
        self.name = name
        self.interval = interval
        self.verbose = verbose
        self.normalize_method = normalize_method
        if normalize_method is None:
            self.normalize_method = self.default_normalize_method
        self.num_images = num_images
        self.save_dir = os.path.join(monitor.save_path, name.replace(' ', '-'))
        try:
            os.makedirs(self.save_dir)
        except OSError:
            pass  # python2 does not support exists_ok arg

    def default_normalize_method(self, x):
        ma = x.max()
        mi = x.min()
        return (x - mi) / (ma - mi)

    def add(self, index, var):
        """Add a minibatch of images to the monitor.

        Args:
            index (int): Index.
            var (:obj:`~nnabla.Variable`, :obj:`~nnabla.NdArray`, or :obj:`~numpy.ndarray`):
                A minibatch of images with ``(N, ..., C, H, W)`` format.
                If C == 2, blue channel is appended with ones. If C > 3,
                the array will be sliced to remove C > 3 sub-array.

        """
        import nnabla as nn
        from nnabla.utils.image_utils import imsave
        if index != 0 and (index + 1) % self.interval != 0:
            return
        if isinstance(var, nn.Variable):
            data = var.d.copy()
        elif isinstance(var, nn.NdArray):
            data = var.data.copy()
        else:
            assert isinstance(var, np.ndarray)
            data = var.copy()
        assert data.ndim > 2
        channels = data.shape[-3]
        data = data.reshape(-1, *data.shape[-3:])
        data = data[:min(data.shape[0], self.num_images)]
        data = self.normalize_method(data)
        if channels > 3:
            data = data[:, :3]
        elif channels == 2:
            data = np.concatenate(
                [data, np.ones((data.shape[0], 1) + data.shape[-2:])], axis=1)
        path_tmpl = os.path.join(self.save_dir, '{:06d}-{}.png')
        for j in range(min(self.num_images, data.shape[0])):
            img = data[j].transpose(1, 2, 0)
            if img.shape[-1] == 1:
                img = img[..., 0]
            path = path_tmpl.format(index, '{:03d}'.format(j))
            imsave(path, img)
        if self.verbose:
            logger.info("iter={} {{{}}} are written to {}.".format(
                index, self.name, path_tmpl.format(index, '*')))


class MonitorImageTile(MonitorImage):

    """Saving a series of images.

    The `.add()` method takes a ``(N,..., C, H, W)`` array as an input,
    and ``num_images`` tiled  ``(H, W, :min(3, C))`` images are saved into
    the monitor folder for each interval.

    The values are displayed and/or output to the file
    ``<name>/{iter}-{image index}.png``.

    Example:

    .. code-block:: python

        import numpy as np
        m = Monitor('tmp.monitor')
        mi = MonitorImageTile('noise_noise', m, interval=2, num_images=4)
        x = np.random.randn(10, 3, 8, 8)
        for i in range(10):
            mi.add(i, x)

    Args:
        name (str): Name of the monitor. Used in the log.
        monitor (~nnabla.monitor.Monitor): Monitor class instance.
        interval (int): Interval of flush the outputs.
        num_images (int): Number of images tiled to be saved into a single image
            in each iteration.
        normalize_method (function): A function that takes a NCHW format image minibatch
            as :obj:`numpy.ndarray`. The function should define a normalizer which
            map any inputs to a range of [0, 1]. The default normalizer normalizes
            the images into min-max normalization.

    """

    def add(self, index, var):
        """Add a minibatch of images to the monitor.

        Args:
            index (int): Index.
            var (:obj:`~nnabla.Variable`, :obj:`~nnabla.NdArray`, or :obj:`~numpy.ndarray`):
                A minibatch of images with ``(N, ..., C, H, W)`` format.
                If C == 2, blue channel is appended with ones. If C > 3,
                the array will be sliced to remove C > 3 sub-array.

        """
        import nnabla as nn
        from nnabla.utils.image_utils import imsave
        if index and (index + 1) % self.interval:
            return
        if isinstance(var, nn.Variable):
            data = var.d.copy()
        elif isinstance(var, nn.NdArray):
            data = var.data.copy()
        else:
            assert isinstance(var, np.ndarray)
            data = var.copy()
        assert data.ndim > 2
        channels = data.shape[-3]
        data = data.reshape(-1, *data.shape[-3:])
        data = data[:min(data.shape[0], self.num_images)]
        data = self.normalize_method(data)
        if channels > 3:
            data = data[:, :3]
        elif channels == 2:
            data = np.concatenate(
                [data, np.ones((data.shape[0], 1) + data.shape[-2:])], axis=1)
        tile = tile_images(data)
        path = os.path.join(self.save_dir, '{:06d}.png'.format(index))
        imsave(path, tile)
        if self.verbose:
            logger.info("iter={} {{{}}} is written to {}.".format(
                index, self.name, path))


def tile_images(data, padsize=1, padval=0):
    """
    Convert an array with shape of (B, C, H, W) into a tiled image.

    Args:
        data (~numpy.ndarray): An array with shape of (B, C, H, W).
        padsize (int): Each tile has padding with this size.
        padval (float): Padding pixels are filled with this value.

    Returns:
        tile_image (~numpy.ndarray): A tile image.

    """
    assert (data.ndim == 4)
    data = data.transpose(0, 2, 3, 1)

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (
        (0, n ** 2 - data.shape[0]),
        (0, padsize),
        (0, padsize)
    ) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(
        data, padding, mode='constant', constant_values=(padval, padval))
    data = data.reshape(
        (n, n)
        + data.shape[1:]
    ).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    if data.shape[2] == 1:
        # Return as (H, W)
        return data.reshape(data.shape[:2])
    return data


def plot_series(filename, plot_kwargs=None):
    """Plot series data from MonitorSeries output text file.

    Args:
        filename (str): Path to *.series.txt file produced by :obj:`~nnabla.MonitorSeries` class.
        plot_kwags (dict, optional):
            Keyward arguments passed to :function:`matplotlib.pyplot.plot`.

    Note:
        matplotlib package is required.

    """
    import matplotlib.pyplot as plt

    if plot_kwargs is None:
        plot_kwargs = {}

    data = np.genfromtxt(filename, dtype='i8,f4', names=['k', 'v'])
    index = data['k']
    values = data['v']
    plt.plot(index, values, **plot_kwargs)


def plot_time_elapsed(filename, elapsed=False, unit='s', plot_kwargs=None):
    """Plot series data from MonitorTimeElapsed output text file.

    Args:
        filename (str): Path to *.series.txt file produced by :obj:`~nnabla.MonitorSeries` class.
        elapsed (bool): If ``True``, it plots the total elapsed time.
        unit (str):
            Time unit chosen from ``'s'``, ``'m'``, ``'h'``, or ``'d'``.
        plot_kwags (dict, optional):
            Keyward arguments passed to :function:`matplotlib.pyplot.plot`.

    Note:
        matplotlib package is required.

    """
    import matplotlib.pyplot as plt

    if plot_kwargs is None:
        plot_kwargs = {}

    data_column = 3 if elapsed else 1
    data = np.genfromtxt(filename, dtype='i8,f4',
                         usecols=(0, data_column), names=['k', 'v'])
    index = data['k']
    values = data['v']
    if unit == 's':
        pass
    elif unit == 'm':
        values /= 60
    elif unit == 'h':
        values /= 3600
    elif unit == 'd':
        values /= 3600 * 24
    else:
        raise ValueError('The argument `unit` must be chosen from {s|m|h|d}.')
    plt.plot(index, values, **plot_kwargs)
