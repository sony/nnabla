'''
Provide data iterator for Cifar100 examples.
'''
from contextlib import contextmanager
import numpy as np
import struct
import tarfile
import zlib

from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download


class Cifar100DataSource(DataSource):
    '''
    Get data directly from cifar100 dataset from Internet(yann.lecun.com).
    '''

    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        return (image, label)

    def __init__(self, train=True, shuffle=False, rng=None):
        super(Cifar100DataSource, self).__init__(shuffle=shuffle)

        # Lock
        lockfile = os.path.join(get_data_home(), "cifar100.lock")
        start_time = time.time()
        while True:  # busy-lock due to communication between process spawn by mpirun
            try:
                fd = os.open(lockfile, os.O_CREAT|os.O_EXCL|os.O_RDWR)
                break;
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise 
                if (time.time() - start_time) >= 60 * 30:  # wait for 30min
                    raise Exception("Timeout occured.")
                
            time.sleep(5)
            
        self._train = train
        data_uri = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        logger.info('Getting labeled data from {}.'.format(data_uri))
        r = download(data_uri)  # file object returned
        with tarfile.open(fileobj=r, mode="r:gz") as fpin:
            # Training data
            if train:
                images = []
                labels = []
                for member in fpin.getmembers():
                    if "train" not in member.name:
                        continue
                    fp = fpin.extractfile(member)
                    data = np.load(fp)
                    images = data["data"]
                    labels = data["fine_labels"]
                self._size = 50000
                self._images = images.reshape(self._size, 3, 32, 32)
                self._labels = np.array(labels).reshape(-1, 1)
            # Validation data
            else:
                for member in fpin.getmembers():
                    if "test" not in member.name:
                        continue
                    fp = fpin.extractfile(member)
                    data = np.load(fp)
                    images = data["data"]
                    labels = data["fine_labels"]
                self._size = 10000
                self._images = images.reshape(self._size, 3, 32, 32)
                self._labels = np.array(labels).reshape(-1, 1)
        r.close()
        logger.info('Getting labeled data from {} done.'.format(data_uri))
 
        self._size = self._labels.size
        self._variables = ('x', 'y')
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.reset()

        # Unlock
        os.close(fd)
        os.unlink(lockfile)
        
    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(Cifar100DataSource, self).reset()

    @property
    def images(self):
        """Get copy of whole data with a shape of (N, 1, H, W)."""
        return self._images.copy()

    @property
    def labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        return self._labels.copy()


@contextmanager
def data_iterator_cifar100(batch_size,
                        train=True,
                        rng=None,
                        shuffle=True,
                        with_memory_cache=False,
                        with_parallel=False,
                        with_file_cache=False):
    '''
    Provide DataIterator with :py:class:`Cifar100DataSource`
    with_memory_cache, with_parallel and with_file_cache option's default value is all False,
    because :py:class:`Cifar100DataSource` is able to store all data into memory.

    For example,

    .. code-block:: python

        with data_iterator_cifar100(True, batch_size) as di:
            for data in di:
                SOME CODE TO USE data.

    '''
    with Cifar100DataSource(train=train, shuffle=shuffle, rng=rng) as ds, \
        data_iterator(ds,
                      batch_size,
                      with_memory_cache,
                      with_parallel,
                      with_file_cache) as di:
        yield di
