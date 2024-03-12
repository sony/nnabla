# Copyright 2023 Sony Corporation.
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

import pytest
import numpy as np

import nnabla.experimental.framework as framework

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from nnabla.utils.data_iterator import data_iterator, data_iterator_simple
from nnabla.utils.data_source import DataSource
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed


def augmentation(h, test, aug):
    if aug is None:
        aug = not test
    if aug:
        h = F.image_augmentation(h, (1, 28, 28), (0, 0), 0.9, 1.1, 0.3,
                                 1.3, 0.1, False, False, 0.5, False, 1.5, 0.5, False, 0.1, 0)
    return h


def mnist_lenet_prediction(image, test=False, aug=None, name=''):
    image /= 255.0
    image = augmentation(image, test, aug)

    c1 = PF.convolution(image, 16, (5, 5), name='mnist-conv1' + name)
    c1 = F.relu(F.max_pooling(c1, (2, 2)))
    c2 = PF.convolution(c1, 16, (5, 5), name='mnist-conv2' + name)
    c2 = F.relu(F.max_pooling(c2, (2, 2)))
    c3 = F.relu(PF.affine(c2, 50, name='mnist-fc3' + name))
    c4 = PF.affine(c3, 10, name='mnist-fc4' + name)
    return c4


def mnist_resnet_prediction(image, test=False, aug=None, name=''):
    image /= 255.0
    image = augmentation(image, test, aug)

    def bn(x):
        return PF.batch_normalization(x, batch_stat=not test)

    def res_unit(x, scope):
        C = x.shape[1]
        with nn.parameter_scope(scope):
            with nn.parameter_scope('mnist-conv1' + name):
                h = F.elu(
                    bn(PF.convolution(x, C / 2, (1, 1), with_bias=False)))
            with nn.parameter_scope('mnist-conv2' + name):
                h = F.elu(
                    bn(PF.convolution(h, C / 2, (3, 3), pad=(1, 1), with_bias=False)))
            with nn.parameter_scope('mnist-conv3' + name):
                h = bn(PF.convolution(h, C, (1, 1), with_bias=False))
        return F.elu(F.add2(h, x))
    # Conv1 --> 64 x 32 x 32
    with nn.parameter_scope('mnist-conv1' + name):
        c1 = F.elu(
            bn(PF.convolution(image, 64, (3, 3), pad=(3, 3), with_bias=False)))
    # Conv2 --> 64 x 16 x 16
    c2 = F.max_pooling(res_unit(c1, 'mnist-conv2' + name), (2, 2))
    # Conv3 --> 64 x 8 x 8
    c3 = F.max_pooling(res_unit(c2, 'mnist-conv3' + name), (2, 2))
    # Conv4 --> 64 x 8 x 8
    c4 = res_unit(c3, 'mnist-conv4' + name)
    # Conv5 --> 64 x 4 x 4
    c5 = F.max_pooling(res_unit(c4, 'mnist-conv5' + name), (2, 2))
    # Conv5 --> 64 x 4 x 4
    c6 = res_unit(c5, 'mnist-conv6' + name)
    pl = F.average_pooling(c6, (4, 4))
    with nn.parameter_scope("classifier"):
        y = PF.affine(pl, 10)
    return y


def lenet(image, test=False, name=''):
    import nnabla.initializer as I
    l_rng = np.random.RandomState(666)
    h = PF.convolution(image, 16, (5, 5), (1, 1),
                       w_init=I.UniformInitializer((-1, 1), rng=l_rng),
                       with_bias=False, name='conv1' + name)
    h = PF.batch_normalization(h, batch_stat=not test, name='conv1-bn' + name)
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.convolution(h, 16, (5, 5), (1, 1),
                       w_init=I.UniformInitializer((-1, 1), rng=l_rng),
                       with_bias=True, name='conv2' + name)
    h = PF.batch_normalization(h, batch_stat=not test, name='conv2-bn' + name)
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.affine(h, 50, w_init=I.UniformInitializer((-1, 1), rng=l_rng),
                  with_bias=False, name='fc1' + name)
    h = F.relu(h)

    pred = PF.affine(h, 10, w_init=I.UniformInitializer((-1, 1), rng=l_rng),
                     with_bias=True, name='fc2' + name)
    return pred


def logistic_regression(image, test=False, name=''):
    import nnabla.initializer as I
    l_rng = np.random.RandomState(666)
    h = nn.Variable(image.shape)
    pred = PF.affine(h, 10, w_init=I.UniformInitializer((-1, 1), rng=l_rng),
                     with_bias=True, name='lr' + name)
    return pred


class MnistData(framework.data.Data):
    '''
    Mnist data.

    Args:
        shuffle (bool): Whether the dataset is shuffled or not.
        rng (None or :obj:`numpy.random.RandomState`): Numpy random number generator.
        batch_size (int): Size of data unit.
    '''

    class MnistDataSource(DataSource):
        '''
        Get data directly from MNIST dataset from Internet(yann.lecun.com).

        Args:
            train (bool): Whether the dataset is for training or validation.
            shuffle (bool): Whether the dataset is shuffled or not.
            rng (None or :obj:`numpy.random.RandomState`): Numpy random number generator.
        '''

        def load_mnist(self, train=True):
            '''
            Load MNIST dataset images and labels from the original page by Yan LeCun or the cache file.

            Args:
                train (bool): The testing dataset will be returned if False. Training data has 60000 images, while testing has 10000 images.

            Returns:
                numpy.ndarray: A shape of (#images, 1, 28, 28). Values in [0.0, 1.0].
                numpy.ndarray: A shape of (#images, 1). Values in {0, 1, ..., 9}.
            '''

            import struct
            import zlib

            from nnabla.logger import logger
            from nnabla.utils.data_source_loader import download

            if train:
                image_uri = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
                label_uri = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
            else:
                image_uri = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
                label_uri = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
            logger.info('Getting label data from {}.'.format(label_uri))

            # With python3 we can write this logic as following, but with
            # python2, gzip.object does not support file-like object and
            # urllib.request does not support 'with statement'.
            #
            #   with request.urlopen(label_uri) as r, gzip.open(r) as f:
            #       _, size = struct.unpack('>II', f.read(8))
            #       labels = numpy.frombuffer(f.read(), numpy.uint8).reshape(-1, 1)

            r = download(label_uri)
            data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
            _, size = struct.unpack('>II', data[0:8])
            labels = np.frombuffer(
                data[8:], np.uint8).reshape(-1, 1)
            r.close()
            logger.info('Getting label data done.')

            logger.info('Getting image data from {}.'.format(image_uri))
            r = download(image_uri)
            data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
            _, size, height, width = struct.unpack('>IIII', data[0:16])
            images = np.frombuffer(data[16:], np.uint8).reshape(
                size, 1, height, width)
            r.close()
            logger.info('Getting image data done.')

            return images, labels

        def _get_data(self, position):
            image = self._images[self._indexes[position]]
            label = self._labels[self._indexes[position]]
            return (image, label)

        def __init__(self, train=True, shuffle=False, rng=None):
            from numpy.random import RandomState

            super().__init__(shuffle=shuffle)
            self._train = train

            self._images, self._labels = self.load_mnist(train)

            self._size = self._labels.size
            self._variables = ('x', 'y')
            if rng is None:
                rng = RandomState(313)
            self.rng = rng
            self.reset()

        def reset(self):
            if self._shuffle:
                self._indexes = self.rng.permutation(self._size)
            else:
                self._indexes = np.arange(self._size)
            super().reset()

        @property
        def images(self):
            # Get copy of whole data with a shape of (N, 1, H, W).
            return self._images.copy()

        @property
        def labels(self):
            # Get copy of whole label with a shape of (N, 1).
            return self._labels.copy()

    def __data_source__(self, train, shuffle, rng):
        return self.MnistDataSource(train, shuffle, rng)

    def __data_iterator__(self, data_source, batch_size, rng=None, with_memory_cache=False, with_file_cache=False):
        return data_iterator(data_source, batch_size, rng, with_memory_cache, with_file_cache)

    def __init__(self, shuffle, rng, batch_size):
        self._data_source = self.__data_source__(True, shuffle, rng)
        self._data_iterator = self.__data_iterator__(
            self._data_source, batch_size, rng)

        self._v_data_source = self.__data_source__(False, shuffle, rng)
        self._v_data_iterator = self.__data_iterator__(
            self._v_data_source, batch_size)

    def reset(self, train=True):
        if train:
            self._data_iterator._reset()
        else:
            self._v_data_iterator._reset()

    def next(self, train=True):
        if train:
            return self._data_iterator.next()
        else:
            return self._v_data_iterator.next()

    def size(self, train=True):
        if train:
            return self._data_iterator.size
        else:
            return self._v_data_iterator.size


class MnistWorker(framework.worker.Worker):
    '''
    Nnist worker.

    Args:
        data (:obj:`Data`): Data object.
        batch_size (int): Size of data unit.
    '''

    def _categorical_error(self, pred, label):
        '''
        Compute categorical error given score vectors and labels as numpy.ndarray.
        '''
        pred_label = pred.argmax(1)
        return (pred_label != label.flat).mean()

    def _mnist_lenet_prediction(self, image, test=False, aug=None):
        '''
        Construct LeNet for MNIST.
        '''
        return mnist_lenet_prediction(image, test, aug, '')

    def _mnist_resnet_prediction(self, image, test=False, aug=None):
        '''
        Construct ResNet for MNIST.
        '''
        return mnist_resnet_prediction(image, test, aug, '')

    def _ctx_init(self):
        self._ctx = nn.get_current_context()
        if len(self._ctx.backend) == 0 or len(self._ctx.array_class) == 0:
            from nnabla.ext_utils import get_extension_context
            self._ctx = get_extension_context('cpu')
            nn.set_default_context(self._ctx)

    def _var_init(self):
        # Create training input variables.
        self._image = nn.Variable([self._batch_size, 1, 28, 28])
        self._label = nn.Variable([self._batch_size, 1])

        # Create testing input variables.
        self._vimage = nn.Variable([self._batch_size, 1, 28, 28])
        self._vlabel = nn.Variable([self._batch_size, 1])

    def _comm_init(self):
        pass

    def _graph_init(self):
        # Create CNN network for both training and testing.
        if self._net == 'lenet':
            self._mnist_cnn_prediction = self._mnist_lenet_prediction
        elif self._net == 'resnet':
            self._mnist_cnn_prediction = self._mnist_resnet_prediction
        else:
            raise ValueError("Unknown network type {}".format(self._net))

        # Create training prediction graph.
        self._pred = self._mnist_cnn_prediction(
            self._image, test=False, aug=self._taug)
        self._pred.persistent = True

        # Create testing prediction graph.
        self._vpred = self._mnist_cnn_prediction(
            self._vimage, test=True, aug=self._vaug)

        # Create evaluation metrics value list, length one for verr only
        self._v_list = [0.0]

    def _loss_init(self):
        # Create loss function.
        self._loss = F.mean(F.softmax_cross_entropy(self._pred, self._label))

    def _solver_init(self):
        # Create Solver. If training from checkpoint, load the info.
        self._solver = S.Adam(self._learning_rate)
        self._solver.set_parameters(nn.get_parameters())

    def _monitor_init(self):
        # Create monitor.
        from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
        monitor = Monitor(self._monitor_path + self._net)
        self._monitor_loss = MonitorSeries(
            "Training loss", monitor, interval=10)
        self._monitor_err = MonitorSeries(
            "Training error", monitor, interval=10)
        self._monitor_time = MonitorTimeElapsed(
            "Training time", monitor, interval=100)
        self._monitor_verr = MonitorSeries("Test error", monitor, interval=1)

    def __init__(self, data, batch_size, max_epoch, net, taug=False, vaug=False, learning_rate=0.001, monitor_path='tmp.monitor-mnist-', val_iterval=4, weight_decay=0):
        self._data = data
        self._batch_size = batch_size
        self._max_epoch = max_epoch
        self._net = net
        self._taug = taug
        self._vaug = vaug
        self._learning_rate = learning_rate
        self._monitor_path = monitor_path
        self._val_interval = val_iterval
        self._weight_decay = weight_decay

        self._iter_per_epoch = data.size() // batch_size
        self._val_iter = data.size(train=False) // batch_size
        self._clear_buffer = True
        self._clear_no_need_grad = True

        self._updater = framework.updater.Updater(train_data_feeder=self._train_data_feeder,
                                                  solver_zero_grad=self._solver_zero_grad,
                                                  solver_update=self._solver_update,
                                                  loss_forward=self._loss_forward,
                                                  loss_backward=self._loss_backward,
                                                  solver_update_callback_on_start=self._solver_update_callback_on_start,
                                                  accum_grad=1, comm=None, grads=[])
        self._evaluator = framework.evaluator.Evaluator(eval_data_feeder=self._eval_data_feeder,
                                                        eval_graph_forward=self._eval_graph_forward,
                                                        eval_callback_on_start=[
                                                            self._eval_callback_on_start],
                                                        eval_callback_on_finish=[
                                                            self._eval_callback_on_finish],
                                                        val_iter=self._val_iter)

    def iter_loop_end(self, i, e):
        self._loss.data.cast(np.float32, self._ctx)
        self._pred.data.cast(np.float32, self._ctx)
        self._err = self._categorical_error(self._pred.d, self._label.d)
        self._monitor_loss.add(
            e * self._iter_per_epoch + i, self._loss.d.copy())
        self._monitor_err.add(e * self._iter_per_epoch + i, self._err)
        self._monitor_time.add(e * self._iter_per_epoch + i)

    def brief(self):
        return self._err

    def evaluate_at_epoch_start(self, e):
        if self._loopCallbackCondition(e=e, toExecute=lambda i, e, ag: (e % self._val_interval == 0)):
            self._evaluator.evaluate(e)

    def evaluate_at_training_end(self):
        self._evaluator.evaluate(self._max_epoch)

    def _train_data_feeder(self, i, e, ag):
        if self._loopCallbackCondition(i=i, e=e, ag=ag):
            self._image.d, self._label.d = self._data.next()

    def _solver_update_callback_on_start(self, i, e):
        if self._loopCallbackCondition(i=i, e=e):
            self._solver.weight_decay(self._weight_decay)

    def _eval_callback_on_start(self, e):
        if self._loopCallbackCondition(e=e):
            self._v_list[0] = 0.0

    def _eval_data_feeder(self, e, vi):
        if self._loopCallbackCondition(e=e, ag=vi):
            self._vimage.d, self._vlabel.d = self._data.next(False)

    def _eval_graph_forward(self, e, vi):
        if self._loopCallbackCondition(i=e, ag=vi):
            self._vpred.forward(clear_buffer=self._clear_buffer)
            self._vpred.data.cast(np.float32, self._ctx)
            self._v_list[0] += self._categorical_error(
                self._vpred.d, self._vlabel.d)

    def _eval_callback_on_finish(self, e):
        if self._loopCallbackCondition(e=e):
            self._monitor_verr.add(e, self._v_list[0] / self._val_iter)


@pytest.mark.parametrize('shuffle', [True])
@pytest.mark.parametrize('seed', [1223])
@pytest.mark.parametrize('batch_size', [1024])
@pytest.mark.parametrize(['max_epoch', 'net'], [(4, 'lenet'), (2, 'resnet')])
def test_mnist_training(shuffle, seed, batch_size, max_epoch, net):
    if net == 'resnet':
        pytest.skip("Skip resnet to save time.")

    rng = np.random.RandomState(seed)

    data = MnistData(shuffle, rng, batch_size)
    worker = MnistWorker(data, batch_size, max_epoch, net)
    trainer = framework.trainer.Trainer(worker)
    trainer.train()
    brief = trainer.brief()

    # Check convergency.
    np.testing.assert_allclose(brief, 0, rtol=0.05, atol=0.05)


class NaiveClassificationData(framework.data.Data):
    '''
    '''

    class NaiveClassificationDataSource(DataSource):
        '''
        '''

        def _get_data(self, position):
            return self._pred[position], self._label[position]

        def __init__(self, seed, channel, height, width, data_size):
            super().__init__()

            self._channel = channel
            self._height = height
            self._width = width

            self._variables = ('x', 'y')
            self._size = data_size
            self._rng = np.random.RandomState(seed)
            self._pred = self._rng.randn(
                self._size, self._channel, self._height, self._width)
            self._label = (self._pred[:, 0, 0, 0] > 0).reshape(-1, 1)

        def reset(self):
            super().reset()

    def __data_source__(self, seed, channel, height, width, data_size):
        return self.NaiveClassificationDataSource(seed, channel, height, width, data_size)

    def __data_iterator__(self, data_source, batch_size):
        return data_iterator(data_source, batch_size)

    def __init__(self, seed, v_seed, batch, channel, height, width, data_size):
        self._data_source = self.__data_source__(
            seed, channel, height, width, data_size)
        self._data_iterator = self.__data_iterator__(self._data_source, batch)

        self._v_data_source = self.__data_source__(
            v_seed, channel, height, width, data_size)
        self._v_data_iterator = self.__data_iterator__(
            self._v_data_source, batch)

    def reset(self, train=True):
        if train:
            self._data_iterator._reset()
        else:
            self._v_data_iterator._reset()

    def next(self, train=True):
        if train:
            return self._data_iterator.next()
        else:
            return self._v_data_iterator.next()

    def size(self, train=True):
        if train:
            return self._data_iterator.size
        else:
            return self._v_data_iterator.size


class NaiveClassificationWorker(framework.worker.Worker):
    '''
    A simple classification model which needs to care about error.

    Args:
        data (:obj:`Data`): Data object.
        max_epoch (:obj:`int`): Max epoch to train.
        batch (:obj:`int`): Batch number of input.
        channel (:obj:`int`): Channel number of input.
        height (:obj:`int`): Height of input.
        width (:obj:`int`): Width of input.
        monitor_path (:obj:`str`): Monitor path.
        iter_per_epoch (:obj:`int`, optional): Iterations per one epoch. If not set, this value are determined by `tdata.size // tdata.batch_size`.
        val_iter (:obj:`int`, optional): Iterations for evaluation. If not set, this value are determined by `vdata.size // vdata.batch_size`.

    '''

    def _lenet(self, image, test=False):
        return lenet(image, test, '')

    def _var_init(self):
        self._tinput = nn.Variable(
            [self._batch, self._channel, self._height, self._width])
        self._tlabel = nn.Variable([self._batch, 1])

        self._vinput = nn.Variable(
            [self._batch, self._channel, self._height, self._width])
        self._vlabel = nn.Variable([self._batch, 1])

    def _comm_init(self):
        pass

    def _graph_init(self):
        self._tpred = self._lenet(self._tinput)
        self._tpred = self._tpred.apply(persistent=True)
        self._vpred = self._lenet(self._vinput, test=True)
        self._vpred = self._vpred.apply(persistent=True)

    def _loss_init(self):
        # Loss and error.
        self._loss = F.mean(
            F.softmax_cross_entropy(self._tpred, self._tlabel))
        self._terr = F.mean(F.top_n_error(
            self._tpred.get_unlinked_variable(), self._tlabel))
        self._vloss = F.mean(
            F.softmax_cross_entropy(self._vpred, self._vlabel))
        self._verr = F.mean(F.top_n_error(
            self._vpred.get_unlinked_variable(), self._vlabel))

    def _solver_init(self):
        self._solver = S.Adam()
        self._solver.set_parameters(nn.get_parameters())

    def _monitor_init(self):
        # Monitors.
        from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
        monitor = Monitor(self._monitor_path)
        self._monitor_loss = MonitorSeries(
            "Training loss", monitor, interval=1)
        self._monitor_err = MonitorSeries(
            "Training error", monitor, interval=1)
        self._monitor_vloss = MonitorSeries("Valid loss", monitor, interval=1)
        self._monitor_verr = MonitorSeries("Valid error", monitor, interval=1)
        self._monitor_time = MonitorTimeElapsed(
            "Training time", monitor, interval=10)

    def __init__(self, data, max_epoch, batch, channel, height, width, monitor_path, iter_per_epoch=None, val_iter=None):
        self._data = data
        self._max_epoch = max_epoch
        self._batch = batch
        self._channel = channel
        self._height = height
        self._width = width
        self._monitor_path = monitor_path
        if iter_per_epoch is None:
            self._iter_per_epoch = self._data.size() // batch
        else:
            self._iter_per_epoch = iter_per_epoch
        if val_iter is None:
            self._val_iter = self._data.size(train=False) // batch
        else:
            self._val_iter = val_iter
        self._clear_buffer = True
        self._clear_no_need_grad = True

        self._updater = framework.updater.Updater(train_data_feeder=self._train_data_feeder,
                                                  solver_zero_grad=self._solver_zero_grad,
                                                  solver_update=self._solver_update,
                                                  loss_forward=self._loss_forward,
                                                  loss_backward=self._loss_backward,
                                                  loss_forward_callback_on_finish=[
                                                      self._loss_forward_callback_on_finish],
                                                  accum_grad=1, comm=None, grads=[])
        self._evaluator = framework.evaluator.Evaluator(eval_data_feeder=self._eval_data_feeder,
                                                        eval_graph_forward=self._eval_graph_forward,
                                                        eval_callback_on_finish=[
                                                            self._eval_callback_on_finish],
                                                        val_iter=self._val_iter)

    def brief(self):
        return self._loss.d, self._terr.d, self._vloss.d, self._verr.d

    def iter_loop_end(self, i, e):
        self._monitor_loss.add(e * self._iter_per_epoch + i, self._loss.d)
        self._monitor_err.add(e * self._iter_per_epoch + i, self._terr.d)
        self._monitor_time.add(e * self._iter_per_epoch + i)

    def _loss_forward_callback_on_finish(self, i, e, ag):
        self._terr.forward(clear_no_need_grad=self._clear_no_need_grad)

    def _train_data_feeder(self, i, e, ag):
        self._tinput.d, self._tlabel.d = self._data.next()

    def _eval_data_feeder(self, e, vi):
        self._vinput.d, self._vlabel.d = self._data.next(train=False)

    def _eval_graph_forward(self, e, vi):
        self._vloss.forward(clear_no_need_grad=self._clear_no_need_grad)
        self._verr.forward(clear_no_need_grad=self._clear_no_need_grad)

    def _eval_callback_on_finish(self, e):
        self._monitor_vloss.add(e * self._iter_per_epoch, self._vloss.d)
        self._monitor_verr.add(e * self._iter_per_epoch, self._verr.d)


def naive_classification_training_without_framework(seed, v_seed, max_epoch, batch, channel, height, width, monitor_path, data_size):
    iter_per_epoch = data_size // batch

    rng = np.random.RandomState(seed)
    v_rng = np.random.RandomState(v_seed)
    data = rng.randn(data_size, channel, height, width)
    label = (data[:, 0, 0, 0] > 0).reshape(-1, 1)
    v_data = v_rng.randn(data_size, channel, height, width)
    v_label = (v_data[:, 0, 0, 0] > 0).reshape(-1, 1)

    def load_func(i):
        return data[i], label[i]

    def v_load_func(i):
        return v_data[i], v_label[i]

    di = data_iterator_simple(load_func, data_size, batch)
    v_di = data_iterator_simple(v_load_func, data_size, batch)

    tinput = nn.Variable([batch, channel, height, width])
    tlabel = nn.Variable([batch, 1])
    tpred = lenet(tinput, name="-com")
    tpred = tpred.apply(persistent=True)

    vinput = nn.Variable([batch, channel, height, width])
    vlabel = nn.Variable([batch, 1])
    vpred = lenet(vinput, test=True, name="-com")
    vpred = vpred.apply(persistent=True)

    solver = S.Adam()
    solver.set_parameters(nn.get_parameters())

    monitor = Monitor(monitor_path + "-com")
    monitor_tloss = MonitorSeries("Training loss", monitor, interval=1)
    monitor_terr = MonitorSeries("Training error", monitor, interval=1)
    monitor_vloss = MonitorSeries("Valid loss", monitor, interval=1)
    monitor_verr = MonitorSeries("Valid error", monitor, interval=1)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=10)

    tloss = F.mean(F.softmax_cross_entropy(tpred, tlabel))
    terr = F.mean(F.top_n_error(tpred.get_unlinked_variable(), tlabel))
    vloss = F.mean(F.softmax_cross_entropy(vpred, vlabel))
    verr = F.mean(F.top_n_error(vpred.get_unlinked_variable(), vlabel))

    for e in range(max_epoch):
        for i in range(iter_per_epoch):
            vinput.d, vlabel.d = v_di.next()
            vloss.forward(clear_no_need_grad=True)
            verr.forward(clear_no_need_grad=True)
        monitor_vloss.add(e * iter_per_epoch, vloss.d)
        monitor_verr.add(e * iter_per_epoch, verr.d)
        for i in range(iter_per_epoch):
            tinput.d, tlabel.d = di.next()
            solver.zero_grad()
            tloss.forward(clear_no_need_grad=True)
            terr.forward(clear_no_need_grad=True)
            tloss.backward(clear_buffer=True)
            solver.update()
            monitor_tloss.add(e * iter_per_epoch + i, tloss.d)
            monitor_terr.add(e * iter_per_epoch + i, terr.d)
            monitor_time.add(e * iter_per_epoch + i)
    for i in range(iter_per_epoch):
        vinput.d, vlabel.d = v_di.next()
        vloss.forward(clear_no_need_grad=True)
        verr.forward(clear_no_need_grad=True)
    monitor_vloss.add(max_epoch * iter_per_epoch, vloss.d)
    monitor_verr.add(max_epoch * iter_per_epoch, verr.d)

    return tloss.d, terr.d, vloss.d, verr.d


@pytest.mark.parametrize('seed, v_seed', [(114, 514)])
@pytest.mark.parametrize('max_epoch', [4])
@pytest.mark.parametrize('batch', [50])
def test_naive_classification_training(seed, v_seed, max_epoch, batch, channel=1, height=8, width=8, monitor_path='tmp.monitor-cls', data_size=400):
    data = NaiveClassificationData(
        seed, v_seed, batch, channel, height, width, data_size)
    worker = NaiveClassificationWorker(
        data, max_epoch, batch, channel, height, width, monitor_path, data_size // batch)
    trainer = framework.trainer.Trainer(worker)
    trainer.train()
    brief = trainer.brief()

    ref = naive_classification_training_without_framework(
        seed, v_seed, max_epoch, batch, channel, height, width, monitor_path, data_size)

    # Commpare the results.
    np.testing.assert_allclose(brief, ref, rtol=1e-06, atol=1e-06)


class NaiveRegressionData(framework.data.Data):
    '''
    In this implementation, two framework.data.Data instances are used for train data and test data.
    '''

    class NaiveRegressionDataSource(DataSource):
        '''
        '''

        def _get_data(self, position):
            img = self._digits.images[position]
            label = self._digits.target[position]
            return img[None], np.array([label]).astype(np.int32)

        def __init__(self, rs, train):
            super().__init__()

            digits = load_digits()
            self._digits = digits
            if train:
                self._digits.data, _, self._digits.target, _, self._digits.images, _ = train_test_split(
                    digits.data, digits.target, digits.images, test_size=0.2, random_state=rs)
            else:
                _, self._digits.data, _, self._digits.target, _, self._digits.images = train_test_split(
                    digits.data, digits.target, digits.images, test_size=0.2, random_state=rs)
            self._variables = ('x', 'y')
            self._size = self._digits.target.size

        def reset(self):
            super().reset()

    def __data_source__(self, rs, train):
        return self.NaiveRegressionDataSource(rs, train)

    def __data_iterator__(self, data_source, batch_size=1):
        return data_iterator(data_source, batch_size)

    def __init__(self, batch, rs=666, train=True):
        self._data_source = self.__data_source__(rs, train)
        self._data_iterator = self.__data_iterator__(
            self._data_source, batch_size=batch)

    def reset(self):
        self._data_iterator._reset()

    def next(self):
        return self._data_iterator.next()

    def size(self):
        return self._data_iterator.size


class NaiveRegressionWorker(framework.worker.Worker):
    '''
    A simple regression model which does not need to care about error.

    Args:
        data (:obj:`Data`): Data object.
        max_epoch (:obj:`int`): Max epoch to train.
        batch (:obj:`int`): Batch number of input.
        channel (:obj:`int`): Channel number of input.
        height (:obj:`int`): Height of input.
        width (:obj:`int`): Width of input.
        learning_rate (:obj:`float`): Learning rate.
        monitor_path (:obj:`str`): Monitor path.
        iter_per_epoch (:obj:`int`, optional): Iterations per one epoch. If not set, this value are determined by `tdata.size // tdata.batch_size`.
        val_iter (:obj:`int`, optional): Iterations for evaluation. If not set, this value are determined by `vdata.size // vdata.batch_size`.

    '''

    def _logistic_regression(self, image, test=False):
        return logistic_regression(image, test, '')

    def _var_init(self):
        self._tinput = nn.Variable(
            [self._batch, self._channel, self._height, self._width])
        self._tlabel = nn.Variable([self._batch, 1])

        self._vinput = nn.Variable(
            [self._batch, self._channel, self._height, self._width])
        self._vlabel = nn.Variable([self._batch, 1])

    def _comm_init(self):
        pass

    def _graph_init(self):
        self._tpred = self._logistic_regression(self._tinput)
        self._tpred = self._tpred.apply(persistent=True)
        self._vpred = self._logistic_regression(self._vinput, test=True)
        self._vpred = self._vpred.apply(persistent=True)

    def _loss_init(self):
        # Loss and error.
        self._loss = F.mean(
            F.softmax_cross_entropy(self._tpred, self._tlabel))
        self._vloss = F.mean(
            F.softmax_cross_entropy(self._vpred, self._vlabel))

    def _solver_init(self):
        self._solver = S.Sgd(self._learning_rate)
        self._solver.set_parameters(nn.get_parameters())

    def _monitor_init(self):
        # Monitors.
        from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
        monitor = Monitor(self._monitor_path)
        self._monitor_loss = MonitorSeries(
            "Training loss", monitor, interval=1)
        self._monitor_vloss = MonitorSeries("Valid loss", monitor, interval=1)
        self._monitor_time = MonitorTimeElapsed(
            "Training time", monitor, interval=10)

    def __init__(self, data, v_data, max_epoch, batch, channel, height, width, learning_rate, monitor_path, iter_per_epoch=None, val_iter=None):
        self._data = data
        self._v_data = v_data
        self._max_epoch = max_epoch
        self._batch = batch
        self._channel = channel
        self._height = height
        self._width = width
        self._learning_rate = learning_rate
        self._monitor_path = monitor_path
        if iter_per_epoch is None:
            self._iter_per_epoch = self._data.size() // batch
        else:
            self._iter_per_epoch = iter_per_epoch
        if val_iter is None:
            self._val_iter = self._v_data.size() // batch
        else:
            self._val_iter = val_iter
        self._clear_buffer = True
        self._clear_no_need_grad = True

        self._updater = framework.updater.Updater(train_data_feeder=self._train_data_feeder,
                                                  solver_zero_grad=self._solver_zero_grad,
                                                  solver_update=self._solver_update,
                                                  loss_forward=self._loss_forward,
                                                  loss_backward=self._loss_backward,
                                                  accum_grad=1, comm=None, grads=[])
        self._evaluator = framework.evaluator.Evaluator(eval_data_feeder=self._eval_data_feeder,
                                                        eval_graph_forward=self._eval_graph_forward,
                                                        eval_callback_on_finish=[
                                                            self._eval_callback_on_finish],
                                                        val_iter=self._val_iter)

    def brief(self):
        return self._loss.d, self._vloss.d

    def iter_loop_end(self, i, e):
        self._monitor_loss.add(e * self._iter_per_epoch + i, self._loss.d)
        self._monitor_time.add(e * self._iter_per_epoch + i)

    def _train_data_feeder(self, i, e, ag):
        self._tinput.d, self._tlabel.d = self._data.next()

    def _eval_data_feeder(self, e, vi):
        self._vinput.d, self._vlabel.d = self._v_data.next()

    def _eval_graph_forward(self, e, vi):
        self._vloss.forward(clear_no_need_grad=self._clear_no_need_grad)

    def _eval_callback_on_finish(self, e):
        self._monitor_vloss.add(e * self._iter_per_epoch, self._vloss.d)


def naive_regression_training_without_framework(max_epoch, batch, channel, height, width, learning_rate, monitor_path):
    digits = load_digits()
    v_digits = load_digits()
    rs = 666
    digits.data, v_digits.data, digits.target, v_digits.target, digits.images, v_digits.images = train_test_split(
        digits.data, digits.target, digits.images, test_size=0.2, random_state=rs)
    data_size = digits.target.size
    v_data_size = v_digits.target.size
    iter_per_epoch = data_size // batch
    v_iter_per_epoch = v_data_size // batch

    def load_func(position):
        img = digits.images[position]
        label = digits.target[position]
        return img[None], np.array([label]).astype(np.int32)

    def v_load_func(position):
        img = v_digits.images[position]
        label = v_digits.target[position]
        return img[None], np.array([label]).astype(np.int32)

    di = data_iterator_simple(load_func, data_size, batch)
    v_di = data_iterator_simple(v_load_func, v_data_size, batch)

    tinput = nn.Variable([batch, channel, height, width])
    tlabel = nn.Variable([batch, 1])
    tpred = logistic_regression(tinput, name='-com')
    tpred = tpred.apply(persistent=True)

    vinput = nn.Variable([batch, channel, height, width])
    vlabel = nn.Variable([batch, 1])
    vpred = logistic_regression(vinput, test=True, name='-com')
    vpred = vpred.apply(persistent=True)

    solver = S.Sgd(learning_rate)
    solver.set_parameters(nn.get_parameters())

    monitor = Monitor(monitor_path + "-com")
    monitor_tloss = MonitorSeries("Training loss", monitor, interval=1)
    monitor_vloss = MonitorSeries("Valid loss", monitor, interval=1)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=10)

    tloss = F.mean(F.softmax_cross_entropy(tpred, tlabel))
    vloss = F.mean(F.softmax_cross_entropy(vpred, vlabel))

    for e in range(max_epoch):
        for i in range(v_iter_per_epoch):
            vinput.d, vlabel.d = v_di.next()
            vloss.forward(clear_no_need_grad=True)
            monitor_vloss.add(e * v_iter_per_epoch, vloss.d)
        for i in range(iter_per_epoch):
            tinput.d, tlabel.d = di.next()
            solver.zero_grad()
            tloss.forward(clear_no_need_grad=True)
            tloss.backward(clear_buffer=True)
            solver.update()
            monitor_tloss.add(e * iter_per_epoch + i, tloss.d)
            monitor_time.add(e * iter_per_epoch + i)
    for i in range(v_iter_per_epoch):
        vinput.d, vlabel.d = v_di.next()
        vloss.forward(clear_no_need_grad=True)
        monitor_vloss.add(max_epoch * v_iter_per_epoch, vloss.d)

    return tloss.d, vloss.d


@pytest.mark.parametrize('max_epoch', [2])
@pytest.mark.parametrize('batch', [80])
def test_naive_regression_training(max_epoch, batch, channel=3, height=8, width=8, learning_rate=0.01, monitor_path='tmp.monitor-reg'):
    data = NaiveRegressionData(batch)
    v_data = NaiveRegressionData(batch, train=False)
    worker = NaiveRegressionWorker(
        data, v_data, max_epoch, batch, channel, height, width, learning_rate, monitor_path)
    trainer = framework.trainer.Trainer(worker)
    trainer.train()
    brief = trainer.brief()

    ref = naive_regression_training_without_framework(
        max_epoch, batch, channel, height, width, learning_rate, monitor_path)

    # Commpare the results.
    np.testing.assert_allclose(brief, ref, rtol=1e-06, atol=1e-06)


class NaiveRegressionDataOne(framework.data.Data):
    '''
    In this implementation, one framework.data.Data instance is used for both train data and test data.
    '''

    class NaiveRegressionDataSource(DataSource):
        '''
        '''

        def _get_data(self, position):
            img = self._digits.images[position]
            label = self._digits.target[position]
            return img[None], np.array([label]).astype(np.int32)

        def reset(self):
            super().reset()

    class NaiveRegressionTrainDataSource(NaiveRegressionDataSource):
        '''
        '''

        def __init__(self, rs):
            super().__init__()

            digits = load_digits()
            self._digits = digits
            self._digits.data, _, self._digits.target, _, self._digits.images, _ = train_test_split(
                digits.data, digits.target, digits.images, test_size=0.2, random_state=rs)
            self._variables = ('x', 'y')
            self._size = self._digits.target.size

    class NaiveRegressionTestDataSource(NaiveRegressionDataSource):
        '''
        '''

        def __init__(self, rs):
            super().__init__()

            digits = load_digits()
            self._digits = digits
            _, self._digits.data, _, self._digits.target, _, self._digits.images = train_test_split(
                digits.data, digits.target, digits.images, test_size=0.2, random_state=rs)
            self._variables = ('x', 'y')
            self._size = self._digits.target.size

    def __data_source__(self, rs, train):
        if train:
            return self.NaiveRegressionTrainDataSource(rs)
        else:
            return self.NaiveRegressionTestDataSource(rs)

    def __data_iterator__(self, data_source, batch_size=1):
        return data_iterator(data_source, batch_size)

    def __init__(self, batch, rs=666):
        self._data_source = self.__data_source__(rs, True)
        self._data_iterator = self.__data_iterator__(
            self._data_source, batch_size=batch)

        self._v_data_source = self.__data_source__(rs, False)
        self._v_data_iterator = self.__data_iterator__(
            self._v_data_source, batch_size=batch)

    def reset(self, train=True):
        if train:
            self._data_iterator._reset()
        else:
            self._v_data_iterator._reset()

    def next(self, train=True):
        if train:
            return self._data_iterator.next()
        else:
            return self._v_data_iterator.next()

    def size(self, train=True):
        if train:
            return self._data_iterator.size
        else:
            return self._v_data_iterator.size


class NaiveRegressionWorkerOne(framework.worker.Worker):
    '''
    A simple regression model which does not need to care about error.

    Args:
        data (:obj:`Data`): Data object.
        max_epoch (:obj:`int`): Max epoch to train.
        batch (:obj:`int`): Batch number of input.
        channel (:obj:`int`): Channel number of input.
        height (:obj:`int`): Height of input.
        width (:obj:`int`): Width of input.
        learning_rate (:obj:`float`): Learning rate.
        monitor_path (:obj:`str`): Monitor path.
        iter_per_epoch (:obj:`int`, optional): Iterations per one epoch. If not set, this value are determined by `tdata.size // tdata.batch_size`.
        val_iter (:obj:`int`, optional): Iterations for evaluation. If not set, this value are determined by `vdata.size // vdata.batch_size`.

    '''

    def _logistic_regression(self, image, test=False):
        return logistic_regression(image, test, '')

    def _var_init(self):
        self._tinput = nn.Variable(
            [self._batch, self._channel, self._height, self._width])
        self._tlabel = nn.Variable([self._batch, 1])

        self._vinput = nn.Variable(
            [self._batch, self._channel, self._height, self._width])
        self._vlabel = nn.Variable([self._batch, 1])

    def _comm_init(self):
        pass

    def _graph_init(self):
        self._tpred = self._logistic_regression(self._tinput)
        self._tpred = self._tpred.apply(persistent=True)
        self._vpred = self._logistic_regression(self._vinput, test=True)
        self._vpred = self._vpred.apply(persistent=True)

    def _loss_init(self):
        # Loss and error.
        self._loss = F.mean(
            F.softmax_cross_entropy(self._tpred, self._tlabel))
        self._vloss = F.mean(
            F.softmax_cross_entropy(self._vpred, self._vlabel))

    def _solver_init(self):
        self._solver = S.Sgd(self._learning_rate)
        self._solver.set_parameters(nn.get_parameters())

    def _monitor_init(self):
        # Monitors.
        from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
        monitor = Monitor(self._monitor_path)
        self._monitor_loss = MonitorSeries(
            "Training loss", monitor, interval=1)
        self._monitor_vloss = MonitorSeries("Valid loss", monitor, interval=1)
        self._monitor_time = MonitorTimeElapsed(
            "Training time", monitor, interval=10)

    def __init__(self, data, max_epoch, batch, channel, height, width, learning_rate, monitor_path, iter_per_epoch=None, val_iter=None):
        self._data = data
        self._max_epoch = max_epoch
        self._batch = batch
        self._channel = channel
        self._height = height
        self._width = width
        self._learning_rate = learning_rate
        self._monitor_path = monitor_path
        if iter_per_epoch is None:
            self._iter_per_epoch = self._data.size() // batch
        else:
            self._iter_per_epoch = iter_per_epoch
        if val_iter is None:
            self._val_iter = self._data.size(train=False) // batch
        else:
            self._val_iter = val_iter
        self._clear_buffer = True
        self._clear_no_need_grad = True

        self._updater = framework.updater.Updater(train_data_feeder=self._train_data_feeder,
                                                  solver_zero_grad=self._solver_zero_grad,
                                                  solver_update=self._solver_update,
                                                  loss_forward=self._loss_forward,
                                                  loss_backward=self._loss_backward,
                                                  accum_grad=1, comm=None, grads=[])
        self._evaluator = framework.evaluator.Evaluator(eval_data_feeder=self._eval_data_feeder,
                                                        eval_graph_forward=self._eval_graph_forward,
                                                        eval_callback_on_finish=[
                                                            self._eval_callback_on_finish],
                                                        val_iter=self._val_iter)

    def brief(self):
        return self._loss.d, self._vloss.d

    def iter_loop_end(self, i, e):
        self._monitor_loss.add(e * self._iter_per_epoch + i, self._loss.d)
        self._monitor_time.add(e * self._iter_per_epoch + i)

    def _train_data_feeder(self, i, e, ag):
        self._tinput.d, self._tlabel.d = self._data.next()

    def _eval_data_feeder(self, e, vi):
        self._vinput.d, self._vlabel.d = self._data.next(train=False)

    def _eval_graph_forward(self, e, vi):
        self._vloss.forward(clear_no_need_grad=self._clear_no_need_grad)

    def _eval_callback_on_finish(self, e):
        self._monitor_vloss.add(e * self._iter_per_epoch, self._vloss.d)


def naive_regression_training_without_framework(max_epoch, batch, channel, height, width, learning_rate, monitor_path):
    digits = load_digits()
    v_digits = load_digits()
    rs = 666
    digits.data, v_digits.data, digits.target, v_digits.target, digits.images, v_digits.images = train_test_split(
        digits.data, digits.target, digits.images, test_size=0.2, random_state=rs)
    data_size = digits.target.size
    v_data_size = v_digits.target.size
    iter_per_epoch = data_size // batch
    v_iter_per_epoch = v_data_size // batch

    def load_func(position):
        img = digits.images[position]
        label = digits.target[position]
        return img[None], np.array([label]).astype(np.int32)

    def v_load_func(position):
        img = v_digits.images[position]
        label = v_digits.target[position]
        return img[None], np.array([label]).astype(np.int32)

    di = data_iterator_simple(load_func, data_size, batch)
    v_di = data_iterator_simple(v_load_func, v_data_size, batch)

    tinput = nn.Variable([batch, channel, height, width])
    tlabel = nn.Variable([batch, 1])
    tpred = logistic_regression(tinput, name='-com')
    tpred = tpred.apply(persistent=True)

    vinput = nn.Variable([batch, channel, height, width])
    vlabel = nn.Variable([batch, 1])
    vpred = logistic_regression(vinput, test=True, name='-com')
    vpred = vpred.apply(persistent=True)

    solver = S.Sgd(learning_rate)
    solver.set_parameters(nn.get_parameters())

    monitor = Monitor(monitor_path + "-com")
    monitor_tloss = MonitorSeries("Training loss", monitor, interval=1)
    monitor_vloss = MonitorSeries("Valid loss", monitor, interval=1)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=10)

    tloss = F.mean(F.softmax_cross_entropy(tpred, tlabel))
    vloss = F.mean(F.softmax_cross_entropy(vpred, vlabel))

    for e in range(max_epoch):
        for i in range(v_iter_per_epoch):
            vinput.d, vlabel.d = v_di.next()
            vloss.forward(clear_no_need_grad=True)
            monitor_vloss.add(e * v_iter_per_epoch, vloss.d)
        for i in range(iter_per_epoch):
            tinput.d, tlabel.d = di.next()
            solver.zero_grad()
            tloss.forward(clear_no_need_grad=True)
            tloss.backward(clear_buffer=True)
            solver.update()
            monitor_tloss.add(e * iter_per_epoch + i, tloss.d)
            monitor_time.add(e * iter_per_epoch + i)
    for i in range(v_iter_per_epoch):
        vinput.d, vlabel.d = v_di.next()
        vloss.forward(clear_no_need_grad=True)
        monitor_vloss.add(max_epoch * v_iter_per_epoch, vloss.d)

    return tloss.d, vloss.d


@pytest.mark.parametrize('max_epoch', [2])
@pytest.mark.parametrize('batch', [80])
def test_naive_regression_training_one(max_epoch, batch, channel=3, height=8, width=8, learning_rate=0.01, monitor_path='tmp.monitor-reg'):
    data = NaiveRegressionDataOne(batch)
    worker = NaiveRegressionWorkerOne(
        data, max_epoch, batch, channel, height, width, learning_rate, monitor_path)
    trainer = framework.trainer.Trainer(worker)
    trainer.train()
    brief = trainer.brief()

    ref = naive_regression_training_without_framework(
        max_epoch, batch, channel, height, width, learning_rate, monitor_path)

    # Commpare the results.
    np.testing.assert_allclose(brief, ref, rtol=1e-06, atol=1e-06)
