# Copyright 2023 Sony Group Corporation.
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


import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save

from .updater import Updater
from .evaluator import Evaluator


class Worker(object):
    '''
    Base class.
    Functions and mechanisms defined to be executed at different steps of the training process.
    An instance of Data is a required input.

    Args:
        data (:obj:`Data`): Data object.

    Example:

        .. code-block:: python

        class MnistWorker(Worker):
            """
            Nnist worker.

            Args:
                data (:obj:`Data`): Data object.
                args: A structure parsed from configuration file or command line flags.
            """

            import os
            import numpy as np

            from nnabla.framework.nnabla_data import MnistData
            from neu.checkpoint_util import save_checkpoint as neu_save_checkpoint
            from neu.checkpoint_util import load_checkpoint as neu_load_checkpoint
            from neu.save_nnp import save_nnp

            def _categorical_error(self, pred, label):
                """
                Compute categorical error given score vectors and labels as numpy.ndarray.
                """
                pred_label = pred.argmax(1)
                return (pred_label != label.flat).mean()

            def _augmentation(self, h, test, aug):
                if aug is None:
                    aug = not test
                if aug:
                    h = F.image_augmentation(h, (1, 28, 28), (0, 0), 0.9, 1.1, 0.3,
                                            1.3, 0.1, False, False, 0.5, False, 1.5, 0.5, False, 0.1, 0)
                return h

            def _mnist_lenet_prediction(self, image, test=False, aug=None):
                """
                Construct LeNet for MNIST.
                """
                image /= 255.0
                image = self._augmentation(image, test, aug)
                c1 = PF.convolution(image, 16, (5, 5), name='conv1')
                c1 = F.relu(F.max_pooling(c1, (2, 2)))
                c2 = PF.convolution(c1, 16, (5, 5), name='conv2')
                c2 = F.relu(F.max_pooling(c2, (2, 2)))
                c3 = F.relu(PF.affine(c2, 50, name='fc3'))
                c4 = PF.affine(c3, 10, name='fc4')
                return c4

            def _mnist_resnet_prediction(self, image, test=False, aug=None):
                """
                Construct ResNet for MNIST.
                """
                image /= 255.0
                image = self._augmentation(image, test, aug)

                def bn(x):
                    return PF.batch_normalization(x, batch_stat=not test)

                def res_unit(x, scope):
                    C = x.shape[1]
                    with nn.parameter_scope(scope):
                        with nn.parameter_scope('conv1'):
                            h = F.elu(
                                bn(PF.convolution(x, C / 2, (1, 1), with_bias=False)))
                        with nn.parameter_scope('conv2'):
                            h = F.elu(
                                bn(PF.convolution(h, C / 2, (3, 3), pad=(1, 1), with_bias=False)))
                        with nn.parameter_scope('conv3'):
                            h = bn(PF.convolution(h, C, (1, 1), with_bias=False))
                    return F.elu(F.add2(h, x))
                # Conv1 --> 64 x 32 x 32
                with nn.parameter_scope("conv1"):
                    c1 = F.elu(
                        bn(PF.convolution(image, 64, (3, 3), pad=(3, 3), with_bias=False)))
                # Conv2 --> 64 x 16 x 16
                c2 = F.max_pooling(res_unit(c1, "conv2"), (2, 2))
                # Conv3 --> 64 x 8 x 8
                c3 = F.max_pooling(res_unit(c2, "conv3"), (2, 2))
                # Conv4 --> 64 x 8 x 8
                c4 = res_unit(c3, "conv4")
                # Conv5 --> 64 x 4 x 4
                c5 = F.max_pooling(res_unit(c4, "conv5"), (2, 2))
                # Conv5 --> 64 x 4 x 4
                c6 = res_unit(c5, "conv6")
                pl = F.average_pooling(c6, (4, 4))
                with nn.parameter_scope("classifier"):
                    y = PF.affine(pl, 10)
                return y

            def _var_init(self):
                # Create training input variables.
                self._image = nn.Variable([self._args.batch_size, 1, 28, 28])
                self._label = nn.Variable([self._args.batch_size, 1])

                # Create testing input variables.
                self._vimage = nn.Variable([self._args.batch_size, 1, 28, 28])
                self._vlabel = nn.Variable([self._args.batch_size, 1])

            def _comm_init(self):
                pass

            def _graph_init(self):
                # Create CNN network for both training and testing.
                if self._args.net == 'lenet':
                    self._mnist_cnn_prediction = self._mnist_lenet_prediction
                elif self._args.net == 'resnet':
                    self._mnist_cnn_prediction = self._mnist_resnet_prediction
                else:
                    raise ValueError("Unknown network type {}".format(self.args.net))

                # Create training prediction graph.
                self._pred = self._mnist_cnn_prediction(
                    self._image, test=False, aug=self._args.augment_train)
                self._pred.persistent = True

                # Create testing prediction graph.
                self._vpred = self._mnist_cnn_prediction(
                    self._vimage, test=True, aug=self._args.augment_test)

                # Create evaluation metrics value list, length one for verr only
                self._v_list = [0.0]

            def _loss_init(self):
                # Create loss function.
                self._loss = F.mean(F.softmax_cross_entropy(self._pred, self._label))

            def _solver_init(self):
                # Create Solver. If training from checkpoint, load the info.
                self._solver = S.Adam(self._args.learning_rate)
                self._solver.set_parameters(nn.get_parameters())

            def _monitor_init(self):
                # Create monitor.
                from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
                monitor = Monitor(self._args.monitor_path)
                self._monitor_loss = MonitorSeries(
                    "Training loss", monitor, interval=10)
                self._monitor_err = MonitorSeries(
                    "Training error", monitor, interval=10)
                self._monitor_time = MonitorTimeElapsed(
                    "Training time", monitor, interval=100)
                self._monitor_verr = MonitorSeries("Test error", monitor, interval=10)

            def __init__(self, data, args):
                from numpy.random import seed
                seed(0)
                # Get context.
                from nnabla.ext_utils import get_extension_context
                logger.info("Running in %s" % args.context)
                self._ctx = get_extension_context(
                    args.context, device_id=args.device_id, type_config=args.type_config)
                nn.set_default_context(self.ctx)
                self._args = args

                self._data = data
                self._max_epoch = args.max_epoch
                self._iter_per_epoch = args.iter_per_epoch
                self._clear_buffer = args.clear_buffer

                self._updater = Updater(train_data_feeder=self._train_data_feeder,
                                            solver_zero_grad=self._solver_zero_grad,
                                            solver_update=self._solver_update,
                                            loss_forward=self._loss_forward,
                                            accum_grad=1, comm=None, grads=[])
                self._evaluator = Evaluator(eval_data_feeder=self._eval_data_feeder,
                                            eval_graph_forward=self._eval_graph_forward,
                                            eval_callback_on_start=[self._eval_callback_on_start],
                                            eval_callback_on_finish=[self._eval_callback_on_finish],
                                            val_iter=self._args.val_iter)

            def iter_loop_end(self, i, e):
                self._loss.data.cast(self.np.float32, self._ctx)
                self._pred.data.cast(self.np.float32, self._ctx)
                err = self._categorical_error(self._pred.d, self._label.d)
                self._monitor_loss.add(e + i, self._loss.d.copy())
                self._monitor_err.add(e + i, err)
                self._monitor_time.add(e + i)

            def load_checkpoint(self):
                start_point = 0
                if self._args.checkpoint is not None:
                    # load weights and solver state info from specified checkpoint file.
                    start_point = self.neu_load_checkpoint(
                        self._args.checkpoint, self._solver)
                return start_point

            def save_checkpoint_at_epoch_end(self, e):
                if e % self._args.model_save_interval == 0:
                    # save checkpoint file
                    self.neu_save_checkpoint(
                        self._args.model_save_path, e, self._solver)

            def evaluate_at_epoch_start(self, e):
                if self._loopCallbackCondition(e=e, toExecute=lambda i, e, ag: (e % self._args.val_interval == 0)):
                    self._evaluator.evaluate(e)

            def evaluate_at_training_end(self):
                self._evaluator.evaluate(self._max_epoch)

            def save_model_at_start(self):
                # save_nnp
                contents = self.save_nnp({'x': self._vimage}, {
                                        'y': self._vpred}, self._args.batch_size)
                save.save(self.os.path.join(self._args.model_save_path,
                                            '{}_result_epoch0.nnp'.format(self._args.net)), contents)

            def save_model_at_end(self):
                # save_nnp_lastepoch
                contents = self.save_nnp({'x': self._vimage}, {
                                        'y': self._vpred}, self._args.batch_size)
                save.save(self.os.path.join(self._args.model_save_path,
                                            '{}_result.nnp'.format(self._args.net)), contents)

            def _train_data_feeder(self, i, e, ag):
                if self._loopCallbackCondition(i=i, e=e, ag=ag):
                    self._image.d, self._label.d = self._data.next()

            def _solver_update_callback_on_start(self, i, e):
                if self._loopCallbackCondition(i=i, e=e):
                    self._solver.weight_decay(self._args.weight_decay)

            def _eval_callback_on_start(self, e):
                if self._loopCallbackCondition(e=e):
                    self._v_list[0] = 0.0

            def _eval_data_feeder(self, e, vi):
                if self._loopCallbackCondition(e=e, ag=vi):
                    self._vimage.d, self._vlabel.d = self._data.next(False)

            def _eval_graph_forward(self, e, vi):
                if self._loopCallbackCondition(i=e, ag=vi):
                    self._vpred.forward(clear_buffer=self._clear_buffer)
                    self._vpred.data.cast(self.np.float32, self._ctx)
                    self._v_list[0] += self._categorical_error(self._vpred.d, self._vlabel.d)

            def _eval_callback_on_finish(self, e):
                if self._loopCallbackCondition(e=e):
                    self._monitor_verr.add(e, self._v_list[0] / self._args.val_iter)

        data = MnistData(shuffle, rng, batch_size)
        worker = MnistWorker(data, batch_size, max_epoch, net)
    '''

    # utilities

    def _load_parameters(self):
        pass

    def _save_parameters(self):
        pass

    def _load_model(self):
        pass

    def _save_model(self):
        pass

    def _loopCallbackCondition(self, i=0, e=0, ag=0, toExecute=lambda i, e, ag: True):
        '''
        Determine whether the context of the callback function would be executed in this loop.
        Since a list of callback functions can be registered into Updater and Evaluator, each may have different execution conditions according to epoch and iteration number.
        It is able to implement mutual exclusive conditions for callback functions.
        It is also able to execute the registered function in a certain interval.
        No-need-to-change method.

        Args:
            i (:obj:`int`, optional, default: 0): Iteration number.
            e (:obj:`int`, optional, default: 0): Epoch number.
            ag (:obj:`int`, optional, default: 0): Accumulated gradiant index number, may also be used for val_iter.
            toExecute (callable :obj:`object`, function, lambda, optional): Check whether to execute the callback's context.
        '''

        return toExecute(i, e, ag)

    def _force_to_list(self, x):
        '''
        No-need-to-change method.
        '''

        if type(x) is list:
            return x
        else:
            return [x]

    # initializers for training preparation

    def _ctx_init(self):
        '''
        Define context of the training.

        Optional method, default context is "cpu".

        Example:

            .. code-block:: python

            import nnabla as nn

            from nnabla.ext_utils import get_extension_context
            ctx = get_extension_context("cudnn")
            nn.set_default_context(ctx)
        '''

        pass

    def _var_init(self):
        '''
        Define variables of the training.

        Mandatory method.
        '''

        pass

    def _comm_init(self):
        '''
        Define communicator of the training.

        Optional method.
        '''

        # self._comm

        pass

    def _data_init(self):
        '''
        Data preparation through Data's interfaces.

        Optional method.
        '''

        pass

    def _graph_init(self):
        '''
        Define network graph of the training and evaluation.
        Initialize metrics list for evaluations.

        Mandatory method, do override it.
        '''

        # self._train_graph

        # self._eval_graph
        # self._eval_num = len(self._eval_graph)
        # self._v_list = [0.0] * self._eval_num

        pass

    def _loss_init(self):
        '''
        Define loss function of the training.
        Define error function of the training.

        Mandatory method, do override it.
        '''

        # self._loss
        # self._err

        pass

    def _solver_init(self):
        '''
        Define solver function of the training.

        Mandatory method, do override it.
        '''

        # self._solver

        pass

    def _monitor_init(self):
        '''
        Define monitors of the training.

        Optional method.
        '''

        pass

    # object initializer

    def __init__(self, data, max_epoch=1, iter_per_epoch=1, clear_buffer=True, clear_no_need_grad=True):
        '''
        Initializer of the object.

        Mandatory method, default definition can be used or overridden.
        '''

        self._data = data

        self._max_epoch = max_epoch
        self._iter_per_epoch = iter_per_epoch
        self._clear_buffer = clear_buffer
        self._clear_no_need_grad = clear_no_need_grad

        self._updater = Updater(train_data_feeder=self._train_data_feeder,
                                solver_zero_grad=self._solver_zero_grad,
                                solver_update=self._solver_update,
                                loss_forward=self._loss_forward,
                                loss_backward=self._loss_backward,
                                loss_forward_callback_on_start=[
                                          self._loss_forward_callback_on_start],
                                loss_forward_callback_on_finish=[
                                          self._loss_forward_callback_on_finish],
                                loss_backward_callback_on_start=[
                                          self._loss_backward_callback_on_start],
                                loss_backward_callback_on_finish=[
                                          self._loss_backward_callback_on_finish],
                                comm_all_reduce_callback_on_start=[
                                          self._comm_all_reduce_callback_on_start],
                                comm_all_reduce=self._comm_all_reduce,
                                comm_all_reduce_callback_on_finish=[
                                          self._comm_all_reduce_callback_on_finish],
                                solver_update_callback_on_start=[
                                          self._solver_update_callback_on_start],
                                solver_update_callback_on_finish=[
                                          self._solver_update_callback_on_finish],
                                accum_grad=1, comm=None, grads=[])
        self._evaluator = Evaluator(eval_data_feeder=self._eval_data_feeder,
                                    eval_graph_forward=self._eval_graph_forward,
                                    eval_callback_on_start=[
                                              self._eval_callback_on_start],
                                    eval_callback_on_finish=[
                                              self._eval_callback_on_finish],
                                    val_iter=1)

    # APIs for training

    def max_epoch(self):
        '''
        Get max epoch number.

        No-need-to-change method.
        '''

        return self._max_epoch

    def iter_per_epoch(self):
        '''
        Get iteration number per epoch.

        No-need-to-change method.
        '''

        return self._iter_per_epoch

    def training_prepare(self):
        '''
        All preparations and configurations need to be done before training start.
        Parse configurations, define variables, define solvers, define losses, define monitors, etc.

        No-need-to-change method.
        '''

        self._ctx_init()
        self._var_init()
        self._comm_init()
        self._data_init()
        self._graph_init()
        self._loss_init()
        self._solver_init()
        self._monitor_init()

    def training_complete(self):
        '''
        Record the training result, summarize the training process, reset environment, garbage collection, etc.

        Optional method, use default definition.
        '''

        pass

    def load_checkpoint(self):
        '''
        Load saved checkpoint, returns epoch number of the checkpoint.

        Optional method, use default definition.
        '''

        return 0

    def brief(self):
        '''
        Get a brief description and result of the training.

        Optional method, use default definition.
        '''

        pass

    def epoch_loop_start(self, e):
        '''
        Executed at the very beginning of each epoch.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(e=e):
            pass

    def epoch_loop_end(self, e):
        '''
        Executed at the end of each training epoch.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(e=e):
            pass

    def save_checkpoint_at_epoch_end(self, e):
        '''
        Save checkpoint at the very end of an epoch.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(e=e):
            pass

    def iter_loop_start(self, i, e):
        '''
        Executed at the very beginning of each iteration.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(i=i, e=e):
            pass

    def iter_loop_end(self, i, e):
        '''
        Executed at the very end of each iteration.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(i=i, e=e):
            pass

    def update_in_iteration(self, i, e):
        '''
        Updater's working loop.
        Including: zero_grad, forward, backward, all_reduce, update, etc.

        No-need-to-change method.
        '''

        if self._loopCallbackCondition(i=i, e=e):
            self._updater.update(i, e)

    def evaluate_at_epoch_start(self, e):
        '''
        Modle evaluation at the beginning of each epoch before updating.

        No-need-to-change method.
        '''

        if self._loopCallbackCondition(e=e):
            self._evaluator.evaluate(e)

    def evaluate_at_training_end(self):
        '''
        Modle evaluation at the end of whole training.

        No-need-to-change method.
        '''

        self._evaluator.evaluate(self._max_epoch)

    def save_model_at_start(self):
        '''
        Save model at the beginning.

        Optional method, use default definition.
        '''

        pass

    def save_model_at_end(self):
        '''
        Save model at the end.

        Optional method, use default definition.
        '''

        pass

    # callbacks for updater

    def _train_data_feeder(self, i, e, ag):
        '''
        Get next batch of training data.

        Mandatory method, do override it.
        '''

        if self._loopCallbackCondition(i=i, e=e, ag=ag):
            pass

    def _solver_zero_grad(self, i, e):
        '''
        Reset solver's gradient before handling a new batch of data.

        No-need-to-change method.
        '''

        if self._loopCallbackCondition(i=i, e=e):
            self._solver.zero_grad()

    def _loss_forward_callback_on_start(self, i, e, ag):
        '''
        Executed right before loss forward.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(i=i, e=e, ag=ag):
            pass

    def _loss_forward(self, i, e, ag):
        '''
        Loss forward.

        No-need-to-change method.
        '''

        if self._loopCallbackCondition(i=i, e=e, ag=ag):
            if self._loss is not None:
                self._loss.forward(clear_no_need_grad=self._clear_no_need_grad)

    def _loss_forward_callback_on_finish(self, i, e, ag):
        '''
        Executed right after loss forward.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(i=i, e=e, ag=ag):
            pass

    def _loss_backward_callback_on_start(self, i, e, ag):
        '''
        Executed right before loss backward.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(i=i, e=e, ag=ag):
            pass

    def _loss_backward(self, i, e, ag):
        '''
        Loss backward.

        No-need-to-change method.
        '''

        if self._loopCallbackCondition(i=i, e=e, ag=ag):
            if self._loss is not None:
                self._loss.backward(clear_buffer=self._clear_buffer)

    def _loss_backward_callback_on_finish(self, i, e, ag):
        '''
        Executed right after loss backward.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(i=i, e=e, ag=ag):
            pass

    def _comm_all_reduce_callback_on_start(self, i, e):
        '''
        Executed right before communicator all reduce.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(i=i, e=e):
            pass

    def _comm_all_reduce(self, i, e):
        '''
        Communicator all reduce.

        No-need-to-change method.
        '''

        if self._loopCallbackCondition(i=i, e=e):
            if self._comm is not None:
                self._comm.all_reduce(
                    self.grads, division=False, inplace=False)

    def _comm_all_reduce_callback_on_finish(self, i, e):
        '''
        Executed right after communicator all reduce.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(i=i, e=e):
            pass

    def _solver_update_callback_on_start(self, i, e):
        '''
        Executed right before solver update.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(i=i, e=e):
            pass

    def _solver_update(self, i, e):
        '''
        Solver update.

        No-need-to-change method.
        '''

        if self._loopCallbackCondition(i=i, e=e):
            if self._solver is not None:
                self._solver.update()

    def _solver_update_callback_on_finish(self, i, e):
        '''
        Executed right after solver update.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(i=i, e=e):
            pass

    # callbacks for evaluator

    def _eval_callback_on_start(self, e):
        '''
        Executed at the very beginning of an evaluation.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(e=e):
            pass

    def _eval_data_feeder(self, e, vi):
        '''
        Get next batch of evaluating data.

        Mandatory method, do override it.
        '''

        if self._loopCallbackCondition(e=e, ag=vi):
            pass

    def _eval_graph_forward(self, e, vi):
        '''
        Do evaluation forward.

        Mandatory method, do override it.
        '''

        if self._loopCallbackCondition(e=e, ag=vi):
            pass

    def _eval_callback_on_finish(self, e):
        '''
        Executed at the very end of an evaluation.
        Monitors' jobs may be put here.

        Optional method, use default definition.
        '''

        if self._loopCallbackCondition(e=e):
            pass
