# Copyright 2019,2020,2021 Sony Corporation.
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


import os

import nnabla as nn
import nnabla.functions as F
from nnabla.experimental.trainers import Evaluator
from nnabla.experimental.trainers import Updater
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from six.moves import range


class Trainer(object):
    '''Trainer API

    Trainer class is the very basic class for training neural network. You can composite this class to your own trainer class and delegate the train method of this class to your class.

    Args:    
        updater (:obj:`Updater` or list of :obj:`Updater`): Updater object.
        evaluator (:obj:`Evaluator` or list of :obj:`Evaluator`): Evaluator object.
        model_save_path (:obj:`str`): Model save path.
        max_epoch (:obj:`int`): Max epoch to train.
        iter_per_epoch (:obj:`int`, optional): Iterations per one epoch.
        callback_on_start (callable :obj:`object`, function, lambda, or list of these, optional): Callback called before the trainer.train.
        callback_on_finish (callable :obj:`object`, function, lambda, or list of these, optional): Callback called after the trainer.train.
        update_callback_on_start (callable :obj:`object`, function, lambda, or list of these, optional): Callback called before the updater.update.
        update_callback_on_finish (callable :obj:`object`, function, lambda, or list of these, optional): Callback called after the updater.update.

    The following example is a complete snippet to use this base trainer.

    Example:

        .. code-block:: python

            import nnabla as nn
            import nnabla.functions as F
            import nnabla.parametric_functions as PF
            import nnabla.solvers as S

            from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed

            import numpy as np

            from nnabla.experimental.trainers import Trainer, Updater, Evaluator

            # Batch, channel, height, width
            b, c, h, w = 32, 1, 128, 128

            # Train Input
            tinput = nn.Variable([b, c, h, w])
            tlabel = nn.Variable([b, c, h, w])

            # Train Model and Loss
            tpred = <training model>.apply(persistent=True)
            tloss = F.mean(F.softmax_cross_entropy(tpred, tlabel))

            # Test Input
            vinput = nn.Variable([b, c, h, w])
            vlabel = nn.Variable([b, c, h, w])

            # Test Model and Error
            vpred = <evaluation model>.apply(persistent=True)
            vloss = F.mean(F.softmax_cross_entropy(vpred, vlabel))
            verror = F.mean(F.top_n_error(vpred.get_unlinked_variable(), vlabel))

            # Solver
            solver = S.Adam()
            solver.set_parameters(nn.get_parameters())

            # DataIterator
            tdata = <training_data_iterator>
            vdata = <validation_data_iterator>

            # Monitor
            monitor = Monitor(<monitor_path>)
            monitor_loss = MonitorSeries("Training loss", monitor, interval=10)
            monitor_err = MonitorSeries("Training error", monitor, interval=10)
            monitor_time = MonitorTimeElapsed("Training time", monitor, interval=100)
            monitor_verr = MonitorSeries("Valid error", monitor, interval=10)

            # Updater
            def tdata_feeder():
                tinput.d, tlabel.d = tdata.next()
            def update_callback_on_finish(i):
                monitor_loss.add(i, tloss.d)
                monitor_time.add(i)
            updater = Updater(solver, tloss, 
                              data_feeder=tdata_feeder, 
                              update_callback_on_finish=update_callback_on_finish)

            # Evaluator
            def vdata_feeder():
                vinput.d, vlabel.d = vdata.next()
            def eval_callback_on_finish(i, ve):
                monitor_verr.add(i, ve)
            evaluator = Evaluator(verror, 
                                  data_feeder=vdata_feeder, 
                                  val_iter=vdata.size // b, 
                                  callback_on_finish=eval_callback_on_finish)

            # Trainer
            trainer = Trainer(updater, evaluator, <model_save_path>, 
                              max_epoch=<max_epoch>, iter_per_epoch=tdata.size // b)
            trainer.train()
    '''

    def _force_to_list(self, x):
        if type(x) is list:
            return x
        else:
            return [x]

    def __init__(self, updater=None, evaluator=None,
                 model_save_path=None,
                 max_epoch=1,
                 iter_per_epoch=None,
                 callback_on_start=lambda: True,
                 callback_on_finish=lambda: True,
                 update_callback_on_start=lambda i: True,
                 update_callback_on_finish=lambda i: True):
        """
        """
        self.updater = self._force_to_list(updater)
        self.model_save_path = model_save_path
        self.evaluator = self._force_to_list(evaluator)
        self.max_epoch = max_epoch
        self.iter_per_epoch = iter_per_epoch
        self.callback_on_start = self._force_to_list(callback_on_start)
        self.callback_on_finish = self._force_to_list(callback_on_finish)
        self.update_callback_on_start = self._force_to_list(
            update_callback_on_start)
        self.update_callback_on_finish = self._force_to_list(
            update_callback_on_finish)

    def train(self):
        # On-start callback
        for callback in self.callback_on_start:
            callback()

        # Training loop
        for e in range(self.max_epoch):
            for j in range(self.iter_per_epoch):
                i = e * self.iter_per_epoch + j
                # On-start callback
                for callback in self.update_callback_on_start:
                    callback(i)
                # Update
                for updater in self.updater:
                    updater.update(i)
                # On-finish callback
                for callback in self.update_callback_on_finish:
                    callback(i)

            # Save parameters
            if self.model_save_path is not None:
                nn.save_parameters(os.path.join(
                    self.model_save_path, 'params_{:06}.h5'.format(i)))

            # Evaluate
            for evaluator in self.evaluator:
                evaluator.evaluate(i)

        # On-finish callback
        for callback in self.callback_on_finish:
            callback()

        # Save parameters
        if self.model_save_path is not None:
            nn.save_parameters(os.path.join(
                self.model_save_path, 'params_{:06}.h5'.format(i)))

        # Evaluate
        for evaluator in self.evaluator:
            evaluator.evaluate(i)


class NaiveClassificationTrainer(object):
    '''Naive Classification Trainer

    Args:
        solver (:obj:`~nnabla.solver.Solver`): Solver object.
        tinput (:obj:`~nnabla.Variable`): Input variable for input feature in training.
        tlabel (:obj:`~nnabla.Variable`): Label variable for lable in training.
        tpred (:obj:`~nnabla.Variable`): Root variable for prediction in the training graph.
        tdata (:obj:`nnabla.utils.data_iterator.DataIterator`): DataIterator for training.
        vinput (:obj:`~nnabla.Variable`): Input variable for input feature in evaluation.
        vlabel (:obj:`~nnabla.Variable`): Label variable for label in evaluation.
        vpred (:obj:`~nnabla.Variable`): Root variable for prediction in the evaluation graph.
        vdata (:obj:`~nnabla.utils.data_iterator.DataIterator`): DataIterator for evaluation.
        monitor_path (:obj:`str`): Monitor path.
        model_save_path (:obj:`str`): Model save path.
        max_epoch (:obj:`int`): Max epoch to train.
        iter_per_epoch (:obj:`int`, optional): Iterations per one epoch. If not set, this value are determined by `tdata.size // tdata.batch_size`.
        val_iter (:obj:`int`, optional): Iterations for evaluation. If not set, this value are determined by `vdata.size // vdata.batch_size`.


    The following example is a complete snippet to use this base trainer.

    Example:

        .. code-block:: python

            import nnabla as nn
            import nnabla.functions as F
            import nnabla.parametric_functions as PF
            import nnabla.solvers as S

            import numpy as np

            from nnabla.experimental.trainers import NaiveClassificationTrainer

            # Batch, channel, height, width
            b, c, h, w = 32, 1, 128, 128

            # Train Input
            tinput = nn.Variable([b, c, h, w])
            tlabel = nn.Variable([b, c, h, w])

            # Train Model and Loss
            tpred = <training model>

            # Test Input
            vinput = nn.Variable([b, c, h, w])

            # Test Model
            vpred = <evaluation model>

            # Solver
            solver = S.Adam()
            solver.set_parameters(nn.get_parameters())

            # DataIterator
            tdata = <training_data_iterator>
            vdata = <validation_data_iterator>

            # Trainer
            trainer = NaiveClassificationTrainer(solver, 
                                                 tinput, tlabel, tpred, tdata, 
                                                 vinput, vlabel, vpred, vdata, 
                                                 <monitor_path>, 
                                                 <model_save_path>, 
                                                 max_epoch=<max_epoch>)
            trainer.train()
    '''

    def __init__(self, solver,
                 tinput=None, tlabel=None, tpred=None, tdata=None,
                 vinput=None, vlabel=None, vpred=None, vdata=None,
                 monitor_path=None, model_save_path=None,
                 max_epoch=1, iter_per_epoch=None,
                 val_iter=None):
        # Monitors
        monitor = Monitor(monitor_path)
        monitor_loss = MonitorSeries("Training loss", monitor, interval=10)
        monitor_err = MonitorSeries("Training error", monitor, interval=10)
        monitor_vloss = MonitorSeries("Valid loss", monitor, interval=1)
        monitor_verr = MonitorSeries("Valid error", monitor, interval=1)
        monitor_time = MonitorTimeElapsed(
            "Training time", monitor, interval=10)

        # Loss and error
        tpred = tpred.apply(persistent=True)
        tloss = F.mean(F.softmax_cross_entropy(tpred, tlabel))
        terror = F.mean(F.top_n_error(tpred.get_unlinked_variable(), tlabel))
        vpred = vpred.apply(persistent=True)
        vloss = F.mean(F.softmax_cross_entropy(vpred, vlabel))
        verror = F.mean(F.top_n_error(vpred.get_unlinked_variable(), vlabel))

        # Updater
        def tdata_feeder():
            tinput.d, tlabel.d = tdata.next()

        def forward_callback_on_finish(i):
            terror.forward()

        def update_callback_on_finish(i):
            monitor_loss.add(i, tloss.d)
            monitor_err.add(i, terror.d)
            monitor_time.add(i)
        updater = Updater(solver, tloss,
                          data_feeder=tdata_feeder,
                          forward_callback_on_finish=forward_callback_on_finish,
                          update_callback_on_finish=update_callback_on_finish)

        # Evaluator
        def vdata_feeder():
            vinput.d, vlabel.d = vdata.next()

        def vloss_callback_on_finish(i, v):
            monitor_vloss.add(i, v)

        def verror_callback_on_finish(i, v):
            monitor_verr.add(i, v)
        val_iter = val_iter if val_iter is not None else vdata.size // vdata.batch_size
        evaluator = Evaluator([vloss, verror],
                              data_feeder=vdata_feeder,
                              val_iter=val_iter,
                              callback_on_finish=[vloss_callback_on_finish, verror_callback_on_finish])

        # Trainer
        iter_per_epoch = iter_per_epoch if iter_per_epoch is not None \
            else tdata.size // tdata.batch_size
        self.trainer = Trainer(updater, evaluator,
                               model_save_path,
                               max_epoch=max_epoch, iter_per_epoch=iter_per_epoch)

    def train(self):
        self.trainer.train()


class NaiveRegressionTrainer(object):
    '''Naive Regression Trainer

    Args:
        solver (:obj:`~nnabla.solver.Solver`): Solver object.
        tinput (:obj:`~nnabla.Variable`): Input variable for input feature in training.
        tlabel (:obj:`~nnabla.Variable`): Label variable for lable in training.
        tpred (:obj:`~nnabla.Variable`): Root variable for prediction in the training graph.
        tdata (:obj:`nnabla.utils.data_iterator.DataIterator`): DataIterator for training.
        vinput (:obj:`~nnabla.Variable`): Input variable for input feature in evaluation.
        vlabel (:obj:`~nnabla.Variable`): Label variable for label in evaluation.
        vpred (:obj:`~nnabla.Variable`): Root variable for prediction in the evaluation graph.
        vdata (:obj:`~nnabla.utils.data_iterator.DataIterator`): DataIterator for evaluation.
        monitor_path (:obj:`str`): Monitor path.
        model_save_path (:obj:`str`): Model save path.
        max_epoch (:obj:`int`): Max epoch to train.
        iter_per_epoch (:obj:`int`, optional): Iterations per one epoch. If not set, this value are determined by `tdata.size // tdata.batch_size`.
        val_iter (:obj:`int`, optional): Iterations for evaluation. If not set, this value are determined by `vdata.size // vdata.batch_size`.


    Example:

        .. code-block:: python

            import nnabla as nn
            import nnabla.functions as F
            import nnabla.parametric_functions as PF
            import nnabla.solvers as S

            import numpy as np

            from nnabla.experimental.trainers import NaiveRegressionTrainer

            # Batch, channel, height, width
            b, c, h, w = 32, 1, 128, 128

            # Train Input
            tinput = nn.Variable([b, c, h, w])
            tlabel = nn.Variable([b, c, h, w])

            # Train Model and Loss
            tpred = <training model>

            # Test Input
            vinput = nn.Variable([b, c, h, w])
            vlabel = nn.Variable([b, c, h, w])

            # Test Model
            vpred = <evaluation model>

            # Solver
            solver = S.Adam()
            solver.set_parameters(nn.get_parameters())

            # DataIterator
            tdata = <training_data_iterator>
            vdata = <validation_data_iterator>

            # Trainer
            trainer = NaiveRegressionTrainer(solver, 
                                             tinput, tlabel, tpred, tdata, 
                                             vinput, vlabel, vpred, vdata, 
                                             <monitor_path>, 
                                             <model_save_path>, 
                                             max_epoch=<max_epoch>)
            trainer.train()

    '''

    def __init__(self, solver,
                 tinput=None, tlabel=None, tpred=None, tdata=None,
                 vinput=None, vlabel=None, vpred=None, vdata=None,
                 monitor_path=None, model_save_path=None,
                 max_epoch=1, iter_per_epoch=None,
                 val_iter=None):
        # Monitors
        monitor = Monitor(monitor_path)
        monitor_loss = MonitorSeries("Training loss", monitor, interval=10)
        monitor_vloss = MonitorSeries("Valid loss", monitor, interval=1)
        monitor_time = MonitorTimeElapsed(
            "Training time", monitor, interval=10)

        # Loss
        tpred = tpred.apply(persistent=True)
        tloss = F.mean(F.squared_error(tpred, tlabel))
        vpred = vpred.apply(persistent=True)
        vloss = F.mean(F.squared_error(vpred, vlabel))

        # Updater
        def tdata_feeder():
            tinput.d, tlabel.d = tdata.next()

        def update_callback_on_finish(i):
            monitor_loss.add(i, tloss.d)
            monitor_time.add(i)
        updater = Updater(solver, tloss,
                          data_feeder=tdata_feeder,
                          forward_callback_on_finish=forward_callback_on_finish,
                          update_callback_on_finish=update_callback_on_finish)

        # Evaluator
        def vdata_feeder():
            vinput.d, vlabel.d = vdata.next()

        def vloss_callback_on_finish(i, v):
            monitor_vloss.add(i, v)
        val_iter = val_iter if val_iter is not None else vdata.size // vdata.batch_size
        evaluator = Evaluator(vloss,
                              data_feeder=vdata_feeder,
                              val_iter=val_iter,
                              callback_on_finish=vloss_callback_on_finish)

        # Trainer
        iter_per_epoch = iter_per_epoch if iter_per_epoch is not None \
            else tdata.size // tdata.batch_size
        self.trainer = Trainer(updater, evaluator,
                               model_save_path,
                               max_epoch=max_epoch, iter_per_epoch=iter_per_epoch)

    def train(self):
        self.trainer.train()
