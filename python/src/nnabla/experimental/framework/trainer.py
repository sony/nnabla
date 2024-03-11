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


class Trainer(object):
    '''
    Trainer class definition. Common for all training process.
    An instance of Worker is required input.
    Custom epoch number and iteration number need to be specified.

    Args:
        worker (:obj:`Worker`): Worker object.

    Example:

        .. code-block:: python

        from nnabla.experimental.framwork import Data
        from nnabla.experimental.framwork import Worker
        from nnabla.experimental.framwork import Trainer

        data = Data()
        worker = Worker(data)
        trainer = Trainer(worker)

        trainer.train()
    '''

    def __init__(self,
                 worker):
        self.worker = worker
        self.start_epoch = 0

    def train(self):
        '''
        Common training process.

        Args:
            /
        '''

        # load checkpoint if user defined
        self.start_epoch = self.worker.load_checkpoint()
        # training prepare
        self.worker.training_prepare()
        # save model if needed
        self.worker.save_model_at_start()
        # training loop
        for e in range(self.start_epoch, (self.start_epoch + self.worker.max_epoch())):
            # epoch loop start
            self.worker.epoch_loop_start(e)
            # evaluate
            self.worker.evaluate_at_epoch_start(e)
            # iterations per epoch
            for i in range(self.worker.iter_per_epoch()):
                # iteration loop start
                self.worker.iter_loop_start(i, e)
                # update
                self.worker.update_in_iteration(i, e)
                # iteration loop end
                self.worker.iter_loop_end(i, e)
            # epoch loop end
            self.worker.epoch_loop_end(e)
            # save checkpoint
            self.worker.save_checkpoint_at_epoch_end(e)
        # evaluate
        self.worker.evaluate_at_training_end()
        # save model at last
        self.worker.save_model_at_end()
        # training complete
        self.worker.training_complete()

    def brief(self):
        return self.worker.brief()
