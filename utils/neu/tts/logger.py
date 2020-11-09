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

from collections import OrderedDict
from pathlib import Path
import sys


class ProgressMeter(object):
    r"""A Progress Meter.

        Args:
            num_batches(int): The number of batches per epoch.
            path(str, optional): Path to save tensorboard and log file.
                Defaults to None.
            quiet(bool, optional): If quite == True, no message will be shown.
    """

    def __init__(self, num_batches, path=None, quiet=False):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = OrderedDict()
        self.terminal = sys.stdout
        self.quiet = quiet
        if not self.quiet:
            self.file = open(Path(path) / 'log.txt', 'w')

    def info(self, message, view=True):
        r"""Shows a message.

        Args:
            message(str): The message.
            view(bool, optional): If shows to terminal. Defaults to True.
        """
        if view and not self.quiet:
            self.terminal.write(message)
            self.terminal.flush()
        if not self.quiet:
            self.file.write(message)
            self.file.flush()

    def display(self, batch, key=None):
        r"""Displays current values for meters.

        Args:
            batch(int): The number of batch.
            key([type], optional): [description]. Defaults to None.
        """

        entries = [self.batch_fmtstr.format(batch)]
        key = key or [m.name for m in self.meters.values()]
        entries += [str(meter) for meter in self.meters.values()
                    if meter.name in key]
        self.info('\t'.join(entries) + '\n')

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, tag, value, n=1):
        r"""Updates the meter.

        Args:
            tag(str): The tag name.
            value(number): The value to update.
            n(int, optional): The len of minibatch. Defaults to 1.
        """
        if tag not in self.meters:
            self.meters[tag] = AverageMeter(tag, fmt=':5.3f')
        self.meters[tag].update(value, n)

    def close(self):
        r"""Closes all the file descriptors."""
        if not self.quiet:
            self.file.close()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def reset(self):
        r"""Resets the ProgressMeter."""
        for m in self.meters.values():
            m.reset()


class AverageMeter(object):
    r"""Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
