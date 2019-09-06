# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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
import numpy as np

import nnabla as nn
import nnabla.monitor as M
from nnabla.utils.image_utils import imsave

from util.webpage import HtmlCreater


class MonitorWrapper(object):
    def __init__(self, save_path, series_losses, interval=1, save_time=True):
        self.monitor = M.Monitor(save_path)

        self.series_monitors = {}
        for loss_name, loss in series_losses.items():
            self.series_monitors[loss_name] = M.MonitorSeries(
                loss_name, self.monitor, interval=interval)

        if save_time:
            self.monitor_time = M.MonitorTimeElapsed(
                "Epoch time", self.monitor, interval=interval)

    def __call__(self, series_losses, epoch):
        for loss_name, loss in series_losses.items():
            if loss_name not in self.series_monitors:
                raise ValueError("Unexpected loss {} is passed."
                                 " You have to register all losses first in __init__.".format(loss_name))

            if isinstance(loss, nn.Variable):
                v = loss.data.get_data("r")
            elif isinstance(loss, nn.NdArray):
                v = loss.get_data("r")
            else:
                assert isinstance(loss, np.array) or isinstance(loss, float)
                v = loss

            self.series_monitors[loss_name].add(epoch, v)

        if self.monitor_time is not None:
            self.monitor_time.add(epoch)


class Reporter(object):
    def __init__(self, comm, losses, save_path=None, nimage_per_epoch=1, show_interval=20):
        # {"loss_name": loss_Variable, ...}
        assert isinstance(losses, dict)

        self.epoch = 0
        self.batch_cnt = 0
        self.piter = None
        self.comm = comm
        self.losses = tuple(losses.items())  # fix loss order
        self.epoch_losses = {k: 0. for k in losses.keys()}
        self.save_path = save_path
        self.nimage_per_epoch = nimage_per_epoch
        self.buff = {k: nn.NdArray() for k in losses.keys()}
        self.show_interval = show_interval

        is_master = comm.rank == 0
        self.monitor = MonitorWrapper(save_path, self.epoch_losses) if (
            save_path is not None and is_master) else None

        self._reset_buffer()

    def _reset_buffer(self):
        # reset buff
        for loss_name, loss in self.losses:
            self.buff[loss_name] = nn.NdArray()
            self.buff[loss_name].zero()

    def _save_image(self, file_name, image):
        if isinstance(image, nn.Variable):
            img = image.data.get_data("r")
        elif isinstance(image, nn.NdArray):
            img = image.get_data("r")
        else:
            assert isinstance(image, np.ndarray)
            img = image

        dir_path = os.path.join(self.save_path, "html", "imgs")

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        save_path = os.path.join(dir_path, file_name)

        img = (img - img.min()) / (img.max() - img.min())
        imsave(save_path, img)

    def _render_html(self, epoch, image_names):
        # currently dominate only supports to add dom attrs from top to bottom.
        # So we have to re-create html from scratch at each epoch in order to place results in reverse order.
        self.html = HtmlCreater(os.path.join(
            self.save_path, "html"), redirect_interval=60)

        for e in range(epoch, -1, -1):
            self.html.add_text("epoch {}".format(e))
            image_files = []
            for i in range(self.nimage_per_epoch):
                for image_name in image_names:
                    image_files.append(
                        "_".join([image_name, str(e), str(i)]) + ".png")

            self.html.add_images(image_files,
                                 [x.split("_")[0] for x in image_files])

        self.html.save()

    def epoch_start(self, epoch, progress_iter):
        self.epoch = epoch
        self.batch_cnt = 0
        for k, _ in self.losses:
            self.epoch_losses[k] = 0.

        self.piter = progress_iter

    def __call__(self):
        # update state
        self.batch_cnt += 1

        for loss_name, loss in self.losses:
            self.buff[loss_name] += loss.data

        if self.batch_cnt % self.show_interval > 0:
            return

        desc = "[reporter][epoch {}]".format(self.epoch)

        for loss_name, loss in self.losses:
            self.epoch_losses[loss_name] += self.buff[loss_name].get_data("r")
            desc += " {}: {:.4}".format(loss_name,
                                        self.epoch_losses[loss_name] / self.batch_cnt)

        # show current values
        self.piter.set_description(desc)

        self._reset_buffer()

    def epoch_end(self, images, epoch):
        # images = {"image_name": image, ...}
        comm_values = {k: nn.NdArray.from_numpy_array(np.asarray(x / self.batch_cnt, dtype=np.float32))
                       for k, x in self.epoch_losses.items()}

        self.comm.all_reduce(list(comm_values.values()),
                             division=True, inplace=True)

        if self.comm.rank == 0:
            if self.monitor is not None:
                self.monitor(comm_values, epoch)

            # write images to files.
            images_as_tuple = tuple(images.items())

            for image_name, image in images_as_tuple:
                assert len(image) >= self.nimage_per_epoch

                for i in range(self.nimage_per_epoch):
                    file_name = "_".join(
                        [image_name, str(epoch), str(i)]) + ".png"
                    self._save_image(file_name, image[i])

            self._render_html(epoch, tuple(images.keys()))
