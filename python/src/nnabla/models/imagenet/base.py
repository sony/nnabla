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
from __future__ import absolute_import
import os


class ImageNetBase(object):

    @property
    def category_names(self):
        '''
        Returns category names of 1000 ImageNet classes.
        '''
        if hasattr(self, '_category_names'):
            return self._category_names
        with open(os.path.join(os.path.dirname(__file__), 'category_names.txt'), 'r') as fd:
            self._category_names = fd.read().splitlines()
        return self._category_names

    @property
    def input_shape(self):
        '''
        Should returns default image size (channel, height, width) as a tuple.
        '''
        return self._input_shape()

    def _input_shape(self):
        raise NotImplementedError('input size is not implemented')
