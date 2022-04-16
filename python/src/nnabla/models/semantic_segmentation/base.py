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
from __future__ import absolute_import

from nnabla.utils.nnp_graph import NnpLoader

from ..utils import *


class SemanticSegmentation(object):

    '''
    Semantic Segmentation pretrained models are inherited from this class
    so that it provides some common interfaces.
    '''

    @property
    def input_shape(self):
        '''
        Should return default image size (channel, height, width) as a tuple.
        '''
        return self._input_shape()

    @property
    def category_names(self):
        '''
        Returns category names of VOC dataset classes.
        '''
        if hasattr(self, '_category_names'):
            return self._category_names
        voc_dir = os.path.join(os.path.dirname(__file__))
        voc_dir = os.path.join(voc_dir.replace('semantic_segmentation', 'object_detection'),
                               'voc.names')
        with open(voc_dir, 'r') as fd:
            self._category_names = fd.read().splitlines()
        return self._category_names

    def _input_shape(self):
        raise NotImplementedError('input size is not implemented')

    def _load_nnp(self, rel_name, rel_url):
        '''
            Args:
                rel_name: relative path to where downloaded nnp is saved.
                rel_url: relative url path to where nnp is downloaded from.

            '''
        from nnabla.utils.download import download
        path_nnp = os.path.join(
                get_model_home(), 'semantic_segmentation/{}'.format(rel_name))
        url = os.path.join(get_model_url_base(),
                           'semantic_segmentation/{}'.format(rel_url))
        logger.info('Downloading {} from {}'.format(rel_name, url))
        dir_nnp = os.path.dirname(path_nnp)
        if not os.path.isdir(dir_nnp):
            os.makedirs(dir_nnp)
        download(url, path_nnp, open_file=False, allow_overwrite=False)
        print('Loading {}.'.format(path_nnp))
        self.nnp = NnpLoader(path_nnp)

    def use_up_to(self, key, callback, **variable_format_dict):
        if key not in self._KEY_VARIABLE:
            raise ValueError('The key "{}" is not present in {}. Available keys are {}.'.format(
                key, self.__class__.__name__, list(self._KEY_VARIABLE.keys())))
        callback.use_up_to(
            self._KEY_VARIABLE[key].format(**variable_format_dict))

    def get_input_var(self, input_var):
        default_shape = (1,) + self.input_shape
        if input_var is None:
            input_var = nn.Variable(default_shape)
        assert input_var.ndim == 4, "input_var must be 4 dimensions. Given {}.".format(
            input_var.ndim)
        assert input_var.shape[1] == 3, "input_var.shape[1] must be 3 (RGB). Given {}.".format(
            input_var.shape[1])
        return input_var

    def __call__(self, input_var=None, use_from=None, use_up_to='segmentation', training=False, returns_net=False, verbose=0):
        '''
        Create a network (computation graph) from a loaded model.

        Args:

            input_var (Variable, optional):
                If given, input variable is replaced with the given variable and a network is constructed on top of the variable. Otherwise, a variable with batch size as 1 and a default shape from ``self.input_shape``.

            use_up_to (str):
                Network is constructed up to a variable specified by a string. A list of string-variable correspondences in a model is described in documentation for each model class.

            training (bool):
                This option enables additional training (fine-tuning, transfer learning etc.) for the constructed network. If True, the ``batch_stat`` option in batch normalization is turned ``True``, and ``need_grad`` attribute in trainable variables (conv weights and gamma and beta of bn etc.) is turned ``True``. The default is ``False``.

            returns_net (bool):
                When ``True``, it returns a :obj:`~nnabla.utils.nnp_graph.NnpNetwork` object. Otherwise, It only returns the last variable of the constructed network. The default is ``False``.
            verbose (bool, or int):
                Verbose level. With ``0``, it says nothing during network construction.
        '''
        raise NotImplementedError()
