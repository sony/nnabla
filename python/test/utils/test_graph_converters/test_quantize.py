# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

import pytest
import numpy as np

import nnabla as nn
import nnabla.experimental.graph_converters as GC
import nnabla.functions as F
import nnabla.solvers as S


from nnabla.utils.qnn import QNNConfig, PrecisionMode, FunctionsRankRecorder
from nnabla.ext_utils import get_extension_context
from nbla_test_utils import list_context
from .ref_graphs.resnets import (small_bn_resnet,
                                 small_nonqnn_to_recording_resnet,
                                 small_nonqnn_to_specific_recording_pos_resnet,
                                 small_bn_fcn,
                                 small_nonqnn_to_recording_skip_conv_fcn,
                                 small_bn_multi_fc_resnet,
                                 small_nonqnn_to_recording_skip_affine_resnet,
                                 small_bn_opp_resnet)


ctxs = list_context('Convolution')  # proxy to switch the context
batch_size = 1
record_layers = [
    [],  # If empty add to all layers
    ['Convolution', 'Affine']
]
skip_layers = [
    [],
    ['Convolution'],
    ['Affine']
]


@pytest.mark.parametrize('ctx, func_name', ctxs)
@pytest.mark.parametrize('seed', [520])
@pytest.mark.parametrize('test', [True])
@pytest.mark.parametrize('w_bias', [True])
@pytest.mark.parametrize('channel_last', [True, False])
@pytest.mark.parametrize('graph_ref, graph_act, folding, \
                          self_folding, rec_lays, rec_pos, skip_lays',
                         [(small_nonqnn_to_recording_resnet, small_bn_resnet,
                           True, False, record_layers[0],
                           QNNConfig.RecorderPosition.BEFORE,
                           skip_layers[0]),
                          (small_nonqnn_to_recording_resnet, small_bn_opp_resnet,
                           True, False, record_layers[0],
                           QNNConfig.RecorderPosition.BEFORE,
                           skip_layers[0]),
                          (small_nonqnn_to_recording_resnet, small_bn_resnet,
                           False, True, record_layers[0],
                           QNNConfig.RecorderPosition.BEFORE,
                           skip_layers[0]),
                          (small_nonqnn_to_recording_resnet, small_bn_resnet,
                           False, True, record_layers[1],
                           QNNConfig.RecorderPosition.BEFORE,
                           skip_layers[0]),
                          (small_nonqnn_to_specific_recording_pos_resnet, small_bn_resnet,
                           True, True, record_layers[1],
                           QNNConfig.RecorderPosition.BOTH,
                           skip_layers[0]),
                          (small_nonqnn_to_recording_skip_conv_fcn, small_bn_fcn,
                           True, True, record_layers[0],
                           QNNConfig.RecorderPosition.BEFORE,
                           skip_layers[1]),
                          (small_nonqnn_to_recording_skip_affine_resnet, small_bn_multi_fc_resnet,
                           True, True, record_layers[0],
                           QNNConfig.RecorderPosition.BEFORE,
                           skip_layers[2])
                          ])
def test_nonqnn_to_recording(ctx, func_name, seed, test, w_bias, channel_last,
                             graph_ref, graph_act, folding, self_folding, rec_lays, rec_pos, skip_lays):
    from .graph_converter_test_utils import structure_tester, value_tester

    if channel_last == True and not func_name.endswith('Cudnn'):
        pytest.skip(
            'ChannelLast conversion is only supported in cuDNN context.')

    cfg = QNNConfig()
    cfg.bn_folding = folding
    cfg.bn_self_folding = self_folding
    cfg.channel_last = channel_last
    cfg.record_layers = rec_lays
    cfg.recorder_position = rec_pos
    cfg.skip_inputs_layers = skip_lays
    cfg.skip_outputs_layers = skip_lays

    with nn.context_scope(ctx):
        # Random number
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        # Graph
        x_data = rng.randn(batch_size, 32, 32, 3) if channel_last == True \
            else rng.randn(batch_size, 3, 32, 32)
        x = nn.Variable.from_numpy_array(x_data)

        y_tgt = graph_act(x, test=test, w_bias=w_bias,
                          channel_last=channel_last)

        # BN folding & BN self folding
        modifiers = []
        if cfg.bn_folding:
            modifiers.append(GC.BatchNormalizationFoldingModifier(
                opposite=False, channel_last=cfg.channel_last))
            modifiers.append(GC.BatchNormalizationFoldingModifier(
                opposite=True, channel_last=cfg.channel_last))
        # Go through BN self folding
        if cfg.bn_self_folding:
            modifiers.append(GC.BatchNormalizationSelfFoldingModifier())
        if len(modifiers) > 0:
            y_tgt_without_bn = GC.GraphConverter(modifiers).convert(y_tgt)
            y_tgt.rewire_on(y_tgt_without_bn)

        # FunctionModifier
        funcrankrecorder = FunctionsRankRecorder()
        y_tgt.visit(funcrankrecorder)
        modifiers = []
        modifiers.append(GC.QuantizeNonQNNToRecordingModifier(
            funcrankrecorder.functions_ranks, config=cfg))

        y_act = GC.GraphConverter(modifiers).convert(y_tgt)

        # Ref Graph
        y_ref = graph_ref(x, cfg, test=test, channel_last=channel_last,
                          bn_self_folding=self_folding, record_layers=rec_lays)

        # Test
        structure_tester(y_ref, y_act)
        value_tester(y_tgt, y_act, rtol=6e-02, atol=5e-02)
