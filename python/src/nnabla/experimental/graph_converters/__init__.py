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

from .graph_converter import (GraphConverter, FunctionModifier)
from .batch_normalization_folding import (BatchNormalizationFoldingModifier,
                                          AddBiasModifier,
                                          BatchNormalizationFoldingModifierInner,
                                          BatchNormalizationFoldingOppositeModifierInner)
from .batch_normalization_self_folding import BatchNormalizationSelfFoldingModifier
from .fused_batch_normalization import FusedBatchNormalizationModifier
from .unfused_batch_normalization import UnfusedBatchNormalizationModifier
from .channel_last import ChannelLastModifier
from .channel_first import ChannelFirstModifier
from .remove_function import RemoveFunctionModifier
from .batch_norm_batchstat import BatchNormBatchStatModifier
from .test_mode import TestModeModifier
from .identity import IdentityModifier
