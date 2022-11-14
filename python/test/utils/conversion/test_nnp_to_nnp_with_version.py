# Copyright 2022 Sony Group Corporation.
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
import pytest
import nnabla as nn
import nnabla.utils.converter as cvt
from .nntxt import (N0001, N0002, N0003, N0004, N0005)
from .common import generate_case_from_nntxt_str

N_ARRAY = [N0001, N0002, N0003, N0004, N0005]


def nnp_file_name():
    return f"{nn.__version__}.nnp"


def set_default_value(args):
    args.api = -1
    args.batch_size = 1
    args.channel_last = False
    args.config = None
    args.dataset = None
    args.default_variable_type = ['FLOAT32']
    args.define_version = None
    args.enable_optimize_pd = False
    args.export_format = 'NNP'
    args.force = False
    args.import_format = 'NNP'
    args.inputs = None
    args.mpi = False
    args.nnp_exclude_parameter = False
    args.nnp_exclude_preprocess = False
    args.nnp_import_executor_index = None
    args.nnp_no_expand_network = False
    args.nnp_parameter_h5 = False
    args.nnp_parameter_nntxt = False
    args.outputs = None
    args.quantization = False
    args.settings = None
    args.split = None


@pytest.mark.parametrize("nn_version", ["1.12.0"])
@pytest.mark.parametrize("nntxt_str", [N0001, N0003])
def test_nnp_to_nnp_with_version_unsupported(nn_version, nntxt_str):
    class Args:
        pass
    args = Args()
    set_default_value(args)
    with pytest.raises(ValueError) as e:
        with generate_case_from_nntxt_str(nntxt_str, nnp_file_name(), ".h5", 32) as nnp_file:
            dirname = os.path.dirname(nnp_file)
            out_file = os.path.join(dirname, f"{nn_version}.nnp")
            args.files = [nnp_file]
            args.nnp_version = nn_version
            ifiles = [nnp_file]
            output = out_file

            cvt.convert_files(args, ifiles, output)
        print(e)


@pytest.mark.parametrize("nn_version", ["1.12.0"])
@pytest.mark.parametrize("nntxt_idx", [1, 3, 4])
def test_nnp_to_nnp_with_version_supported(nn_version, nntxt_idx):
    class Args:
        pass
    args = Args()
    set_default_value(args)
    nntxt_str = N_ARRAY[nntxt_idx]
    with generate_case_from_nntxt_str(nntxt_str, nnp_file_name(), ".h5", 32) as nnp_file:
        dirname = os.path.dirname(nnp_file)
        out_file = os.path.join(dirname, f"{nn_version}.nnp")
        args.files = [nnp_file]
        args.nnp_version = nn_version
        ifiles = [nnp_file]
        output = out_file

        assert cvt.convert_files(args, ifiles, output), "conversion failed!"
