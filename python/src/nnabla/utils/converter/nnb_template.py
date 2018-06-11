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

import os
import sys

from .nnabla import NnpReader, NnpExporter
from .nnablart import NnbExporter, CsrcExporter
from .onnx import OnnxReader, OnnxExporter


def generate_nnb_template(args, nnp, output):
    NnbExporter(nnp, args.batch_size).export(
        None, output, None, args.default_variable_type)
    return True


def nnb_template(args, ifiles, output):
    nnp = None
    if args.read_format == 'NNP':
        # Input file that has unsuported extension store into output nnp archive or directory.
        nnp = NnpReader(*ifiles, expand_network=args.nnp_expand_network).read()
    elif args.read_format == 'ONNX':
        nnp = OnnxReader(*ifiles).read()
    if nnp is not None:
        return generate_nnb_template(args, nnp, output)
    else:
        print('Read from [{}] failed.'.format(ifiles))
        return False
