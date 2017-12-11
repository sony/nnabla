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

from nnabla.utils.converter.nnabla import NnpReader
from nnabla.utils.converter.nnabla import NnpExporter
from nnabla.utils.converter.nnablart import NnbExporter


def convert_files(args, ifiles, output):
    # Currently input file seems to be NNP input.
    # Input file that has unsuported extension store into output nnp archive or directory.
    reader = NnpReader(*ifiles, expand_network=args.nnp_expand_network)
    nnp = reader.read()

    if nnp is not None:
        output_ext = os.path.splitext(output)[1].lower()
        if os.path.isdir(output) or output_ext == '.nnp':
            parameter_type = 'protobuf'
            if args.nnp_parameter_nntxt:
                parameter_type = 'included'
            elif args.nnp_parameter_h5:
                parameter_type = 'h5'
            if args.nnp_exclude_parameter:
                parameter_type = 'none'

            exporter = NnpExporter(nnp, parameter_type)
            exporter.export(output)
        elif output_ext == '.nnb':
            exporter = NnbExporter(nnp)
            exporter.export(output)
        else:
            print('Output file extension ({}) is not supported.'.format(oext))
            return False
    else:
        print('Read failed.')
    return False
