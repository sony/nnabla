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

import nnabla.utils.converter as cvt


class NnablaHandler:
    def __init__(self, config):
        self._func_set = cvt.func_set_nnabla_support()
        if config:
            self._config_set = cvt.func_set_import_config(config)
            self._func_set &= self._config_set
            self._func_dict = cvt.func_dict_import_config(config)
            self._func_dict = {
                k: v for k, v in self._func_dict.items() if k in self._func_set}
        else:
            self._func_dict = {k: ['ALL'] for k in self._func_set}

    def execute(self, output):
        if output:
            output_ext = os.path.splitext(output)[1].lower()
            if output_ext == '.yaml':
                cvt.func_set_export_yaml(self._func_dict, output)
            else:
                with open(output, 'w') as f:
                    f.write('\n'.join(list(self._func_set)))
        else:
            sys.stdout.write('\n'.join(list(self._func_set)))
            sys.stdout.write('\n')


class NncrHandler:
    def __init__(self, files, config, nnp_no_expand_network):
        self._func_set = cvt.func_set_nncr_support()
        self._nonsupport_set = set()
        if files:
            self._nnp_set = set()
            for f in files:
                nnp = cvt.nnabla.NnpImporter(
                    f, expand_network=not nnp_no_expand_network).execute()
                self._nnp_set |= cvt.func_set_import_nnp(nnp)
            self._func_set &= self._nnp_set
            self._nonsupport_set = self._nnp_set - self._func_set
        if config:
            self._config_set = cvt.func_set_import_config(config)
            self._func_set &= self._config_set
            self._func_dict = cvt.func_dict_import_config(config)
            self._func_dict = {
                k: v for k, v in self._func_dict.items() if k in self._func_set}
            self._nonsupport_set = self._config_set - self._func_set
        else:
            self._func_dict = {k: ['ALL'] for k in self._func_set}

    def execute(self, output):
        if output:
            output_ext = os.path.splitext(output)[1].lower()
            if output_ext == '.yaml':
                cvt.func_set_export_yaml(self._func_dict, output)
            else:
                with open(output, 'w') as f:
                    f.write('\n'.join(list(self._func_set)))
        else:
            sys.stdout.write(
                "nnabla-c-runtime currently support the following functions in model:\n")
            sys.stdout.write('\n'.join(list(self._func_set)))
            sys.stdout.write('\n')
        if len(self._nonsupport_set):
            sys.stderr.write(
                "nnabla-c-runtime currently does not support the following functions in model:\n")
            sys.stderr.write('\n'.join(list(self._nonsupport_set)))
            sys.stderr.write('\n')


class OnnxHandler:
    def __init__(self, files, config, target, nnp_no_expand_network):
        self._func_set = cvt.func_set_onnx_support()
        self._target = target
        self._nonsupport_set = set()
        if files:
            self._nnp_set = set()
            for f in files:
                nnp = cvt.nnabla.NnpImporter(
                    f, expand_network=not nnp_no_expand_network).execute()
                self._nnp_set |= cvt.func_set_import_nnp(nnp)
            self._func_set &= self._nnp_set
            self._nonsupport_set = self._nnp_set - self._func_set
        if config:
            if os.path.exists(config):
                self._config_set = cvt.func_set_import_onnx_config(config)
                self._func_set &= self._config_set
            elif config.startswith('opset_'):
                self._config_set = cvt.func_set_import_onnx_opset(config)
                self._func_set &= self._config_set
            else:
                print("ERROR: config file not found!")
            self._nonsupport_set = self._config_set - self._func_set
        if self._target:
            self._func_set = cvt.func_set_onnx_output_target_list(
                self._func_set)

    def execute(self, output):
        if output:
            output_ext = os.path.splitext(output)[1].lower()
            if output_ext == '.yaml':
                cvt.func_set_exporter_funcs_opset_yaml(self._func_set, output)
            else:
                with open(output, 'w') as f:
                    f.write('\n'.join(list(self._func_set)))
        else:
            sys.stdout.write(
                "nnabla support the following onnx functions in model:\n")
            sys.stdout.write('\n'.join(list(self._func_set)))
            sys.stdout.write('\n')
        if len(self._nonsupport_set):
            sys.stderr.write(
                "nnabla does not support the following onnx functions in model:\n")
            sys.stderr.write('\n'.join(list(self._nonsupport_set)))
            sys.stderr.write('\n')


def function_info_command(args):
    if args.all_support:
        if args.all_support == 'NNB':
            NncrHandler(args.files, args.config,
                        args.nnp_no_expand_network).execute(args.output)
        elif args.all_support == 'ONNX':
            OnnxHandler(args.files, args.config, args.target,
                        args.nnp_no_expand_network).execute(args.output)
    else:
        NnablaHandler(args.config).execute(args.output)
    return True


def add_function_info_command(subparsers):
    ################################################################################
    # Function Info
    subparser = subparsers.add_parser(
        'function_info', help='Output function info.')
    subparser.add_argument('-o', '--output', default=None,
                           help='output filename')
    subparser.add_argument('-f', '--all_support', default=None,
                           help='select function set: NNB, ONNX')
    subparser.add_argument('files', metavar='FILE', type=str, nargs='*',
                           help='*.nnp files that the function sets want to be shown')
    subparser.add_argument('-c', '--config', default=None,
                           help='user config file for target constraint')
    subparser.add_argument('-t', '--target', action='store_true',
                           help='specify function set is output for target format/device')
    subparser.add_argument('--nnp-no-expand-network', action='store_true',
                           help='[import][NNP] expand network with repeat or recurrent.')
    subparser.set_defaults(func=function_info_command)
