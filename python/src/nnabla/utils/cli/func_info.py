# Copyright 2018,2019,2020,2021 Sony Corporation.
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


class FuncInfo:
    params = None


class NnablaHandler:
    def __init__(self):
        self._func_set = cvt.func_set_nnabla_support()
        if FuncInfo.params.files:
            self._nnp_set = set()
            for f in FuncInfo.params.files:
                nnp = cvt.nnabla.NnpImporter(
                    f, expand_network=not FuncInfo.params.nnp_no_expand_network).execute()
                self._nnp_set |= cvt.func_set_import_nnp(nnp)
            self._func_set &= self._nnp_set
        if FuncInfo.params.config:
            self._config_set = cvt.func_set_import_config(
                FuncInfo.params.config)
            self._func_set &= self._config_set
            self._func_dict = cvt.func_dict_import_config(
                FuncInfo.params.config)
            self._func_dict = {
                k: v for k, v in self._func_dict.items() if k in self._func_set}
        else:
            self._func_dict = {k: ['ALL'] for k in self._func_set}

    def execute(self):
        if FuncInfo.params.api != 0:
            api_info = cvt.get_api_level_info()
            api_info.set_api_level(FuncInfo.params.api)
            # query current api level
            print("API_LEVEL: {}".format(api_info.get_current_level()))
            if FuncInfo.params.api == -1:
                return
            for func in api_info.get_function_list():
                print(api_info.get_func_uniq_name(func))
            return
        if FuncInfo.params.output:
            output_ext = os.path.splitext(
                FuncInfo.params.output)[1].lower()
            if output_ext == '.yaml':
                cvt.func_set_export_yaml(
                    self._func_dict, FuncInfo.params.output)
            else:
                with open(FuncInfo.params.output, 'w') as f:
                    f.write('\n'.join(list(self._func_set)))
        else:
            sys.stdout.write('\n'.join(list(self._func_set)))
            sys.stdout.write('\n')


class NncrHandler:
    def __init__(self):
        self._func_set = cvt.func_set_nncr_support()
        self._nonsupport_set = set()
        if FuncInfo.params.files:
            self._nnp_set = set()
            for f in FuncInfo.params.files:
                nnp = cvt.nnabla.NnpImporter(
                    f, expand_network=not FuncInfo.params.nnp_no_expand_network).execute()
                self._nnp_set |= cvt.func_set_import_nnp(nnp)
            self._func_set &= self._nnp_set
            self._nonsupport_set = self._nnp_set - self._func_set
        if FuncInfo.params.config:
            self._config_set = cvt.func_set_import_config(
                FuncInfo.params.config)
            self._func_set &= self._config_set
            self._func_dict = cvt.func_dict_import_config(
                FuncInfo.params.config)
            self._func_dict = {
                k: v for k, v in self._func_dict.items() if k in self._func_set}
            self._nonsupport_set = self._config_set - self._func_set
        else:
            self._func_dict = {k: ['ALL'] for k in self._func_set}

    def execute(self):
        if FuncInfo.params.output:
            output_ext = os.path.splitext(
                FuncInfo.params.output)[1].lower()
            if output_ext == '.yaml':
                cvt.func_set_export_yaml(
                    self._func_dict, FuncInfo.params.output)
            else:
                with open(FuncInfo.params.output, 'w') as f:
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
    def __init__(self):
        self._func_set = cvt.func_set_onnx_support()
        self._nonsupport_set = set()
        if FuncInfo.params.files:
            self._nnp_set = set()
            for f in FuncInfo.params.files:
                nnp = cvt.nnabla.NnpImporter(
                    f, expand_network=not FuncInfo.params.nnp_no_expand_network).execute()
                self._nnp_set |= cvt.func_set_import_nnp(nnp)
            self._func_set &= self._nnp_set
            self._nonsupport_set = self._nnp_set - self._func_set
        if FuncInfo.params.config:
            if os.path.exists(FuncInfo.params.config):
                self._config_set = cvt.func_set_import_onnx_config(
                    FuncInfo.params.config)
                self._func_set &= self._config_set
            elif FuncInfo.params.config.startswith('opset_'):
                self._config_set = cvt.func_set_import_onnx_opset(
                    FuncInfo.params.config)
                self._func_set &= self._config_set
            else:
                print("ERROR: config file not found!")
            self._nonsupport_set = self._config_set - self._func_set
            if FuncInfo.params.files:
                self._nonsupport_set = self._nnp_set - self._func_set
        if FuncInfo.params.target:
            self._func_set = cvt.func_set_onnx_output_target_list(
                self._func_set)

    def execute(self):
        if FuncInfo.params.query and not FuncInfo.params.target:
            yaml_data = cvt.func_set_exporter_funcs_opset_yaml(
                set([FuncInfo.params.query]))
            sys.stdout.write(yaml_data)
            sys.stdout.write('\n')
            return
        if FuncInfo.params.output:
            output_ext = os.path.splitext(
                FuncInfo.params.output)[1].lower()
            if output_ext == '.yaml':
                yaml_data = cvt.func_set_exporter_funcs_opset_yaml(
                    self._func_set)
                with open(FuncInfo.params.output, 'w') as f:
                    f.write(yaml_data)
            else:
                with open(FuncInfo.params.output, 'w') as f:
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
    FuncInfo.params = args
    if args.all_support:
        if args.all_support.upper() == 'NNB':
            NncrHandler().execute()
        elif args.all_support.upper() == 'ONNX':
            OnnxHandler().execute()
        else:
            print("Unsupport args: {}".format(args.all_support))
            return False
    else:
        NnablaHandler().execute()
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
    subparser.add_argument('-q', '--query', default=None,
                           help='query the detail of a function')
    subparser.add_argument('--nnp-no-expand-network', action='store_true',
                           help='[import][NNP] expand network with repeat or recurrent.')
    subparser.add_argument('--api', type=int, default=0,
                           help='List up api levels')
    subparser.set_defaults(func=function_info_command)
