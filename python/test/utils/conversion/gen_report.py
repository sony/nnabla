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
import re
import fnmatch
import yaml

### Below code is only used to initially create yaml ###

#header_file_path = "/opt/miniconda3/envs/py35/lib/python3.5/site-packages/onnx-1.2.2-py3.5-linux-x86_64.egg/onnx/defs/operator_sets.h"
header_file_path = "../../../../../../onnx/onnx/defs/operator_sets.h"


def obtain_opset_defs(header_file):
    func_opset_dict = {}
    opset_dict = {}
    parser = re.compile(
        r'class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME\(Onnx, (\d*), (.*)\)'
    )
    with open(header_file, "r") as f:
        for line in f.readlines():
            opset_extractor = parser.match(line)
            if opset_extractor:
                opset = opset_extractor.group(1)
                func = opset_extractor.group(2)
                if func in func_opset_dict:
                    func_opset_dict[func].add(opset)
                else:
                    func_opset_dict[func] = set({opset})
                if opset in opset_dict:
                    opset_dict[opset].add(func)
                else:
                    opset_dict[opset] = set({func})
    return func_opset_dict, opset_dict


def initial_write_import_func_opver_yaml():
    import yaml
    import_func_opset_d, opset_d = obtain_opset_defs(header_file_path)
    with open("importer_funcs_opset.yaml", "w") as f:
        refine_d = {}
        for k, v in import_func_opset_d.items():
            d = {}
            for opv in sorted(v):
                d[opv] = True
            refine_d[k] = d
        f.write(yaml.dump(refine_d, default_flow_style=False))


def generate_initial_nnabla_funcs_yaml():
    from collections import OrderedDict
    import yaml

    def type_to_pack_format(typestring):
        fmt = None
        if typestring == 'bool':
            fmt = 'B'
        elif typestring == 'double' or typestring == 'float':
            fmt = 'f'
        elif typestring == 'int64':
            fmt = 'i'
        elif typestring == 'repeated int64' or typestring == 'Shape':
            fmt = 'iI'
        elif typestring == 'string':
            fmt = 'i'
        return fmt

    def represent_odict(dumper, instance):
        return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())

    yaml.add_representer(OrderedDict, represent_odict)

    def load_yaml_ordered(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
        '''
        Load function with keeping the order of dictionaries.
        '''
        class OrderedLoader(Loader):
            pass

        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return object_pairs_hook(loader.construct_pairs(node))
        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping)
        return yaml.load(stream, OrderedLoader)

    def write_initial_nnabla_func_list():
        funcs_yaml = "../../../../build-tools/code_generator/functions.yaml"
        string = open(funcs_yaml, 'r').read()
        info = load_yaml_ordered(string)
        func_info_d = {}
        for cat, cat_info in info.items():
            for func, func_info in cat_info.items():
                func_info_d[func] = ["Not implemented"]
        with open("exporter_funcs_opset.yaml", "w") as f:
            f.write(yaml.dump(func_info_d, default_flow_style=False))
    write_initial_nnabla_func_list()

### Above code is only used to initially create yaml file ###

    def load_yaml_ordered(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
        '''
        Load function with keeping the order of dictionaries.
        '''
        class OrderedLoader(Loader):
            pass

        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return object_pairs_hook(loader.construct_pairs(node))
        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping)
        return yaml.load(stream, OrderedLoader)

    def write_initial_nnabla_func_list():
        funcs_yaml = "../../../../build-tools/code_generator/functions.yaml"
        string = open(funcs_yaml, 'r').read()
        info = load_yaml_ordered(string)
        func_info_d = {}
        for cat, cat_info in info.items():
            for func, func_info in cat_info.items():
                func_info_d[func] = ["Not implemented"]
        with open("exporter_funcs_opset.yaml", "w") as f:
            f.write(yaml.dump(func_info_d, default_flow_style=False))
    write_initial_nnabla_func_list()

### Above code is only used to initially create yaml file ###


class StateMachine():
    def __init__(self, name, handler, **kwargs):
        self.transit_map_ = {}
        self.name_ = name
        self.current_state_ = None
        self.handler_ = handler
        self.events_ = set()
        self.event = None
        self.__dict__.update(kwargs)
        self.next_state = None
        self.default_handle = None
        if 'start' in kwargs:
            self.current_state_ = kwargs['start']
        self.parser_ = re.compile(
            r'^([^ \t\n\r\f\v\->]*)[\s]*[\-]+[>]?[\s]*([^ \t\n\r\f\v\->]*)[\s]*[\-]+>[\s]*([^ \t\n\r\f\v\->]*)$')
        cls = handler.__class__
        for k, v in cls.__dict__.items():
            if hasattr(v, '__call__') and v.__doc__ is not None:
                self._add_transit_by(v, v.__doc__)

    def _event_func(self, *args, **kwargs):
        self.handle_event(self.event, *args, **kwargs)

    def _add_transit_by(self, v, trans):
        for tran in trans.split('\n'):
            tran = tran.strip()
            trans_line = self.parser_.match(tran)
            if trans_line:
                self.add_transit(trans_line.group(1), trans_line.group(2),
                                 trans_line.group(3), v)
                if self.current_state_ is None:
                    self.current_state_ = trans_line.group(1)
                self.events_.add(trans_line.group(2))
            elif tran.strip() == 'default_handle':
                self.default_handle = v

    def __getattr__(self, item):
        for event in self.events_:
            if fnmatch.fnmatch(item, event):
                self.event = item
                return self._event_func

    def add_transit(self, s0, e, s1, func=None):
        if s0 in self.transit_map_:
            handles = self.transit_map_[s0]
            handles[e] = {'func': func, 'state': s1}
        else:
            self.transit_map_[s0] = {e: {'func': func, 'state': s1}}

    def start_state(self, s):
        self.current_state_ = s

    def handle_event(self, e, *args, **kwargs):
        handled = False
        self.handler_.current_event = e
        if self.current_state_ in self.transit_map_:
            handles = self.transit_map_[self.current_state_]
            for k, trans in handles.items():
                if fnmatch.fnmatch(e, k):
                    func = trans['func']
                    self.next_state = handles[k]['state']
                    ret = func(self.handler_, *args, **kwargs)
                    current_state = self.current_state_
                    transit_done = True
                    if ret is None:
                        self.current_state_ = self.next_state
                    elif ret == True:
                        self.current_state_ = self.next_state
                    else:
                        transit_done = False
                    handled = True
                    if self.debug:
                        if transit_done:
                            print("[%s][%s -- %s --> %s]" % (self.name_,
                                                             current_state,
                                                             e,
                                                             self.current_state_))
                        else:
                            print("[%s][%s -- %s --> %s[%s]][Transition is refused]" % (self.name_,
                                                                                        current_state,
                                                                                        e,
                                                                                        self.current_state_,
                                                                                        self.next_state))
                        # for a in args:
                        #     print(a)
                        # for k, v in kwargs.items():
                        #     print('%s=%o' %(k,v))
        if not handled:
            if self.debug:
                print("[%s][%s -- %s <-- %s]" % (self.name_,
                                                 self.current_state_,
                                                 e,
                                                 'not handled'))
            if self.default_handle:
                self.default_handle(self.handler_, *args, **kwargs)

    def get_state(self):
        return self.current_state_

    def set_next_state(self, next_state):
        self.next_state = next_state

    def dump(self):
        for (s, v) in self.transit_map_.items():
            print(s, v)


class StatusHandler:
    def __init__(self, output_buffer, test_result, opset_d):
        self.output_buffer = output_buffer
        self.test_result = test_result
        self.count = 0
        self.ok = 0
        self.count_line = -1
        self.total_line = -1
        self.total_ok = 0
        self.total = 0
        self.opset_d = opset_d

    def parse_upper_line(self, input_line):
        'start -- equal_line --> upper_line_found'
        self.output_buffer += [input_line]

    def parse_operator(self, input_line):
        'upper_line_found -- accept_operator --> operator_found'
        self.output_buffer += [input_line]

    def parse_lower_line(self, input_line):
        'operator_found -- equal_line --> import_table'
        self.field_lens = [len(f) + 1 for f in input_line.split(' ')]
        self.output_buffer += [input_line]

    def handle_import_table(self, input_line):
        'import_table --> process_line --> import_table'
        output_line = '{{:<{}}}{{:<{}}}{{:<{}}}{{:<{}}}'.format(
            *self.field_lens)
        if input_line[0] != ' ':
            fields = filter(lambda x: x != '', input_line.split('  '))
            fields = [f.strip() for f in fields]
            self.count += 1
            opset = self.opset_d.get(fields[0], [])
            if isinstance(opset, list):
                opset_s = ','.join(sorted(opset))
                desc = ' '.join(fields[3:])
            elif isinstance(opset, dict):
                opset_s = ','.join(sorted(opset['version']))
                desc = opset['functions']
            if fields[0] in self.test_result:
                if self.test_result[fields[0]] == 'OK' or self.test_result[fields[0]] == 'Partial OK':
                    self.ok += 1
                line = output_line.format(
                    fields[0], opset_s, self.test_result[fields[0]], desc)
            else:
                if opset:
                    line = output_line.format(
                        fields[0], opset_s, 'Not test', desc)
                else:
                    line = output_line.format(
                        fields[0], opset_s, 'Unimplemented', desc)
            line = line.strip()
            self.output_buffer += [line]
            self.output_buffer += '\n'
        else:
            self.output_buffer += [input_line]

    def default_handle(self, input_line):
        '''
        default_handle
        finish_table-->process_line-->finish_table
        '''
        self.output_buffer += [input_line]

    def finish_import_table(self, input_line):
        'import_table --> equal_line --> finish_table'
        self.output_buffer += [input_line]
        if self.count_line > 0:
            self.output_buffer[self.count_line] = 'Count {}/{}\n'.format(
                self.ok, self.count)
        self.count_line = -1
        self.total += self.count
        self.total_ok += self.ok
        if self.total_line > 0:
            self.output_buffer[self.total_line] = 'Total {}/{}\n'.format(
                self.total_ok, self.total)

    def process_count(self, input_line):
        'start --> accept_count --> start'
        self.count_line = len(self.output_buffer)
        self.output_buffer += ['']
        self.count = 0
        self.ok = 0

    def process_total(self, input_line):
        '''
        start-->accept_total-->start
        upper_line_found-->accept_total-->start
        '''
        self.total_line = len(self.output_buffer)
        self.output_buffer += ['']
        self.total = 0
        self.total_ok = 0


CURRENT_PATH = os.path.dirname(__file__)
TEMPALTE_FILE = os.path.join(CURRENT_PATH, 'onnx_test_report.rst.tmpl')
OUTPUT_FILE = os.path.join(
    CURRENT_PATH, '../../../../doc/python/file_format_converter/onnx/operator_coverage.rst')


def obtain_import_opset_d():
    funcs_opset_d = yaml.load(
        open(os.path.join(CURRENT_PATH, 'importer_funcs_opset.yaml'), 'r'))
    refine_d = {}
    for k, v in funcs_opset_d.items():
        op_ver = []
        for kk, vv in v.items():
            if vv:
                op_ver.append(kk)
        refine_d[k] = op_ver
    return refine_d


def obtain_export_opset_d():
    funcs_opset_d = yaml.load(
        open(os.path.join(CURRENT_PATH, 'exporter_funcs_opset.yaml'), 'r'))
    refine_d = {}
    for func, impl in funcs_opset_d.items():
        if impl and '@' in impl[0]:
            op_ver = {'6', '9'}
            func_list = [func_decl.split('@')[0] for func_decl in impl]
            refine_d[func] = {
                'version': op_ver, 'functions': "Implemented by {}".format(','.join(func_list))}
        else:
            refine_d[func] = {'version': [], 'functions': 'Not implemented'}
    return refine_d


def integration_export_result(export_result, export_opset_d):
    for func, result in export_result.items():
        if len(set(result.values())) == 1:
            if 'NG' in set(result.values()):
                export_result[func] = 'NG'
            elif 'OK' in set(result.values()):
                export_result[func] = 'OK'
        else:
            desc = ""
            for op_ver, test_result in result.items():
                desc += ", opset_{} status is {}".format(op_ver, test_result)
            export_opset_d[func]['functions'] += desc
            export_result[func] = 'Partial OK'


def gen_report(import_result, export_result):
    import_opset_d = obtain_import_opset_d()
    export_opset_d = obtain_export_opset_d()
    integration_export_result(export_result, export_opset_d)
    with open(TEMPALTE_FILE, 'r') as f:
        line_buffer = []
        importer_status_handler = StateMachine('ImporterSM',
                                               StatusHandler(
                                                   line_buffer, import_result, import_opset_d),
                                               start='start', debug=False)
        exporter_status_handler = StateMachine('ExporterSM',
                                               StatusHandler(
                                                   line_buffer, export_result, export_opset_d),
                                               start='start', debug=False)
        for line in f.readlines():
            field = line[:8]
            if exporter_status_handler.get_state() == 'finish_table':
                exporter_status_handler.start_state('start')
            if importer_status_handler.get_state() != 'finish_table':
                if field == '=' * 8:
                    importer_status_handler.equal_line(line)
                elif field[:5] == 'Total':
                    importer_status_handler.accept_total(line)
                elif field == 'Operator':
                    importer_status_handler.accept_operator(line)
                else:
                    importer_status_handler.process_line(line)
            else:
                if field == '=' * 8:
                    exporter_status_handler.equal_line(line)
                elif field == 'Operator':
                    exporter_status_handler.accept_operator(line)
                elif field[:5] == 'Count':
                    exporter_status_handler.accept_count(line)
                elif field[:5] == 'Total':
                    exporter_status_handler.accept_total(line)
                else:
                    exporter_status_handler.process_line(line)
        with open(OUTPUT_FILE, 'w') as of:
            line_buffer = ''.join(line_buffer)
            of.write(line_buffer)
            print('\n{} is updated.'.format(os.path.basename(OUTPUT_FILE)))


if __name__ == '__main__':
    import_d = {}
    export_d = {}
    gen_report(import_d, export_d)
