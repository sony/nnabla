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
import io
import yaml
import pytablewriter
from mako.template import Template

CURRENT_PATH = os.path.dirname(__file__)
IMPORT_SUPPORTED_SYMBOL = b'\xe2\x88\x9a'.decode('utf-8')
IMPORT_DEFINED_SYMBOL = b'\x58'.decode('utf-8')
EXPORT_SUPPORTED_SYMBOL = b'\xe2\x88\x9a'.decode('utf-8')
PART_EXPORT_SUPPORTED_SYMBOL = b'\xe2\x96\xb3'.decode('utf-8')
import_supported_num = 0
import_total_num = 0
export_supported_num = 0
export_total_num = 0


def generate_import_table(opsets):
    global import_supported_num
    global import_total_num
    headers = ['ONNX Operator']
    value_matrix = []
    import_funcs_opset_d = yaml.load(
        open(os.path.join(CURRENT_PATH, 'importer_funcs_opset.yaml'), 'r'))
    for o in opsets:
        headers.append(str(o))
    headers.append('Description')
    for func_name in sorted(import_funcs_opset_d):
        v = [" "+func_name+" "]
        opset_support_list = import_funcs_opset_d[func_name]
        import_total_num += 1
        is_supported = False
        for ops in opsets:
            if str(ops) in opset_support_list:
                if opset_support_list[str(ops)]:
                    v.append(IMPORT_SUPPORTED_SYMBOL)
                    is_supported = True
                else:
                    v.append(IMPORT_DEFINED_SYMBOL)
            else:
                v.append(None)
        if is_supported:
            import_supported_num += 1
        value_matrix.append(v)
    headers = [x.replace(x, " "+x+" ") for x in headers]
    writer = pytablewriter.RstSimpleTableWriter()
    writer.headers = headers
    writer.value_matrix = value_matrix
    return writer.dumps()


def generate_export_table(opsets, export_result):
    global export_supported_num
    global export_total_num
    headers = ['NNabla Functions']
    current_supported = ['6', '9']
    for o in opsets:
        headers.append(str(o))
    headers.append('Description')
    value_matrix = []
    funcs_opset_d = yaml.load(
        open(os.path.join(CURRENT_PATH, 'exporter_funcs_opset.yaml'), 'r'))
    export_refine_d = {}
    for func, impl in funcs_opset_d.items():
        if impl and '@' in impl[0]:
            func_list = [func_decl.split('@')[0] for func_decl in impl]
            export_refine_d[func] = func_list
    for func_name in sorted(funcs_opset_d):
        v = [" "+func_name+" "]
        export_total_num += 1
        test_result = export_result.get(func_name, {}).copy()
        decl = None
        if func_name in export_refine_d:
            decl = 'By ' + ','.join(export_refine_d[func_name])
            for ops in current_supported:
                if ops not in test_result:
                    test_result[str(ops)] = "Not Test"
        for ops in opsets:
            if str(ops) in test_result:
                if test_result[str(ops)] == 'OK':
                    v.append(EXPORT_SUPPORTED_SYMBOL)
                else:
                    v.append(PART_EXPORT_SUPPORTED_SYMBOL)
            else:
                v.append(None)
        v.append(decl)
        if test_result:
            export_supported_num += 1
        value_matrix.append(v)
    headers = [x.replace(x, " "+x+" ") for x in headers]
    writer = pytablewriter.RstSimpleTableWriter()
    writer.headers = headers
    writer.value_matrix = value_matrix
    return writer.dumps()


def gen_report(import_result, export_result):
    OPSET_MAX = 10
    opsets = list(range(1, OPSET_MAX + 1))
    OUTPUT_FILE = os.path.join(
        CURRENT_PATH, '../../../../doc/python/file_format_converter/onnx/operator_coverage.rst')
    import_table = generate_import_table(opsets)
    export_table = generate_export_table(opsets, export_result)
    filename = os.path.join(CURRENT_PATH, 'onnx_test_report.rst.tmpl')
    tmpl = Template(filename=filename)
    output = tmpl.render(IMPORT_SUPPORTED_SYMBOL=IMPORT_SUPPORTED_SYMBOL,
                         IMPORT_DEFINED_SYMBOL=IMPORT_DEFINED_SYMBOL,
                         import_supported_num=import_supported_num,
                         import_total_num=import_total_num,
                         import_table=import_table,
                         EXPORT_SUPPORTED_SYMBOL=EXPORT_SUPPORTED_SYMBOL,
                         PART_EXPORT_SUPPORTED_SYMBOL=PART_EXPORT_SUPPORTED_SYMBOL,
                         export_supported_num=export_supported_num,
                         export_total_num=export_total_num,
                         export_table=export_table)
    with io.open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(output)
