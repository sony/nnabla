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

import collections
import re
import os


def convert_function_name(out):
    if len(out) < 6:
        return out.lower()
    else:
        # Temporary workaround.
        # Function name fields in functions.rst should take a snake case name
        # optionally.
        out = out.replace('ReLU', '_relu')
        return (''.join(["_" + x.lower() if i < len(out) - 1 and x.isupper() and out[i + 1].islower()
                         else x.lower() + "_" if i < len(out) - 1 and x.islower() and out[i + 1].isupper()
                         else x.lower() for i, x in enumerate(list(out))])).lstrip('_').replace('__', '_')


class DefaultRHS:

    def __init__(self, rhs_str):
        self.rhs = rhs_str


class Functions:

    def __init__(self):

        base = os.path.abspath(os.path.dirname(
            os.path.abspath(__file__)) + '/../../..')
        filename = '{}/doc/functions.rst'.format(base)

        error_exists = False
        info = collections.OrderedDict()
        info['Names'] = collections.OrderedDict()
        root = None
        category = None
        function = None
        param_category = None
        l_0 = ''

        info['Functions'] = collections.OrderedDict()

        function_desc_line_num = -1

        with open(filename, 'r') as f:
            for n, l in enumerate(f.readlines()):
                l = l.rstrip()
                m = re.match(r'^[-]+$', l)
                if m:
                    category = l_0
                    assert(category not in info['Functions'])
                    info['Functions'][category] = collections.OrderedDict()

                m = re.match(r'^[\^]+$', l)
                if m:
                    function = l_0
                    info['Names'][l_0] = convert_function_name(l_0)
                    assert(function not in info['Functions'][category])
                    info['Functions'][category][
                        function] = collections.OrderedDict()
                    info['Functions'][category][function]['name'] = function
                    function_desc_line_num = 0

                m = re.match(r'^\*\s+(Input|Argument|Output)\(s\)$', l)
                if m:
                    function_desc_line_num = -1
                    param_type = m.group(1).lower()
                    param_count = -1
                    param_line_count = 0
                    param_name = None
                    info['Functions'][category][function][
                        param_type] = collections.OrderedDict()
                    param_name_list = []
                m = re.match(r'^   ([\s\*]) -\s*(.*)$', l)
                if m:
                    if m.group(1) == '*':
                        param_count += 1
                        param_line_count = 0
                        if param_count == 0:
                            param_name_list.append(m.group(2))
                        else:
                            param_name = m.group(2)
                            info['Functions'][category][function][param_type][
                                param_name] = collections.OrderedDict()
                    else:
                        param_line_count += 1
                        if param_count == 0:
                            param_name_list.append(m.group(2))
                        else:
                            if len(m.group(2).strip()) > 0:
                                info['Functions'][category][function][param_type][
                                    param_name][param_name_list[param_line_count]] = m.group(2).strip()

                if function_desc_line_num >= 0:
                    if function_desc_line_num == 0:
                        info['Functions'][category][
                            function]['description'] = []
                    else:
                        info['Functions'][category][
                            function]['description'].append(l)
                    function_desc_line_num += 1

                l_0 = l  # end of loop

        for cat, cat_info in info['Functions'].items():
            for fun, fun_info in cat_info.items():
                for n in ['input', 'output', 'argument']:
                    if n in fun_info:
                        for i in fun_info[n].values():
                            if 'Description' not in i:
                                i['Description'] = 'No Description'
                if 'argument' in fun_info:
                    for arg, arg_info in fun_info['argument'].items():
                        if 'Default' in arg_info:
                            default_value = arg_info['Default']
                            try:
                                e = eval(default_value)
                                default_value = e
                            except:
                                default_value = DefaultRHS(default_value)
                            arg_info['Default'] = default_value

        if error_exists:
            exit(-1)
        self.info = info


# Main command
# ===============
def main():
    import pprint
    pp = pprint.PrettyPrinter(indent=2)

    f = Functions()
    print('Info Keys = {}'.format(f.info.keys()))
    pp.pprint(f.info['Names'])
    all_functions = f.info['Functions']
    for category, functions in all_functions.items():
        print('Category: {}'.format(category))
        for function, function_info in functions.items():
            print('  Function: {}'.format(function))
            print('    name : {}'.format(function_info['name']))
            for pt in ['input', 'argument', 'output']:
                if pt in function_info:
                    pi = function_info[pt]
                    for n, i in pi.items():
                        print('    {}: {}'.format(pt, n))
                        for k, v in i.items():
                            print('        {}: {}'.format(k, v))


if __name__ == '__main__':
    main()
