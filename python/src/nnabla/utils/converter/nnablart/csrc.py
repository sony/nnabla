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

import nnabla.utils.converter

from .csrc_templates import \
     csrc_defines, \
     csrc_implements, \
     csrc_example, \
     csrc_gnumake

class CsrcExporter:

    def __init__(self, nnp):
        print('CsrcExporter')
        
        functions = nnabla.utils.converter.get_function_info()

        executor = nnabla.utils.converter.select_executor(nnp)
        nnp.protobuf.executor[0]

        # Search network.
        network = nnabla.utils.converter.search_network(nnp, executor.network_name)

        if network is None:
            print('Network for executor [{}] does not found.'.format(
                executor.network_name))
            return
        print('Using network [{}].'.format(executor.network_name))

        self._network_name = executor.network_name

        parameters = {}
        for p in nnp.protobuf.parameter:
            parameters[p.variable_name] = p

        variables = {}
        for v in network.variable:
            variables[v.name] = v
            
        for n, i in enumerate(executor.data_variable):
            print('Data', n, i.variable_name, i.variable_name in variables)

        for n, o in enumerate(executor.output_variable):
            print('Output', n, o.variable_name, o.variable_name in variables)

        for n, p in enumerate(executor.parameter_variable):
            print('Parameter', n, p.variable_name, p.variable_name in variables, p.variable_name in parameters)


    def export_csrc_defines(self, dirname, name, prefix):
        header = []
        header.append(csrc_defines.format(name_upper=name.upper(), prefix=prefix))
        header_filename = os.path.join(dirname, '{}_inference.h'.format(name))
        with open(header_filename, 'w') as f:
            f.write('\n'.join(header))
        
    def export_csrc_implements(self, dirname, name, prefix):
        source = []
        source.append(csrc_implements.format(name=name, prefix=prefix))
            
        source_filename = os.path.join(dirname, '{}_inference.c'.format(name))
        with open(source_filename, 'w') as f:
            f.write('\n'.join(source))

    def export_csrc_example(self, dirname, name, prefix):
        example = []
        example.append(csrc_example.format(name=name, prefix=prefix))

        example_filename = os.path.join(dirname, '{}_example.c'.format(name))
        with open(example_filename, 'w') as f:
            f.write('\n'.join(example))

    def export_csrc_gnumake(self, dirname, name, prefix):
        gnumake = []
        gnumake.append(csrc_gnumake.format(name=name))

        gnumake_filename = os.path.join(dirname, 'GNUmakefile'.format(name))
        with open(gnumake_filename, 'w') as f:
            f.write('\n'.join(gnumake))
        

    def export_csrc(self, dirname):
        name = self._network_name
        prefix = 'nnablart_{}'.format(name.lower())
        self.export_csrc_defines(dirname, name, prefix)
        self.export_csrc_implements(dirname, name, prefix)
        self.export_csrc_example(dirname, name, prefix)
        self.export_csrc_gnumake(dirname, name, prefix)

            
    def export(self, *args):
        print('CsrcExporter.export')
        if len(args) == 1:
            if os.path.isdir(args[0]):
                self.export_csrc(args[0])
