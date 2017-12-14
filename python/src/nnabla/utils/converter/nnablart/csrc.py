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

        executor = nnabla.utils.converter.select_executor(nnp)

        # Search network.
        network = nnabla.utils.converter.search_network(
            nnp, executor.network_name)

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

        self._num_of_input_buffers = len(executor.data_variable)
        self._input_buffer_sizes = []
        for n, i in enumerate(executor.data_variable):
            v = variables[i.variable_name]
            self._input_buffer_sizes.append(
                nnabla.utils.converter.calc_shape_size(v.shape, network.batch_size))
            print('Data', n, i.variable_name, i.variable_name in variables)

        self._num_of_output_buffers = len(executor.output_variable)
        self._output_buffer_sizes = []
        for n, o in enumerate(executor.output_variable):
            v = variables[o.variable_name]
            self._output_buffer_sizes.append(
                nnabla.utils.converter.calc_shape_size(v.shape, network.batch_size))
            print('Output', n, o.variable_name, o.variable_name in variables)

        for n, p in enumerate(executor.parameter_variable):
            print('Parameter', n, p.variable_name,
                  p.variable_name in variables, p.variable_name in parameters)

        self._parameters = parameters
        self._network = network
        self._function_info = nnabla.utils.converter.get_function_info()


    def export_csrc_defines(self, dirname, name, prefix):
        input_buffer_size_defines = []
        for n, s in enumerate(self._input_buffer_sizes):
            input_buffer_size_defines.append(
                '#define {}_INPUT{}_SIZE ({})'.format(prefix.upper(), n, s))
        output_buffer_size_defines = []
        for n, s in enumerate(self._output_buffer_sizes):
            output_buffer_size_defines.append(
                '#define {}_OUTPUT{}_SIZE ({})'.format(prefix.upper(), n, s))
        header = csrc_defines.format(name_upper=name.upper(),
                                     prefix=prefix,
                                     prefix_upper=prefix.upper(),
                                     num_of_input_buffers=self._num_of_input_buffers,
                                     input_buffer_size_defines='\n'.join(
                                         input_buffer_size_defines),
                                     num_of_output_buffers=self._num_of_output_buffers,
                                     output_buffer_size_defines='\n'.join(output_buffer_size_defines))
        header_filename = os.path.join(dirname, '{}_inference.h'.format(name))
        with open(header_filename, 'w') as f:
            f.write(header)

    def export_csrc_implements(self, dirname, name, prefix):

        batch_size = self._network.batch_size

        variable_buffers = []
        buffer_index = {}
        for n, v in enumerate(self._network.variable):
            buffer_index[n] = n
            size = nnabla.utils.converter.calc_shape_size(v.shape, batch_size)
            variable_buffers.append(size)
        
        internal_defines = []
        internal_defines.append('typedef struct {')
        internal_defines.append('    float* variable_buffers[{}];'.format(len(variable_buffers)))
        internal_defines.append('')
        internal_defines.append('    // Variables')
        for n, v in enumerate(self._network.variable):
            vsize = nnabla.utils.converter.calc_shape_size(v.shape, batch_size)
            #print(v.name, v.type, v.shape, vsize)
            internal_defines.append('    rt_variable_t v{}; ///< {}'.format(n, v.name))
            internal_defines.append('    int v{}_shape[{}];'.format(n, len(v.shape.dim)))

        internal_defines.append('')
        internal_defines.append('    // Fnctions')
        for n, f in enumerate(self._network.function):
            internal_defines.append('    rt_function_t f{}; ///< {}'.format(n, f.name))
            finfo = self._function_info[f.name]
            internal_defines.append('    rt_variable_t* f{0}_input[{1}];'.format(n, len(finfo['input'])))
            internal_defines.append('    rt_variable_t* f{0}_output[{1}];'.format(n, len(finfo['output'])))
            if 'argument' in finfo:
                internal_defines.append('    {1}_config_t c{0};'.format(n, \
                    finfo['snakecase_name']))

        if len(self._parameters) > 0:
            internal_defines.append('')
            internal_defines.append('    // Parameters')
            internal_defines.append('    void* params[{}];'.format(len(self._parameters)))
            internal_defines.append('    void* param_data;')
            

        internal_defines.append('}} {}_local_context_t;'.format(prefix))


        initialize_context = []
        initialize_context.append('    // Variable buffer')
        for n, size in enumerate(variable_buffers):
            initialize_context.append('    c->variable_buffers[{}] = calloc(sizeof(float), {});'.format(n, size))
            
        initialize_context.append('')
        initialize_context.append('    // Variables')
        for n, v in enumerate(self._network.variable):
            initialize_context.append('    // {}'.format(v.name))
            initialize_context.append('    (c->v{}).type = NN_DATA_TYPE_FLOAT;'.format(n))
            initialize_context.append('    (c->v{}).shape.size = {};'.format(n, len(v.shape.dim)))
            initialize_context.append('    (c->v{0}).shape.data = c->v{0}_shape;'.format(n))
            initialize_context.append('    (c->v{}).data = c->variable_buffers[{}];'.format(n, buffer_index[n]))
            
        initialize_context.append('')
        initialize_context.append('    // Functions')
        for n, f in enumerate(self._network.function):
            finfo = self._function_info[f.name]
            initialize_context.append('    // {}'.format(f.name))
            initialize_context.append('    (c->f{}).num_of_inputs = {};'.format(n, len(finfo['input'])))
            initialize_context.append('    (c->f{0}).inputs = c->f{0}_input;'.format(n))
            initialize_context.append('    (c->f{}).num_of_outputs = {};'.format(n, len(finfo['output'])))
            initialize_context.append('    (c->f{0}).outputs = c->f{0}_output;'.format(n))
            if 'argument' in finfo:
                initialize_context.append('    (c->f{0}).config = &(c->c{0});'.format(n))
                args = []
                for a in finfo['argument']:
                    val = eval('f.{}_param.{}'.format(finfo['snakecase_name'], a))
                    initialize_context.append('    (c->c{}).{} = {};'.format(n, a, val))
                    args.append(str(val))
                initialize_context.append('    init_{}_config(&(c->c{}), {});'.format(finfo['snakecase_name'], n, ', '.join(args)))
                initialize_context.append('    init_{}_local_context(&(c->f{}));'.format(finfo['snakecase_name'], n))

                

        source = csrc_implements.format(name=name,
                                        prefix=prefix,
                                        internal_defines='\n'.join(internal_defines),
                                        initialize_context='\n'.join(initialize_context))

        source_filename = os.path.join(dirname, '{}_inference.c'.format(name))
        with open(source_filename, 'w') as f:
            f.write(source)

    def export_csrc_example(self, dirname, name, prefix):

        prepare_input_file = []
        for n in range(self._num_of_input_buffers):
            prepare_input_file.append(
                '    FILE* input{} = fopen(argv[{}], "rb");'.format(n, n + 1))
            prepare_input_file.append('    assert(input{});'.format(n))
            prepare_input_file.append(
                '    // int input_read_size{2} = fread({0}_output_buffer(context, {2}), sizeof(float), {1}_INPUT{2}_SIZE, input{2});'.format(prefix, prefix.upper(), n))
            prepare_input_file.append(
                '    // assert(input_read_size{1} == {0}_INPUT{1}_SIZE);'.format(prefix.upper(), n))
            prepare_input_file.append('    fclose(input{});'.format(n))
            prepare_input_file.append('')

        prepare_output_file = []
        pos = self._num_of_input_buffers
        for n in range(self._num_of_output_buffers):
            prepare_output_file.append(
                '    char* output_filename{} = malloc(strlen(argv[{}]) + 10);'.format(n, pos + n + 1))
            prepare_output_file.append(
                '    sprintf(output_filename{0}, "%s_{0}.bin", argv[{1}]);'.format(n, pos + n + 1))
            prepare_output_file.append(
                '    FILE* output{0} = fopen(output_filename{0}, "wb");'.format(n))
            prepare_output_file.append('    assert(output{});'.format(n))
            prepare_output_file.append(
                '    // int output_write_size{2} = fwrite({0}_output_buffer(context, {2}), sizeof(float), {1}_OUTPUT{2}_SIZE, output{2});'.format(prefix, prefix.upper(), n))
            prepare_output_file.append(
                '    // assert(output_write_size{1} == {0}_OUTPUT{1}_SIZE);'.format(prefix.upper(), n))
            prepare_output_file.append('    fclose(output{});'.format(n))
            prepare_output_file.append('')

        example = csrc_example.format(name=name,
                                      prefix=prefix,
                                      prefix_upper=prefix.upper(),
                                      num_of_input_buffers=self._num_of_input_buffers,
                                      prepare_input_file='\n'.join(
                                          prepare_input_file),
                                      prepare_output_file='\n'.join(prepare_output_file))

        example_filename = os.path.join(dirname, '{}_example.c'.format(name))
        with open(example_filename, 'w') as f:
            f.write(example)

    def export_csrc_gnumake(self, dirname, name, prefix):
        gnumake = csrc_gnumake.format(name=name)

        gnumake_filename = os.path.join(dirname, 'GNUmakefile'.format(name))
        with open(gnumake_filename, 'w') as f:
            f.write(gnumake)

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
