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
import re

import nnabla.utils.nnabla_pb2 as nnabla_pb2
import nnabla.utils.converter


class NnbExporter:

    def __init__(self, nnp):

        functions = nnabla.utils.converter.get_function_info()

        message_types = nnabla_pb2.DESCRIPTOR.message_types_by_name
        function_message = message_types['Function']

        import google.protobuf

        TYPE_BOOL = google.protobuf.descriptor.FieldDescriptor.TYPE_BOOL
        TYPE_BYTES = google.protobuf.descriptor.FieldDescriptor.TYPE_BYTES
        TYPE_DOUBLE = google.protobuf.descriptor.FieldDescriptor.TYPE_DOUBLE
        TYPE_ENUM = google.protobuf.descriptor.FieldDescriptor.TYPE_ENUM
        TYPE_FIXED32 = google.protobuf.descriptor.FieldDescriptor.TYPE_FIXED32
        TYPE_FIXED64 = google.protobuf.descriptor.FieldDescriptor.TYPE_FIXED64
        TYPE_FLOAT = google.protobuf.descriptor.FieldDescriptor.TYPE_FLOAT
        TYPE_GROUP = google.protobuf.descriptor.FieldDescriptor.TYPE_GROUP
        TYPE_INT32 = google.protobuf.descriptor.FieldDescriptor.TYPE_INT32
        TYPE_INT64 = google.protobuf.descriptor.FieldDescriptor.TYPE_INT64
        TYPE_MESSAGE = google.protobuf.descriptor.FieldDescriptor.TYPE_MESSAGE
        TYPE_SFIXED32 = google.protobuf.descriptor.FieldDescriptor.TYPE_SFIXED32
        TYPE_SFIXED64 = google.protobuf.descriptor.FieldDescriptor.TYPE_SFIXED64
        TYPE_SINT32 = google.protobuf.descriptor.FieldDescriptor.TYPE_SINT32
        TYPE_SINT64 = google.protobuf.descriptor.FieldDescriptor.TYPE_SINT64
        TYPE_STRING = google.protobuf.descriptor.FieldDescriptor.TYPE_STRING
        TYPE_UINT32 = google.protobuf.descriptor.FieldDescriptor.TYPE_UINT32
        TYPE_UINT64 = google.protobuf.descriptor.FieldDescriptor.TYPE_UINT64

        for param_message in function_message.oneofs[0].fields:
            print(param_message.name, param_message.number)
            message_type = param_message.message_type
            name = re.sub(r'Parameter$', '', message_type.name)
            if not name in functions:
                assert('[{}] not in functions_info'.format(message_type.name))
            for arg in message_type.fields:
                type_name = None
                if arg.type == TYPE_BOOL:
                    type_name = 'BOOL'
                elif arg.type == TYPE_DOUBLE or arg.type == TYPE_FLOAT:
                    type_name = 'FLOAT'
                elif arg.type == TYPE_INT32 or arg.type == TYPE_INT64:
                    type_name = 'INTEGER'
                elif arg.type == TYPE_MESSAGE and arg.message_type.name == 'Shape':
                    type_name = 'SHAPE'
                elif arg.type == TYPE_STRING:
                    type_name = 'STRING'
                else:
                    assert('TYPE[{}] unknown !!!!!!!!!!!!!!'.format(arg.type))
                print('    ', type_name, arg.name, arg.number)

        # TODO: specify executor if number of exec in nnp.
        executor = nnp.protobuf.executor[0]
        print('Using executor [{}].'.format(executor.network_name))

        # Search network.
        network = None
        for n in nnp.protobuf.network:
            if n.name == executor.network_name:
                network = n

        if network is None:
            print('Network for executor [{}] does not found.'.format(
                executor.network_name))
            return

        variables = {}
        for v in network.variable:
            variables[v.name] = v

        for n, i in enumerate(executor.data_variable):
            pass
            #print(n, variables[i.variable_name], i.data_name)

        for n, o in enumerate(executor.output_variable):
            pass
            #print(n, variables[o.variable_name], o.data_name)

        for n, o in enumerate(executor.parameter_variable):
            print(n, o)

        for f in network.function:
            print('Function [{}]'.format(f.name))
            for field in f.ListFields():
                m = re.match('.*_param$', field[0].name)
                if m:
                    print('  Processing [{}].'.format(m.group(0)))
                    params = field[1]
                    for param in params.ListFields():
                        print('    ', param[0].name, type(param[1]))

    def export(self, *args):
        if len(args) == 1:
            print(args[0])
