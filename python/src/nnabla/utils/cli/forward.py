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

from scipy.misc import imsave
import csv
import numpy as np
import os

from nnabla.logger import logger
from nnabla.utils.progress import configure_progress, progress
from nnabla.utils.cli.utility import let_data_to_variable, is_float, compute_full_path
import nnabla.utils.load as load
from nnabla.utils.data_iterator import data_iterator_csv_dataset


def set_initial_values(result, type_and_name, d):
    result.names.append(type_and_name[1])
    vtype = ''
    dim = 0
    if not type_and_name[0]:
        # Infer modal type from tensor shape
        if len(d.shape) == 2:
            # csv for matrix
            vtype = '.csv'
            dim = 1
        elif len(d.shape) == 3:
            # png for 1 map or 3 maps tensor
            vtype = '.png'
            dim = 1 if d.shape[0] == 1 or d.shape[0] == 3 else d.shape[0]
        else:
            # col for others
            vtype = 'col'
            dim = np.prod(d.shape)
    result.types.append(vtype)
    result.dims.append(dim)
    return result


def update_result(args, index, result, values, output_index, type_end_names):
    outputs = []
    for o, type_and_name in zip(values, type_end_names):
        for data_index, d in enumerate(o):
            if len(result.dims) <= output_index:
                result = set_initial_values(result, type_and_name, d)
            if len(outputs) <= data_index:
                outputs.append([])
            name = result.names[output_index]
            vtype = result.types[output_index]
            dim = result.dims[output_index]

            # Output data
            if vtype == 'col':
                # Vector type output
                outputs[data_index].extend(np.ndarray.flatten(d))
            else:
                for dim_index in range(dim):
                    file_index = index + data_index
                    file_name = '{}_{:04d}'.format(
                        output_index, file_index // 1000) + os.path.sep
                    if dim > 1:
                        file_name += str(dim_index) + '_'
                    file_name += '{}{}'.format(file_index, vtype)
                    full_path = os.path.join(args.outdir, file_name)
                    directory = os.path.dirname(full_path)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    if vtype in ['.bmp', '.jpeg', '.jpg', '.png', '.gif', '.tif']:
                        x = np.array(d, dtype=np.float32) * 256.
                        while len(x.shape) == 4:
                            x = x[0]
                        if x.shape[0] > 3 or x.shape[0] == 2:
                            x = x[dim_index]
                        elif x.shape[0] == 3:
                            x = x.transpose(1, 2, 0)
                        else:
                            x = x.reshape(x.shape[1], x.shape[2])
                        x = x.clip(0, 255).astype(np.uint8)
                        imsave(full_path, x)
                    else:
                        # CSV type
                        with open(full_path, 'w') as f:
                            writer = csv.writer(f, lineterminator='\n')
                            x = np.array(d, dtype=np.float32).transpose()
                            writer.writerows(x)
                    outputs[data_index].append(os.path.join('.', file_name))
        output_index += 1

    return result, outputs


def forward(args, index, config, data, variables):
    class ForwardResult:
        pass

    result = ForwardResult()
    result.dims = []
    result.types = []
    result.names = []

    output_index = 0
    for e in config.executors:
        for v, d in e.dataset_assign.items():
            vind = variables.index(d)
            if v.variable_instance.d.shape != data[vind].shape:
                let_data_to_variable(v.variable_instance,
                                     np.reshape(data[vind], v.variable_instance.d.shape))
            else:
                let_data_to_variable(v.variable_instance,
                                     data[vind].astype(v.variable_instance.d.dtype))

        # Generate data
        for v, generator in e.generator_assign.items():
            v.variable_instance.d = generator(v.shape)

        # Forward recursive
        sum = [np.zeros(o.shape) for o in e.output_assign.keys()]
        for i in range(e.num_evaluations):
            e.network.forward(e.forward_sequence)
            if e.need_back_propagation:
                e.network.backward(e.backward_sequence)

            for o_index, o in enumerate(e.output_assign.keys()):
                if e.repeat_evaluation_type == "last":
                    sum[o_index] = o.variable_instance.d
                else:
                    sum[o_index] += o.variable_instance.d
        if e.repeat_evaluation_type == "last":
            avg = sum
        else:
            avg = [s / e.num_evaluations for s in sum]

        result_1, outputs_1 = update_result(
            args, index, result, avg, output_index, e.output_assign.values())
        if 'outputs' in locals():
            outputs = [output + output_1 for output,
                       output_1 in zip(outputs, outputs_1)]
        else:
            outputs = outputs_1
            result = result_1
        output_index += len(outputs_1[0])

    return result, outputs


def forward_command(args):
    configure_progress(os.path.join(args.outdir, 'progress.txt'))
    files = []
    files.append(args.config)
    if args.param:
        files.append(args.param)

    class ForwardConfig:
        pass
    config = ForwardConfig
    info = load.load(files, prepare_data_iterator=False)
    config.global_config = info.global_config

    config.executors = info.executors.values()

    config.networks = []
    for e in config.executors:
        if e.network.name in info.networks.keys():
            config.networks.append(info.networks[e.network.name])
        else:
            logger.critical('Network {} does not found.'.format(
                config.executor.network.name))
            return

    normalize = True
    for d in info.datasets.values():
        if d.uri == args.dataset:
            normalize = d.normalize
    data_iterator = (lambda: data_iterator_csv_dataset(
        args.dataset, config.networks[0].batch_size, False, padding=True, normalize=normalize))

    # load dataset as csv
    with open(args.dataset, 'rt') as f:
        rows = [row for row in csv.reader(f)]
    row0 = rows.pop(0)
    root_path = os.path.dirname(args.dataset)
    root_path = os.path.abspath(root_path.replace('/|\\', os.path.sep))
    rows = map(lambda row: map(lambda x: x if is_float(
        x) else compute_full_path(root_path, x), row), rows)

    with data_iterator() as di:
        index = 0
        while index < di.size:
            data = di.next()
            result, outputs = forward(args, index, config, data, di.variables)
            if index == 0:
                for name, dim in zip(result.names, result.dims):
                    if dim == 1:
                        row0.append(name)
                    else:
                        for d in range(dim):
                            row0.append(name + '__' + str(d))
            for i, output in enumerate(outputs):
                if index + i < len(rows):
                    rows[index + i].extend(output)
            index += len(outputs)
            logger.log(
                99, 'data {} / {}'.format(min([index, len(rows)]), len(rows)))

    with open(os.path.join(args.outdir, 'output_result.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(row0)
        writer.writerows(rows)

    logger.log(99, 'Forward Completed.')
    progress(None)
