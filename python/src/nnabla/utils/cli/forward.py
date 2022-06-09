# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

import csv
import os

import nnabla.utils.callback as callback
import nnabla.utils.load as load
import numpy as np
from nnabla.logger import logger
from nnabla.utils.cli.utility import let_data_to_variable, is_float, compute_full_path
from nnabla.utils.data_iterator import data_iterator_cache
from nnabla.utils.data_iterator import data_iterator_csv_dataset
from nnabla.utils.data_source_loader import FileReader
from nnabla.utils.image_utils import imsave
from nnabla.utils.progress import configure_progress, progress
from six.moves import map


def _set_initial_values(result, type_and_name, d):
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


def _update_result(args, index, result, values, output_index, type_end_names, output_image):
    outputs = []
    for o, type_and_name in zip(values, type_end_names):
        for data_index, d in enumerate(o):
            if len(result.dims) <= output_index:
                result = _set_initial_values(result, type_and_name, d)
            if len(outputs) <= data_index:
                outputs.append([])
            name = result.names[output_index]
            vtype = result.types[output_index]
            dim = result.dims[output_index]

            # Output data
            if vtype == 'col' or not output_image:
                # Vector type output
                if 'numpy.ndarray' != type(d):
                    tmp = np.array([d])
                    outputs[data_index].extend(np.ndarray.flatten(tmp))
                else:
                    outputs[data_index].extend(np.ndarray.flatten(d))
            else:
                for dim_index in range(dim):
                    file_index = index + data_index
                    file_name = '{}_{:04d}'.format(
                        output_index, file_index // 1000) + os.path.sep
                    if dim > 1:
                        file_name += str(dim_index) + '_'
                    file_name += '{}{}'.format(file_index, vtype)
                    full_path = os.path.join(
                        args.outdir, args.result_outdir, file_name)
                    directory = os.path.dirname(full_path)
                    try:
                        os.makedirs(directory)
                    except OSError:
                        pass  # python2 does not support exists_ok arg
                    if vtype in ['.bmp', '.jpeg', '.jpg', '.png', '.gif', '.tif']:
                        x = np.array(d, dtype=np.float32) * 255.
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
                            x = np.array(d)
                            writer.writerows(x)
                    outputs[data_index].append(
                        os.path.join('.', args.result_outdir, file_name))
        output_index += 1

    return result, outputs


def _forward(args, index, config, data, variables, output_image=True):
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
                                     np.reshape(
                                         data[vind], v.variable_instance.d.shape),
                                     data_name=d, variable_name=v.name)
            else:
                let_data_to_variable(v.variable_instance,
                                     data[vind].astype(
                                         v.variable_instance.d.dtype),
                                     data_name=d, variable_name=v.name)

        # Generate data
        for v, generator in e.generator_assign.items():
            v.variable_instance.d = generator(v.variable_instance.d.shape)

        # Forward recursive
        sum = [np.zeros(o.variable_instance.d.shape, dtype=o.variable_instance.d.dtype)
               for o in e.output_assign.keys()]
        sum_mux = [np.zeros(o.variable_instance.d.shape, dtype=o.variable_instance.d.dtype)
                   for o in e.output_assign.keys()]
        for i in range(e.num_evaluations):
            e.forward_target.forward(clear_buffer=True)
            if e.need_back_propagation:
                e.backward_target.backward(clear_buffer=True)

            for o_index, o in enumerate(e.output_assign.keys()):
                if e.repeat_evaluation_type == "last":
                    sum[o_index] = o.variable_instance.d
                else:
                    sum[o_index] += o.variable_instance.d
                    sum_mux[o_index] += (o.variable_instance.d)**2
        if e.repeat_evaluation_type == "last":
            avg = sum
        elif e.repeat_evaluation_type == "std":
            std_result = [np.nan_to_num(np.sqrt(
                x / e.num_evaluations - (y / e.num_evaluations)**2)) for x, y in zip(sum_mux, sum)]
            avg = std_result
        else:
            avg = [s / e.num_evaluations for s in sum]

        result_1, outputs_1 = _update_result(
            args, index, result, avg, output_index, e.output_assign.values(), output_image)
        if 'outputs' in locals():
            outputs = [output + output_1 for output,
                       output_1 in zip(outputs, outputs_1)]
        else:
            outputs = outputs_1
            result = result_1
        output_index += len(avg)

    return result, outputs


def forward_command(args):
    callback.update_status(args)

    configure_progress(os.path.join(args.outdir, 'progress.txt'))
    files = []
    files.append(args.config)
    if args.param:
        files.append(args.param)
    batch_size = args.batch_size
    if batch_size < 1:
        batch_size = None

    class ForwardConfig:
        pass
    config = ForwardConfig
    info = load.load(files, prepare_data_iterator=False, batch_size=batch_size)
    config.global_config = info.global_config

    config.executors = info.executors.values()

    config.networks = []
    for e in config.executors:
        if e.network.name in info.networks.keys():
            config.networks.append(info.networks[e.network.name])
        else:
            logger.critical('Network {} is not found.'.format(
                config.executor.network.name))
            return False

    normalize = True
    for d in info.datasets.values():
        if d.uri == args.dataset or d.cache_dir == args.dataset:
            normalize = d.normalize
    for e in config.executors:
        normalize = normalize and not e.no_image_normalization

    orders = {}
    # With CSV
    if os.path.splitext(args.dataset)[1] == '.csv':
        data_iterator = (lambda: data_iterator_csv_dataset(
            uri=args.dataset,
            batch_size=config.networks[0].batch_size,
            shuffle=False,
            normalize=normalize,
            with_memory_cache=False,
            with_file_cache=False))

        # load dataset as csv
        filereader = FileReader(args.dataset)
        with filereader.open(textmode=True, encoding='utf-8-sig') as f:
            rows = [row for row in csv.reader(f)]
        row0 = rows.pop(0)
        if args.replace_path:
            root_path = os.path.dirname(args.dataset)
            root_path = os.path.abspath(root_path.replace('/|\\', os.path.sep))
        else:
            root_path = '.'
        rows = [row for row in rows if len(row)]
        rows = list(map(lambda row: list(map(lambda i, x: x if row0[i][0] == '#' or is_float(
            x) else compute_full_path(root_path, x), range(len(row)), row)), rows))
        for i in range(len(rows)):
            orders[i] = i
    # With Cache
    elif os.path.splitext(args.dataset)[1] == '.cache':
        data_iterator = (lambda: data_iterator_cache(
            uri=args.dataset,
            batch_size=config.networks[0].batch_size,
            shuffle=False,
            normalize=normalize))

        # Get original CSV
        original_csv = os.path.join(args.dataset, 'original.csv')
        try:
            # load dataset as csv
            filereader = FileReader(original_csv)
            with filereader.open(textmode=True, encoding='utf-8-sig') as f:
                rows = [row for row in csv.reader(f)]
            row0 = rows.pop(0)
            root_path = '.'
            rows = list(map(lambda row: list(map(lambda x: x if is_float(
                x) else compute_full_path(root_path, x), row)), rows))
        except:
            print('Cannot open', original_csv)
            pass

        # Get original Data order.
        order_csv = os.path.join(args.dataset, 'order.csv')
        try:
            filereader = FileReader(order_csv)
            with filereader.open(textmode=True) as f:
                for original, shuffled in [[int(x) for x in row] for row in csv.reader(f)]:
                    orders[original] = shuffled
        except:
            print('Cannot open', order_csv)
            for i in range(len(rows)):
                orders[i] = i
    else:
        print('Unsupported extension "{}" in "{}".'.format(
            os.path.splitext(args.dataset)[1], args.dataset))

    callback.update_status(('data.max', len(rows)))
    callback.update_status(('data.current', 0))
    callback.update_status('processing', True)

    result_csv_filename = os.path.join(args.outdir, args.outfile)
    with open(result_csv_filename, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        with data_iterator() as di:
            index = 0
            while index < di.size:
                data = di.next()
                result, outputs = _forward(
                    args, index, config, data, di.variables)
                if index == 0:
                    for name, dim in zip(result.names, result.dims):
                        if dim == 1:
                            if e.repeat_evaluation_type == "std":
                                name = "Uncertainty(Std)"
                            row0.append(name)
                        else:
                            for d in range(dim):
                                row0.append(name + '__' + str(d))
                    writer.writerow(row0)
                for i, output in enumerate(outputs):
                    if index + i < len(rows):
                        import copy
                        row = copy.deepcopy(rows[orders[index + i]])
                        row.extend(output)
                        writer.writerow(row)
                index += len(outputs)

                callback.update_status(
                    ('data.current', min([index, len(rows)])))
                callback.update_forward_time()
                callback.update_status()

                logger.log(
                    99, 'data {} / {}'.format(min([index, len(rows)]), len(rows)))

    callback.process_evaluation_result(args.outdir, result_csv_filename)

    logger.log(99, 'Forward Completed.')
    progress(None)

    callback.update_status(('output_result.csv_header', ','.join(row0)))
    callback.update_status(('output_result.column_num', len(row0)))
    callback.update_status(('output_result.data_num', len(rows)))
    callback.update_status('finished')

    return True


def infer(info, input_data):
    class tmp:
        pass
    args = tmp
    tmp.outdir = ''
    tmp.result_outdir = ''

    class ForwardConfig:
        pass
    config = ForwardConfig

    config.executors = info.executors.values()
    config.networks = []

    for e in config.executors:
        if e.network.name in info.networks.keys():
            config.networks.append(info.networks[e.network.name])
        else:
            logger.critical('Network {} is not found.'.format(
                config.executor.network.name))
            return False

    normalize = True
    for d in info.datasets.values():
        normalize = d.normalize

    input_file_index = 0
    inputs = []
    for e in config.executors:
        for v, d in e.dataset_assign.items():
            data = input_data[input_file_index].reshape(
                v.variable_instance.d.shape)
            inputs.append((d, data))
            input_file_index += 1
    data = []
    variables = []
    for v, d in inputs:
        variables.append(v)
        data.append(d)

    return _forward(tmp, 0, config, data, variables, False)


def infer_command(args):
    files = []
    files.append(args.config)
    if args.param:
        files.append(args.param)
    batch_size = args.batch_size
    if batch_size < 1:
        batch_size = None

    class ForwardConfig:
        pass
    config = ForwardConfig

    # To improve load performance
    os.environ['NNABLA_CUDNN_ALGORITHM_BY_HEURISTIC'] = '1'
    info = load.load(files, prepare_data_iterator=False, batch_size=batch_size)

    inputs = []
    for input_filename in args.inputs:
        if args.data_type == 'uint8':
            inputs.append(np.fromfile(input_filename, np.uint8))
        elif args.data_type == 'int32':
            inputs.append(np.fromfile(input_filename, np.int32))
        elif args.data_type == 'float32':
            if 'int32' in input_filename:
                inputs.append(np.fromfile(input_filename, np.int32))
            elif 'uint8' in input_filename:
                inputs.append(np.fromfile(input_filename, np.uint8))
            else:
                inputs.append(np.fromfile(input_filename, np.float32))
        else:
            logger.critical('Type is one of ("uint8", "int32" or "float32").')
            return False

    result, outputs = infer(info, inputs)

    for i, o in enumerate(outputs):
        if args.output is not None:
            (np.array(o).astype(np.float32)).tofile(
                "{}_{}.bin".format(args.output, i))
    return True


def add_infer_command(subparsers):
    # Infer
    subparser = subparsers.add_parser(
        'infer', help='Do inference with NNP and binary data file input.')
    subparser.add_argument(
        '-c', '--config', help='path to nntxt', required=True)
    subparser.add_argument(
        '-o', '--output', help='output file prefix', required=False)
    subparser.add_argument(
        '-p', '--param', help='path to parameter file', required=False)
    subparser.add_argument(
        '-t', '--data_type', help='Parameter type (uint8, int32 or float32)', default='float32', required=False)
    subparser.add_argument(
        '-b', '--batch_size',
        help='Batch size to use batch size in nnp file set -1.',
        type=int, default=1)
    subparser.add_argument('inputs', nargs='+')
    subparser.set_defaults(func=infer_command)


def add_forward_command(subparsers):
    # Forward
    subparser = subparsers.add_parser(
        'forward', help='Do evaluation with NNP and test dataset.')
    subparser.add_argument(
        '-c', '--config', help='path to nntxt', required=True)
    subparser.add_argument(
        '-p', '--param', help='path to parameter file', required=False)
    subparser.add_argument(
        '-d', '--dataset', help='path to CSV dataset', required=False)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.add_argument(
        '-f', '--outfile', help='output file name', default='output_result.csv')
    subparser.add_argument(
        '--replace_path', help='replace data path in the dataset with absolute path', action='store_true')
    subparser.add_argument(
        '--result_outdir', help='output result directory', type=str, default='')
    subparser.add_argument(
        '-b', '--batch_size',
        help='Batch size to use batch size in nnp file set -1.',
        type=int, default=-1)
    subparser.set_defaults(func=forward_command)
