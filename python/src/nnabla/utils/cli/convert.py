# Copyright 2018,2019,2020,2021 Sony Corporation.
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


import os

import nnabla.logger as logger
import nnabla.utils.converter


def dump_command(args):
    resolve_file_format(args, args.files)

    if args.import_format not in nnabla.utils.converter.formats.import_name:
        print('Import format ({}) is not supported.'.format(args.import_format))
        return False

    nnabla.utils.converter.dump_files(args, args.files)
    return True


def nnb_template_command(args):
    if len(args.files) >= 2:
        output = args.files.pop()
        resolve_file_format(args, args.files)

        if args.import_format not in nnabla.utils.converter.formats.import_name:
            print('Import format ({}) is not supported.'.format(args.import_format))
            return False

        nnabla.utils.converter.nnb_template(args, args.files, output)
        return True

    print('Input and Output arg is mandatory.')
    return False


def resolve_file_format(args, import_files, export_file=None):
    if len(import_files) == 1:
        input_ext = os.path.splitext(import_files[0])[1]
        if input_ext == '.nnp' or input_ext == '.nntxt':
            args.import_format = 'NNP'
        elif input_ext == '.onnx':
            args.import_format = 'ONNX'
        elif input_ext == '.pb':
            args.import_format = "TF_PB"
        elif input_ext == '.ckpt':
            args.import_format = "TF_CKPT_V1"
        elif input_ext == '.meta':
            args.import_format = "TF_CKPT_V2"
        elif input_ext == '.tflite':
            args.import_format = "TFLITE"
        elif input_ext == '' and os.path.isdir(import_files[0]):
            args.import_format = "SAVED_MODEL"

    if export_file:
        output_ext = os.path.splitext(export_file)[1]
        if output_ext == '.nnp':
            args.export_format = 'NNP'
        elif output_ext == '.nnb':
            args.export_format = 'NNB'
        elif output_ext == '.onnx':
            args.export_format = 'ONNX'
        elif output_ext == '.pb':
            args.export_format = 'TF_PB'
        elif output_ext == '.tflite':
            args.export_format = 'TFLITE'
        elif output_ext == '':
            logger.warning(
                "The export file format is 'CSRC' or 'SAVED_MODEL' that argument '--export-format' will have to be set!!!")
            assert (args.export_format ==
                    'CSRC' or args.export_format == 'SAVED_MODEL')
    else:
        args.export_format = ''

    if args.import_format in ['ONNX', 'TF_CKPT_V1', 'TF_CKPT_V2', 'TF_PB', 'SAVED_MODEL', 'TFLITE'] or \
            args.export_format in ['ONNX', 'TFLITE', 'SAVED_MODEL', 'TF_PB']:
        try:
            import nnabla.utils.converter.onnx
            import nnabla.utils.converter.tensorflow
        except ImportError:
            raise ImportError(
                'nnabla_converter python package is not found, install nnabla_converter package with "pip install nnabla_converter"')


def convert_command(args):
    if len(args.files) >= 2:
        output = args.files.pop()
        resolve_file_format(args, args.files, output)

        if args.import_format not in nnabla.utils.converter.formats.import_name:
            print('Import format ({}) is not supported.'.format(args.import_format))
            return False

        if args.export_format not in nnabla.utils.converter.formats.export_name:
            print('Export format ({}) is not supported.'.format(args.export_format))
            return False

        nnabla.utils.converter.convert_files(args, args.files, output)
        return True

    print('Input and Output arg is mandatory.')
    return False


def add_convert_command(subparsers):

    def add_import_arg(parser):
        parser.add_argument('-I', '--import-format', type=str, default='NNP',
                            help='[import] import format. (one of [{}])'.format(import_formats_string))
        parser.add_argument('-E', '--nnp-import-executor-index', type=int, default=None,
                            help='[import][NNP] import only specified executor.')
        parser.add_argument('--nnp-exclude-preprocess', action='store_true',
                            help='[import][NNP] EXPERIMENTAL exclude preprocess functions when import.')
        parser.add_argument('--nnp-no-expand-network', action='store_true',
                            help='[import][NNP] expand network with repeat or recurrent.')

    import_formats_string = ','.join(
        nnabla.utils.converter.formats.import_name)

    ################################################################################
    # Dump Network.
    subparser = subparsers.add_parser(
        'dump', help='Dump network with supported format.')
    subparser.add_argument('-v', '--dump-verbose', action='store_true',
                           help='[dump] verbose output.')
    subparser.add_argument('-F', '--dump-functions', action='store_true',
                           help='[dump] dump function list.')
    subparser.add_argument('-V', '--dump-variables', action='store_true',
                           help='[dump] dump variable list.')
    subparser.add_argument('--dump-limit', type=int, default=-1,
                           help='[dump] limit num of items.')
    subparser.add_argument('-n', '--dump-variable-name', type=str, default=None,
                           help='[dump] Specific variable name to display.')
    subparser.add_argument('files', metavar='FILE', type=str, nargs='+',
                           help='File or directory name(s) to convert.')
    # import option
    add_import_arg(subparser)
    subparser.set_defaults(func=dump_command)

    ################################################################################
    # Generate NNB template
    subparser = subparsers.add_parser(
        'nnb_template', help='Generate NNB config file template.')
    subparser.add_argument('files', metavar='FILE', type=str, nargs='+',
                           help='File to generate NNB config file template. The last is setting yaml file.')
    # import option
    add_import_arg(subparser)

    subparser.add_argument('-b', '--batch-size', type=int, default=-1,
                           help='[export] overwrite batch size.')
    subparser.add_argument('-T', '--default-variable-type', type=str, nargs=1, default=['FLOAT32'],
                           help='Default type of variable')
    subparser.add_argument('--api', type=int, default=-1,
                           help='Set API Level to convert to, default is highest API Level.')

    subparser.set_defaults(func=nnb_template_command)

    ################################################################################
    # Converter
    subparser = subparsers.add_parser('convert', help='File format converter.')
    subparser.add_argument('files', metavar='FILE', type=str, nargs='+',
                           help='File or directory name(s) to convert. \
                           (When convert ckpt format of the tensorflow model, \
                           If the version of the checkpoint is V1, need to enter the `.ckpt` file, \
                           otherwise need to enter the `.meta` file.)')
    # import option
    subparser.add_argument('--outputs', type=str, default=None,
                           help='[import][tensorflow] The name(s) of the output nodes, comma separated. \
                           Only needed when convert CKPT format.')
    subparser.add_argument('--inputs', type=str, default=None,
                           help='[import][tensorflow] The name(s) of the input nodes, comma separated. \
                           Only needed when convert CKPT format.')
    add_import_arg(subparser)

    # export option
    export_formats_string = ','.join(
        nnabla.utils.converter.formats.export_name)
    subparser.add_argument('-O', '--export-format', type=str, default='NNP',
                           help='[export] export format. (one of [{}]), \
                                the export file format is \'CSRC\' or \'SAVED_MODEL\' that \
                                argument \'--export-format\' will have to be set!!!'.format(export_formats_string))
    subparser.add_argument('-f', '--force', action='store_true',
                           help='[export] overwrite output file.')
    subparser.add_argument('-b', '--batch-size', type=int, default=-1,
                           help='[export] overwrite batch size.')
    subparser.add_argument('-S', '--split', type=str, default=None,
                           help='[export] This option need to set  "-E" option.' +
                           'Split executor with specified index. ' +
                           '(eg. "1-9", "1-2,5-")')
    subparser.add_argument('-d', '--define_version', type=str, default=None,
                           help='[export][ONNX] define onnx opset version. e.g. opset_6' + '\n' +
                           '[export][ONNX] define convert to onnx for SNPE. e.g. opset_snpe' + '\n' +
                           '[export][ONNX] define convert to onnx for TensorRT. e.g. opset_tensorrt' + '\n' +
                           '[export][NNB] define binary format version. e.g. nnb_3')
    subparser.add_argument('--enable-optimize-pb', action='store_true',
                           help='[export][tensorflow] enable optimization when export to pb or tflite.')
    subparser.add_argument('--channel-last', action='store_true',
                           help='[export][TFLite] Specify the data_format of the NNP network,\
                           data_format default is channel_first')
    subparser.add_argument('--quantization', action='store_true',
                           help='[export][TFLite] export to INT8 quantized tflite model.')
    subparser.add_argument('--dataset', type=str, default=None,
                           help='[export][TFLite] Specify the path of represent dataset which will be passed to INT8 quantized tflite converter.')

    # For config function list
    subparser.add_argument('-c', '--config', type=str, default=None,
                           help='[export] config target function list.')

    # NNP
    subparser.add_argument('--nnp-parameter-h5', action='store_true',
                           help='[export][NNP] store parameter with h5 format')
    subparser.add_argument('--nnp-parameter-nntxt', action='store_true',
                           help='[export][NNP] store parameter into nntxt')
    subparser.add_argument('--nnp-exclude-parameter', action='store_true',
                           help='[export][NNP] output without parameter')
    subparser.add_argument('--nnp-version', type=str, default=None,
                           help='[export][NNP] specify the version of nnp (e.g. 1.25.0)')

    # Both NNB and CSRC
    subparser.add_argument('-T', '--default-variable-type', type=str, nargs=1, default=['FLOAT32'],
                           help='Default type of variable')
    subparser.add_argument('-s', '--settings', type=str, nargs=1, default=None,
                           help='Settings in YAML format file.')
    subparser.add_argument('--api', type=int, default=-1,
                           help='Set API Level to convert to, default is highest API Level.')

    subparser.set_defaults(func=convert_command)
