# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from nnabla.logger import logger


def _nnabla_version():
    import nnabla
    import nnabla.utils.callback as callback
    version_string = 'Version:{}, Build:{}'.format(nnabla.__version__,
                                                   nnabla.__build_number__)
    callback_version_string = callback.get_callback_version()
    if callback_version_string is not None:
        version_string += ', Callback:{}'.format(callback_version_string)
    return version_string


def version_command(args):
    print(_nnabla_version())


return_value = None


def main():

    parser = argparse.ArgumentParser(description='Command line interface ' +
                                     'for NNabla({})'.format(_nnabla_version()))
    parser.add_argument(
        '-m', '--mpi', help='exec with mpi.', action='store_true')

    subparsers = parser.add_subparsers()

    from nnabla.utils.cli.train import add_train_command
    add_train_command(subparsers)

    from nnabla.utils.cli.forward import add_infer_command, add_forward_command
    add_infer_command(subparsers)
    add_forward_command(subparsers)

    from nnabla.utils.cli.encode_decode_param import add_decode_param_command, add_encode_param_command
    add_encode_param_command(subparsers)
    add_decode_param_command(subparsers)

    from nnabla.utils.cli.profile import add_profile_command
    add_profile_command(subparsers)

    from nnabla.utils.cli.conv_dataset import add_conv_dataset_command
    add_conv_dataset_command(subparsers)

    from nnabla.utils.cli.compare_with_cpu import add_compare_with_cpu_command
    add_compare_with_cpu_command(subparsers)

    from nnabla.utils.cli.create_image_classification_dataset import add_create_image_classification_dataset_command
    add_create_image_classification_dataset_command(subparsers)

    from nnabla.utils.cli.create_object_detection_dataset import add_create_object_detection_dataset_command
    add_create_object_detection_dataset_command(subparsers)

    from nnabla.utils.cli.uploader import add_upload_command
    add_upload_command(subparsers)

    from nnabla.utils.cli.uploader import add_create_tar_command
    add_create_tar_command(subparsers)

    from nnabla.utils.cli.convert import add_convert_command
    add_convert_command(subparsers)

    from nnabla.utils.cli.func_info import add_function_info_command
    add_function_info_command(subparsers)

    from nnabla.utils.cli.optimize_model import add_optimize_command
    add_optimize_command(subparsers)

    from nnabla.utils.cli.plot import (
        add_plot_series_command, add_plot_timer_command)
    add_plot_series_command(subparsers)
    add_plot_timer_command(subparsers)

    from nnabla.utils.cli.draw_graph import add_draw_graph_command
    add_draw_graph_command(subparsers)

    # Version
    subparser = subparsers.add_parser(
        'version', help='Print version and build number.')
    subparser.set_defaults(func=version_command)

    print('NNabla command line interface ({})'.format(_nnabla_version()))

    args = parser.parse_args()

    import nnabla.utils.callback as callback
    r = callback.alternative_cli(args)
    if r is not None:
        return r

    global return_value
    import six.moves._thread as thread
    import threading
    thread.stack_size(8 * 1024 * 1024)
    sys.setrecursionlimit(1024 * 1024)
    main_thread = threading.Thread(target=cli_main, args=(parser, args))
    main_thread.start()
    main_thread.join()
    if not return_value:
        sys.exit(-1)


def cli_main(parser, args):
    global return_value
    return_value = False

    if 'func' not in args:
        parser.print_help(sys.stderr)
        sys.exit(-1)

    if args.mpi:
        from nnabla.utils.communicator_util import create_communicator
        comm = create_communicator()
        try:
            return_value = args.func(args)
        except:
            import traceback
            print(traceback.format_exc())

            logger.log(99, "ABORTED")
            os.kill(os.getpid(), 9)
            # comm.abort()
    else:
        try:
            return_value = args.func(args)
        except:
            import traceback
            print(traceback.format_exc())
            return_value = False
            sys.exit(-1)


if __name__ == '__main__':
    main()
