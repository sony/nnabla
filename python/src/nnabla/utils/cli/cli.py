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

import argparse


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Train
    from nnabla.utils.cli.train import train_command
    subparser = subparsers.add_parser('train')
    subparser.add_argument(
        '-r', '--resume', help='resume from last saved parameter.', action='store_true')
    subparser.add_argument(
        '-c', '--config', help='path to nntxt', required=True)
    subparser.add_argument(
        '-s', '--sdcproj', help='path to sdcproj', required=False)
    subparser.add_argument(
        '-j', '--job_url_list', help='path to job url list', required=False)
    subparser.add_argument(
        '-p', '--param', help='path to parameter file', required=False)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.set_defaults(func=train_command)

    # Forward
    from nnabla.utils.cli.forward import forward_command
    subparser = subparsers.add_parser('forward')
    subparser.add_argument(
        '-c', '--config', help='path to nntxt', required=True)
    subparser.add_argument(
        '-p', '--param', help='path to parameter file', required=False)
    subparser.add_argument(
        '-d', '--dataset', help='path to CSV dataset', required=False)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.set_defaults(func=forward_command)

    # Decode param
    from nnabla.utils.cli.encode_decode_param import decode_param_command
    subparser = subparsers.add_parser('decode_param')
    subparser.add_argument(
        '-p', '--param', help='path to parameter file', required=False)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.set_defaults(func=decode_param_command)

    # Encode param
    from nnabla.utils.cli.encode_decode_param import encode_param_command
    subparser = subparsers.add_parser('encode_param')
    subparser.add_argument(
        '-i', '--indir', help='input directory', required=True)
    subparser.add_argument(
        '-p', '--param', help='path to parameter file', required=False)
    subparser.set_defaults(func=encode_param_command)

    # Profile
    from nnabla.utils.cli.profile import profile_command
    subparser = subparsers.add_parser('profile')
    subparser.add_argument(
        '-c', '--config', help='path to nntxt', required=True)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.set_defaults(func=profile_command)

    # Convert dataset
    from nnabla.utils.cli.conv_dataset import conv_dataset_command
    subparser = subparsers.add_parser('conv_dataset')
    subparser.add_argument('-F', '--force', action='store_true',
                           help='force overwrite destination', required=False)
    subparser.add_argument(
        '-S', '--shuffle', action='store_true', help='shuffle data', required=False)
    subparser.add_argument('-N', '--normalize', action='store_true',
                           help='normalize data range', required=False)
    subparser.add_argument('source')
    subparser.add_argument('destination')
    subparser.set_defaults(func=conv_dataset_command)

    # Compare with CPU
    from nnabla.utils.cli.compare_with_cpu import compare_with_cpu_command
    subparser = subparsers.add_parser('compare_with_cpu')
    subparser.add_argument(
        '-c', '--config', help='path to nntxt', required=True)
    subparser.add_argument(
        '-c2', '--config2', help='path to cpu nntxt', required=True)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.set_defaults(func=compare_with_cpu_command)

    from nnabla.utils.cli.create_image_classification_dataset import create_image_classification_dataset_command
    # Create image classification dataset
    subparser = subparsers.add_parser('create_image_classification_dataset')
    subparser.add_argument(
        '-i', '--sourcedir', help='source directory with directories for each class', required=True)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.add_argument(
        '-c', '--channel', help='number of output color channels', required=True)
    subparser.add_argument(
        '-w', '--width', help='width of output image', required=True)
    subparser.add_argument(
        '-g', '--height', help='height of output image', required=True)
    subparser.add_argument(
        '-m', '--mode', help='shaping mode (trimming or padding)', required=True)
    subparser.add_argument(
        '-s', '--shuffle', help='shuffle mode (true or false)', required=True)
    subparser.add_argument(
        '-f1', '--file1', help='output file name 1', required=True)
    subparser.add_argument(
        '-r1', '--ratio1', help='output file ratio(%) 1')
    subparser.add_argument(
        '-f2', '--file2', help='output file name 2')
    subparser.add_argument(
        '-r2', '--ratio2', help='output file ratio(%) 2')
    subparser.set_defaults(func=create_image_classification_dataset_command)

    # Uploader
    from nnabla.utils.cli.uploader import upload_command, Uploader
    subparser = subparsers.add_parser('upload')
    subparser.add_argument('-e', '--env', help='select an environment to upload from [{}].'.format(
        ', '.join(sorted(Uploader.environments.keys())), type=str))
    subparser.add_argument('token', help='token for upload')
    subparser.add_argument('filename', help='filename to upload')
    subparser.set_defaults(func=upload_command)

    # Create TAR for uploader
    from nnabla.utils.cli.uploader import create_tar_command
    subparser = subparsers.add_parser('create_tar')
    subparser.add_argument('source', help='CSV dataset')
    subparser.add_argument('destination', help='TAR filename')
    subparser.set_defaults(func=create_tar_command)

    # Extract nnp file
    from nnabla.utils.cli.extract import extract_command
    subparser = subparsers.add_parser('extract')
    subparser.add_argument(
        '-l', '--list', help='list contents.', action='store_true')
    subparser.add_argument(
        '-x', '--extract', help='extract contents to current dir.', action='store_true')
    subparser.add_argument('nnp', help='nnp filename')
    subparser.set_defaults(func=extract_command)

    from nnabla.utils.cli.create_image_classification_dataset import create_image_classification_dataset_command
    # Create image classification dataset
    subparser = subparsers.add_parser(
        'create_image_classification_dataset')
    subparser.add_argument(
        '-i', '--sourcedir', help='source directory with directories for each class', required=True)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.add_argument(
        '-c', '--channel', help='number of output color channels', required=True)
    subparser.add_argument(
        '-w', '--width', help='width of output image', required=True)
    subparser.add_argument(
        '-g', '--height', help='height of output image', required=True)
    subparser.add_argument(
        '-m', '--mode', help='shaping mode (trimming or padding)', required=True)
    subparser.add_argument(
        '-s', '--shuffle', help='shuffle mode (true or false)', required=True)
    subparser.add_argument(
        '-f1', '--file1', help='output file name 1', required=True)
    subparser.add_argument(
        '-r1', '--ratio1', help='output file ratio(%) 1')
    subparser.add_argument(
        '-f2', '--file2', help='output file name 2')
    subparser.add_argument(
        '-r2', '--ratio2', help='output file ratio(%) 2')
    subparser.set_defaults(
        func=create_image_classification_dataset_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    import six.moves._thread as thread
    import threading
    thread.stack_size(128 * 1024 * 1024)
    sys.setrecursionlimit(0x3fffffff)
    main_thread = threading.Thread(target=main)
    main_thread.start()
    main_thread.join()
