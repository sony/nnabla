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
try:
    import nnabla_ext.cuda.cudnn
except:
    pass


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

    # Uploader
    from nnabla.utils.cli.upload import upload_command
    subparser = subparsers.add_parser('upload')
    subparser.add_argument(
        '-d', '--dest', help='destination path', required=True)
    subparser.add_argument(
        '-s', '--size', help='split size in GB', required=False, default=10, type=int)
    subparser.add_argument(
        '-t', '--tmp', help='specify temporary directory')
    subparser.add_argument('source')
    subparser.set_defaults(func=upload_command)

    args = parser.parse_args()

    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
