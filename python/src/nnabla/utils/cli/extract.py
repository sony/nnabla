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
import zipfile


def extract_command(args):
    if os.path.isfile(args.nnp):
        if args.list:
            print('Files in  {}'.format(args.nnp))
            with zipfile.ZipFile(args.nnp, 'r') as nnp:
                for filename in nnp.namelist():
                    print('    {}'.format(filename))
        elif args.extract:
            with zipfile.ZipFile(args.nnp, 'r') as nnp:
                for filename in nnp.namelist():
                    with nnp.open(filename, 'r') as zf:
                        with open(filename, 'wb') as f:
                            f.write(zf.read())
                            print('    Extracting {}'.format(filename))
        else:
            print('Please specify -l or -x')
    else:
        print('[{}] not found.'.format(args.nnp))
