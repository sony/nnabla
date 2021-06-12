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
'''
Data exchange format for "Neural Network Library".

Current version of .nnp file is just a ZIP format archive file but filename extension is '.nnp'.

'.nnp' file will contain following files. If '.nnp' file contain other file, nnabla will just ignore that files.

* 'nnp_version.txt'

  * Specify version of nnp file. Version string in this file is got from nnp_version().
* '*.nntxt' (or '*.prototxt')

  * Network structure in Protocol buffer text format.
* '*.protobuf'

  * Trained parameter in Protocol buffer binary format.
* '*.h5'

  * Trained parameter in HDF5 format.(Will be obsolete soon.)



'''


def nnp_version():
    '''nnp_version

    Current version is "0.1"

    * Version history

      * Version 0.1

        * First version.

    '''
    return '0.1'
