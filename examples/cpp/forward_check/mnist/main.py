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

import subprocess

commands = [
    'python classification.py -m "tmp.result.lenet.cudnn" -o "tmp.result.lenet.cudnn" -n lenet -c cuda.cudnn -i 10',
    'python classification.py -m "tmp.result.resnet.cudnn" -o "tmp.result.resnet.cudnn" -n resnet -c cuda.cudnn -i 10',
    'python classification_bnn.py -m "tmp.result.bnn.bws_resnet.cudnn" -o "tmp.result.bnn.bws_resnet.cudnn" -n bws-resnet -c cuda.cudnn -i 10',
    'python classification_bnn.py -m "tmp.result.bnn.bincon_resnet.cudnn" -o "tmp.result.bnn.bincon_resnet.cudnn" -n bincon-resnet -c cuda.cudnn -i 10',
    'python classification_bnn.py -m "tmp.result.bnn.binnet_resnet.cudnn" -o "tmp.result.bnn.binnet_resnet.cudnn"  -n binnet-resnet -c cuda.cudnn -i 10',
    'python siamese.py -m "tmp.result.siamese.cudnn" -o "tmp.result.siamese.cudnn" -c cuda.cudnn',
    'python dcgan.py -m "tmp.result.dcgan.cudnn" -o "tmp.result.dcgan.cudnn" -c cuda.cudnn -i 5',
    'python vat.py -o "tmp.result.vat.cudnn" -c cuda.cudnn',
]
for command in commands:
    subprocess.call(command, shell=True)
