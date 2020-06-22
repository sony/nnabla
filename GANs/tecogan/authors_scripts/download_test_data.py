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
import os


def main():
    # download some test data
    if(not os.path.exists("./LR/")):
        os.mkdir("./LR/")
    cmd2 = "wget https://ge.in.tum.de/download/data/TecoGAN/vid3_LR.zip -O LR/vid3.zip;"
    cmd2 += "unzip LR/vid3.zip -d LR; rm LR/vid3.zip"
    subprocess.call(cmd2, shell=True)

    cmd2 = "wget https://ge.in.tum.de/download/data/TecoGAN/tos_LR.zip -O LR/tos.zip;"
    cmd2 += "unzip LR/tos.zip -d LR; rm LR/tos.zip"
    subprocess.call(cmd2, shell=True)

    # download the ground-truth data
    if(not os.path.exists("./HR/")):
        os.mkdir("./HR/")
    cmd3 = "wget https://ge.in.tum.de/download/data/TecoGAN/vid4_HR.zip -O HR/vid4.zip;"
    cmd3 += "unzip HR/vid4.zip -d HR; rm HR/vid4.zip"
    subprocess.call(cmd3, shell=True)

    cmd3 = "wget https://ge.in.tum.de/download/data/TecoGAN/tos_HR.zip -O HR/tos.zip;"
    cmd3 += "unzip HR/tos.zip -d HR; rm HR/tos.zip"
    subprocess.call(cmd3, shell=True)


if __name__ == '__main__':
    main()
