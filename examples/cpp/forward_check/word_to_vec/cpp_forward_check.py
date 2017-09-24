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
import numpy as np
import subprocess

import nnabla


def check_cpp_forward(save_path, data, x, y, nnp_file, exec_name="Runtime"):

    eval_data_files = []
    for i in range(len(data)):
        eval_data_file = os.path.join(
            save_path, "eval_data_" + str(i) + ".bin")
        eval_data_files.append(eval_data_file)
        data[i].astype(np.float32).tofile(eval_data_file)
    python_forward_file = os.path.join(save_path, "python_forward.txt")
    cpp_forward_file = os.path.join(save_path, "cpp_forward.txt")

    # python forward
    for i in range(len(x)):
        x[i].d = data[i]
    y.forward(clear_buffer=True)
    h = y.d
    if len(h.shape) != 1:
        h = h.reshape(h.shape[0], np.prod(h.shape[1:]))
    np.savetxt(python_forward_file, h, delimiter=',')

    # cpp forward
    command_file = "{} infer".format(
        os.path.join(nnabla.__path__[0], 'bin', 'nbla'))
    command = command_file + " "
    command += nnp_file + " "
    for eval_data_file in eval_data_files:
        command += eval_data_file + " "
    command += " -e {}".format(exec_name)
    command += " > " + cpp_forward_file
    subprocess.call(command, shell=True)
    lines = []
    with open(cpp_forward_file, "r") as f:
        for line in f:
            lines.append(line.split(','))
    z = np.array(list(map(float, lines[-1][:-1]))
                 )
    z = z.reshape(h.shape)

    line = "max_abs_diff_rate_forward=%.10f" % (
        np.max(np.abs(h - z)) / np.mean(np.abs(h)))
    print(line)
    result_file = os.path.join(save_path, line)
    subprocess.call("touch " + result_file, shell=True)
