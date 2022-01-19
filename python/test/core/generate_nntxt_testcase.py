# Copyright 2021 Sony Corporation.
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
import glob
import argparse
import os

sample_project_paths = [
    "/home/woody/aid/nnabla-sample-data/sample_project/tutorial/*/*/*/*.nntxt"]


def get_nntxt_file():
    for sample_path in sample_project_paths:
        nntxt_files = glob.glob(sample_path)
        for i, fn in enumerate(nntxt_files):
            with open(fn, "r") as f:
                yield "N{:08}".format(i), f, fn


def main(args):
    nntxt_names = []
    with open("nntxt.py", "w") as f:
        for name, nntxt_file, nntxt_filename in get_nntxt_file():
            comment = "# The nntxt is generated from {}. \n".format(
                nntxt_filename)
            f.write(comment)
            code = "{} = r'''{}'''\n".format(name, nntxt_file.read())
            nntxt_names.append(name)
            f.write(code)
        f.write("\n")
        code = "NNTXT_CASES=[{}]".format(",".join(nntxt_names))
        f.write(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
