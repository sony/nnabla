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
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("input_utf8")
parser.add_argument("output_utf16")

args = parser.parse_args()


with codecs.open(args.input_utf8, "r", "utf-8") as f_in:
    with codecs.open(args.output_utf16, "w", "utf-16") as f_out:
        for row in f_in:
            f_out.write(row)
