# Copyright 2020,2021 Sony Corporation.
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
BINARY_PATH=$(find . -name "test_nbla_utils")
LD_PATH=$(find . -name "libnnabla_util*.*")
LD_PATH+=" "
LD_PATH+=$(find .. -name "libhdf5*.*")

test_nbla_utils=''

for b_p in ${BINARY_PATH}; do
    test_nbla_utils="$(dirname $b_p)/test_nbla_utils"
    export PATH=$(dirname $b_p):$PATH
    break
done

for ld_p in ${LD_PATH}; do
    export LD_LIBRARY_PATH=$(dirname $ld_p):$LD_LIBRARY_PATH
done

echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

chmod u+x $test_nbla_utils
$test_nbla_utils
