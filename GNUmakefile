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

########################################################################################################################
# Suppress most of make message.
.SILENT:

########################################################################################################################
# Default target.

all: bwd-cpplib bwd-wheel

bwd-cpplib: bwd-nnabla-cpplib
bwd-wheel: bwd-nnabla-cpplib bwd-nnabla-wheel

cpplib: nnabla-cpplib
wheel: nnabla-cpplib nnabla-wheel

clean: nnabla-clean
clean-all: nnabla-clean-all


########################################################################################################################
# Settings

include build-tools/make/build.mk

# Some docker settings

include build-tools/make/build-with-docker.mk
