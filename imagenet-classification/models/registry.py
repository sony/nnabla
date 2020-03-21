# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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


__registry = dict()


def query_arch_fn(key):
    if key not in __registry:
        raise ValueError(
            'Key not found: {} not in {}'.format(
                key, list(__registry.keys())))
    return __registry[key]


def get_available_archs():
    return list(__registry.keys())


def register_arch_fn(key, fn):
    global __registry
    if key in __registry:
        raise ValueError('Already exists in registry: {} in {}'.format(
            key, list(__registry.keys())))
    __registry[key] = fn
