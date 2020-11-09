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

import json


class HParams(object):
    r"""Hyperparameter container.

    A `HParams` object holds hyperparameters used to build and train a model,
    such as the learning rate, batch size, etc. The hyperparameter are
    available as attributes of the HParams object as follow:

    ```python
    hp = HParams(learning_rate=0.1, num_hidden_units=100)
    hp.learning_rate    # ==> 0.1
    hp.num_hidden_units # ==> 100
    ```
    """

    def __init__(self, **kargs):
        self.__dict__.update(**kargs)

    def save(self, file_name):
        with open(file_name, 'w') as json_file:
            json.dump(self.__dict__, json_file, ensure_ascii=False, indent=4,
                      default=lambda o: '<not serializable>')

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __repr__(self):
        rep = [k + '=' + str(v) for k, v in self.__dict__.items()]
        return self.__class__.__name__ + '(' + ','.join(rep) + ')'
