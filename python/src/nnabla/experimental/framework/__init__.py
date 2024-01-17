# Copyright 2023 Sony Group Corporation.
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
The three pillars of machine learning are: data, algorithm, and computing power.
The three classes defined in this file are designed to aim at each of them.
Data is for all works related to dataset preparation, including downloading, sampling, generation, etc.
Worker is for all works related to model designing, including training configuration, solver definitions, loss definitions, etc.
Trainer is a standardized common training process.
If we want to provide computing power service in the future, we can work on this trainer without interrupting customer's data or algorithm.
'''
from .data import Data
from .worker import Worker
from .trainer import Trainer
