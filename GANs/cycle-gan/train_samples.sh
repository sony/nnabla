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


# Normal init
python train.py -d 0 -c "cudnn" \
       --monitor-path default.monitor \
       --model-save-path default.monitor
python train.py -d 0 -c "cudnn" \
       --unpool \
       --monitor-path unpool.monitor \
       --model-save-path unpool.monitor
python train.py -d 1 -c "cudnn" \
       --lambda-idt 0.5 \
       --monitor-path identity.monitor \
       --model-save-path identity.monitor
python train.py -d 2 -c "cudnn" \
       --lambda-idt 0.5 \
       --unpool \
       --monitor-path identity.unpool.monitor \
       --model-save-path identity.unpool.monitor

# Paper init
python train.py -d 0 -c "cudnn" \
       --init-method paper \
       --monitor-path paper-init.default.monitor \
       --model-save-path paper-init.default.monitor
python train.py -d 0 -c "cudnn" \
       --unpool \
       --init-method paper \
       --monitor-path paper-init.unpool.monitor \
       --model-save-path paper-init.unpool.monitor
python train.py -d 1 -c "cudnn" \
       --lambda-idt 0.5 \
       --init-method paper \
       --monitor-path paper-init.identity.monitor \
       --model-save-path paper-init.identity.monitor 
python train.py -d 2 -c "cudnn" \
       --lambda-idt 0.5 \
       --unpool \
       --init-method paper \
       --monitor-path paper-init.identity.unpool.monitor \
       --model-save-path paper-init.identity.unpool.monitor


