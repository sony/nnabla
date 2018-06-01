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
python test.py -d 0 -c "cudnn" \
       --monitor-path default.monitor.evaluation \
       --model-load-path default.monitor/params_266799.h5
python test.py -d 0 -c "cudnn" --unpool \
       --monitor-path unpool.monitor.evaluation \
       --model-load-path unpool.monitor/params_266799.h5
python test.py -d 1 -c "cudnn" \
       --monitor-path identity.monitor.evaluation \
       --model-load-path identity.monitor/params_266799.h5
python test.py -d 2 -c "cudnn" --unpool \
       --monitor-path identity.unpool.monitor.evaluation \
       --model-load-path identity.unpool.monitor/params_266799.h5

# Paper init
python test.py -d 0 -c "cudnn" \
       --init-method paper \
       --monitor-path paper-init.default.monitor.evaluation \
       --model-load-path paper-init.default.monitor/params_266799.h5
python test.py -d 0 -c "cudnn" \
       --unpool --init-method paper \
       --monitor-path paper-init.unpool.monitor.evaluation \
       --model-load-path paper-init.unpool.monitor/params_266799.h5
python test.py -d 1 -c "cudnn" \
       --init-method paper \
       --monitor-path paper-init.identity.monitor.evaluation \
       --model-load-path paper-init.identity.monitor/params_266799.h5
python test.py -d 2 -c "cudnn" \
       --unpool --init-method paper \
       --monitor-path paper-init.identity.unpool.monitor.evaluation \
       --model-load-path paper-init.identity.unpool.monitor/params_266799.h5

