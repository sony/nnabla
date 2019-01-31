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

import eval
import nnabla.logger as logger
import nnabla.monitor as M
import numpy as np
import nnabla as nn
from args import get_args
import os
import matplotlib.pyplot as plt

def main():
    args = get_args()
    rng = np.random.RandomState(1223)

    # Get context
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)
    iterations = []
    mean_iou = []
    model_dir = args.model_load_path
    for filename in os.listdir(model_dir):
        args.model_load_path=model_dir+filename
        miou = eval.validate(args)
        iterations.append(filename.split('.')[0])
        mean_iou.append(miou)


    for i in range(len(iterations)):
        iterations[i] = iterations[i].replace('param_','')

    itr = list(map(int, iterations))

    #Plot Iterations Vs mIOU
    plt.axes([0, max(itr), 0.0, 1.0])
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy - mIOU')
    plt.scatter(itr,mean_iou)
    plt.show()

    print(iterations)
    print(mean_iou)
    with open('iterations.txt','w') as f:
        for item in iterations:
            f.write('%s\n' %item)
    with open('miou.txt','w') as f2:
        for item in mean_iou:
            f2.write('%s\n' %item)

        
    #plt.plot(iterations, mean_iou)
if __name__ == '__main__':
    main()
