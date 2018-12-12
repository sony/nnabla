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


import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
from nnabla.monitor import Monitor, MonitorSeries
from nnabla.ext_utils import get_extension_context
from nnabla.utils.nnp_graph import NnpLoader

from imagenet_data import data_iterator_imagenet
from models import generator
from helpers import (generate_random_class, generate_one_class, 
                     get_input_and_output, 
                     preprocess)
from args import get_args, save_args

import functools
import cv2
import glob

from scipy.linalg import sqrtm


def compute_inception_score(z, y_fake, x_fake, x, y, args):
    preds = []
    for i in range(args.max_iter):
        logger.info("Compute at {}-th batch".format(i))
        # Generate
        z.d = np.random.randn(args.batch_size, args.latent)
        y_fake.d = generate_random_class(args.n_classes, args.batch_size)
        x_fake.forward(clear_buffer=True)
        # Predict
        x_fake_d = x_fake.d.copy()
        x_fake_d = preprocess(x_fake_d, (args.image_size, args.image_size), args.nnp_preprocess)
        x.d = x_fake_d
        y.forward(clear_buffer=True)
        preds.append(y.d.copy())
    p_yx = np.concatenate(preds)
    # Score
    p_y = np.mean(p_yx, axis=0)
    kld = np.sum(p_yx * (np.log(p_yx) - np.log(p_y)), axis=1)
    score = np.exp(np.mean(kld))
    return score
        
        
def compute_frechet_inception_distance(z, y_fake, x_fake, x, y, args, di=None):
    h_fakes = []
    h_reals = []
    for i in range(args.max_iter):
        logger.info("Compute at {}-th batch".format(i))
        # Generate
        z.d = np.random.randn(args.batch_size, args.latent)
        y_fake.d = generate_random_class(args.n_classes, args.batch_size)
        x_fake.forward(clear_buffer=True)
        # Predict for fake
        x_fake_d = x_fake.d.copy()
        x_fake_d = preprocess(x_fake_d, (args.image_size, args.image_size), args.nnp_preprocess)
        x.d = x_fake_d
        y.forward(clear_buffer=True)
        h_fakes.append(y.d.copy().squeeze())
        # Predict for real
        x_d, _ = di.next()
        x_d = preprocess(x_d, (args.image_size, args.image_size), args.nnp_preprocess)
        x.d = x_d
        y.forward(clear_buffer=True)
        h_reals.append(y.d.copy().squeeze())
    h_fakes = np.concatenate(h_fakes)
    h_reals = np.concatenate(h_reals)

    # FID score
    ave_h_real = np.mean(h_reals, axis=0)    
    ave_h_fake = np.mean(h_fakes, axis=0)
    cov_h_real = np.cov(h_reals, rowvar=False)
    cov_h_fake = np.cov(h_fakes, rowvar=False)
    score = np.sum((ave_h_real - ave_h_fake) ** 2) \
            + np.trace(cov_h_real + cov_h_fake - 2.0 * sqrtm(np.dot(cov_h_real, cov_h_fake)))
    return score


def evaluate(args):
    # Context
    ctx = get_extension_context(args.context, device_id=args.device_id,
                                type_config=args.type_config)
    nn.set_default_context(ctx)

    # Args
    latent = args.latent
    maps = args.maps
    batch_size = args.batch_size
    image_size = args.image_size
    n_classes = args.n_classes
    not_sn = args.not_sn
    
    # Model (Inception model) from nnp file
    nnp = NnpLoader(args.nnp_inception_model_load_path)
    x, y = get_input_and_output(nnp, args.batch_size, name=args.variable_name)

    if args.evaluation_metric == "IS":
        is_model = None
        compute_metric = compute_inception_score
    if args.evaluation_metric == "FID":
        di = data_iterator_imagenet(args.valid_dir, args.dirname_to_label_path, 
                                    batch_size=args.batch_size, 
                                    ih=args.image_size, iw=args.image_size, 
                                    shuffle=True, train=False, noise=False)
        compute_metric = functools.partial(compute_frechet_inception_distance, di=di)

    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_metric = MonitorSeries("{}".format(args.evaluation_metric), monitor, interval=1)

    # Compute the evaluation metric for all models
    cmp_func = lambda path: int(path.split("/")[-1].strip("params_").rstrip(".h5"))
    model_load_path = sorted(glob.glob("{}/*.h5".format(args.model_load_path)), key=cmp_func) \
                      if os.path.isdir(args.model_load_path) else \
                         [args.model_load_path]

    for path in model_load_path:
        # Model (SAGAN)
        nn.load_parameters(path)
        z = nn.Variable([batch_size, latent])
        y_fake = nn.Variable([batch_size])
        x_fake = generator(z, y_fake, maps=maps, n_classes=n_classes, test=True, sn=not_sn)\
                 .apply(persistent=True)
        # Compute the evaluation metric
        score = compute_metric(z, y_fake, x_fake, x, y, args)
        itr = cmp_func(path)
        monitor_metric.add(itr, score)


def main():
    args = get_args()
    save_args(args, "evaluate")

    evaluate(args)
    

if __name__ == '__main__':
    main()

                        
