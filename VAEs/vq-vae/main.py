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

import os
import sys
from argparse import ArgumentParser
import time

common_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(common_utils_path)
from neu.yaml_wrapper import read_yaml
from neu.comm import CommunicatorWrapper

import nnabla as nn
import nnabla.solvers as S
from nnabla.monitor import Monitor, MonitorSeries, MonitorImageTile
from nnabla.ext_utils import get_extension_context

from models import VQVAE, GatedPixelCNN
from trainers import VQVAEtrainer, TrainerPrior
from data import mnist_iterator, imagenet_iterator, cifar10_iterator


def make_parser():
    parser = ArgumentParser(description='VQVAE Implementation in NNabla')
    parser.add_argument('--data', '-d', type=str, default='cifar10', required=True,
                        choices=['mnist', 'cifar10', 'imagenet'])
    parser.add_argument('--load-checkpoint', action='store_true', default=False,
                        help='Pass this argument to load saved parameters. Path of the saved parameters needs to be defined in config file.')
    parser.add_argument('--pixelcnn-prior', action='store_true', default=False,
                        help='Pass this argument to train a PixelCNN on the trained discretized latent space')
    
    parser.add_argument('--sample-from-pixelcnn', type=int,
                        help='To generate images by randomly sampling using a trained pixelcnn prior. Enter number of images to generate')
    parser.add_argument('--sample-save-path', type=str, default='',
                        help='Path to save samples generated via pixelcnn prior')
    return parser


def train(data_iterator, monitor, config, comm, args):
    monitor_train_loss, monitor_train_recon = None, None
    monitor_val_loss, monitor_val_recon = None, None
    if comm.rank == 0:
        monitor_train_loss = MonitorSeries(
            config['monitor']['train_loss'], monitor, interval=config['train']['logger_step_interval'])
        monitor_train_recon = MonitorImageTile(config['monitor']['train_recon'], monitor, interval=config['train']['logger_step_interval'],
                                               num_images=config['train']['batch_size'])

        monitor_val_loss = MonitorSeries(
            config['monitor']['val_loss'], monitor, interval=config['train']['logger_step_interval'])
        monitor_val_recon = MonitorImageTile(config['monitor']['val_recon'], monitor, interval=config['train']['logger_step_interval'],
                                             num_images=config['train']['batch_size'])

    model = VQVAE(config)
    if config['train']['solver'] == 'adam':
        solver = S.Adam()
    else:
        solver = S.momentum()
    solver.set_learning_rate(config['train']['learning_rate'])

    train_loader = data_iterator(config, comm, train=True)
    if config['dataset']['name'] != 'imagenet':
        val_loader = data_iterator(config, comm, train=False)
    else:
        val_loader = None 

    if not args.pixelcnn_prior:
        trainer = VQVAEtrainer(model, solver, train_loader, val_loader, monitor_train_loss, 
                               monitor_train_recon, monitor_val_loss, monitor_val_recon, config, comm)
        num_epochs = config['train']['num_epochs']
    else:
        pixelcnn_model = GatedPixelCNN(config['prior'])
        trainer = TrainerPrior(model, pixelcnn_model, solver, train_loader, val_loader, monitor_train_loss, 
                               monitor_train_recon, monitor_val_loss, monitor_val_recon, config, comm, eval=args.sample_from_pixelcnn)
        num_epochs = config['prior']['train']['num_epochs']
        

    if os.path.exists(config['model']['checkpoint']) and (args.load_checkpoint or args.sample_from_pixelcnn):
        checkpoint_path = config['model']['checkpoint'] if not args.pixelcnn_prior else config['prior']['checkpoint']
        trainer.load_checkpoint(checkpoint_path, msg='Parameters loaded from {}'.format(config["model"]["checkpoint"]), load_solver=not args.sample_from_pixelcnn)
        
    if args.sample_from_pixelcnn:
        trainer.random_generate(args.sample_from_pixelcnn, args.sample_save_path)
        return

    for epoch in range(num_epochs):
        
        trainer.train(epoch)
        
        if epoch%config['val']['interval'] == 0 and val_loader!=None:
            trainer.validate(epoch)        
            
        if comm.rank == 0:
            if epoch % config['train']['save_param_step_interval'] == 0 or epoch == config['train']['num_epochs']-1:
                trainer.save_checkpoint(
                    config['model']['saved_models_dir'], epoch)


if __name__ == '__main__':

    parser = make_parser()
    args = parser.parse_args()
    config = read_yaml(os.path.join('configs', '{}.yaml'.format(args.data)))
    ctx = get_extension_context(config['extension_module'], device_id=config['device_id'])
    nn.set_auto_forward(True)

    if args.data == 'mnist':
        data_iterator = mnist_iterator
    elif args.data == 'imagenet':
        data_iterator = imagenet_iterator
    elif args.data =='cifar10':
        data_iterator = cifar10_iterator
    else:
        print('Dataset not recognized')
        exit(1)

    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(ctx)

    monitor = None
    if comm.rank == 0:
        monitor = Monitor(config['monitor']['path'])
        start_time = time.time()

    acc = train(data_iterator, monitor, config, comm, args)

    if comm.rank == 0:
        end_time = time.time()
        training_time = (end_time-start_time)/3600

        print('Finished Training!')
        print('Total Training time: {} hours'.format(training_time))
