import os
from argparse import ArgumentParser
import time
import numpy as np
import nnabla as nn
import nnabla.solvers as S
from nnabla.monitor import Monitor, MonitorSeries, MonitorImageTile
from nnabla.ext_utils import get_extension_context

from models.vq_vae import Model
from trainers.vq_vae_train import VQVAEtrainer
from utils.communication_wrapper import CommunicationWrapper
from utils.read_yaml import read_yaml
from data.cifar10_data import data_iterator_cifar10
from data.mnist_data import data_iterator_mnist


def make_parser():
    parser = ArgumentParser(description='VQVAE: Dataset Name for training.')
    parser.add_argument('--data', '-d', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'imagenet'])
    return parser


def train(data_iterator, monitor, config, comm):
    monitor_loss, monitor_acc, = None, None
    if comm.rank == 0:
        monitor_loss = MonitorSeries(
            config['monitor']['train_loss'], monitor, interval=config['train']['logger_step_interval'])
        monitor_recon = MonitorImageTile(config['monitor']['train_recon'], monitor, interval=config['train']['logger_step_interval'],
                                         num_images=config['train']['batch_size'])

    model = Model(config)
    if config['train']['solver'] == 'adam':
        solver = S.Adam()
    else:
        solver = S.momentum()
    solver.set_learning_rate(config['train']['learning_rate'])

    train_loader_ = data_iterator(config['train']['batch_size'], train=True,
                                  shuffle=False, rng=np.random.RandomState(config['model']['rng']))
    val_loader_ = data_iterator(
        config['val']['batch_size'], train=False, rng=np.random.RandomState(config['model']['rng']))
    if comm.n_procs > 1:
        train_loader = train_loader_.slice(rng=None, num_of_slices=comm.n_procs,
                                           slice_pos=comm.rank)
        val_loader = val_loader_.slice(rng=None, num_of_slices=comm.n_procs,
                                       slice_pos=comm.rank)
    else:
        train_loader = train_loader_
        val_loader = val_loader_

    trainer = VQVAEtrainer(model, solver, train_loader,
                           monitor_loss, monitor_recon, config, comm)

    if os.path.exists(config['model']['checkpoint']):
        trainer.load_checkpoint(config['model']['checkpoint'])

    iteration = 0
    for epoch in range(config['train']['num_epochs']):
        iteration = trainer.update(iteration)

        if comm.rank == 0:
            if epoch % config['train']['save_param_step_interval'] == 0 or epoch == config['train']['num_epochs']-1:
                trainer.save_checkpoint(
                    config['model']['saved_models_dir'], epoch)


if __name__ == '__main__':

    parser = make_parser()
    args = parser.parse_args()
    config = read_yaml(args.data + '.yaml')
    ctx = get_extension_context(
        config['extension_module'], device_id=config['device_id'])
    nn.set_auto_forward(True)

    if args.data == 'mnist':
        data_iterator = data_iterator_mnist
    else:
        data_iterator = data_iterator_cifar10

    comm = CommunicationWrapper(ctx)
    nn.set_default_context(ctx)

    monitor = None
    if comm.rank == 0:
        monitor = Monitor(config['monitor']['path'])
        start_time = time.time()

    acc = train(data_iterator, monitor, config, comm)

    if comm.rank == 0:
        end_time = time.time()
        training_time = (end_time-start_time)/3600

        print('Finished Training!')
        print('Total Training time: {} hours'.format(training_time))
