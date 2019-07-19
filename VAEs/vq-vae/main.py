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
from data.imagenet_data import data_iterator_imagenet


def make_parser():
    parser = ArgumentParser(description='VQVAE: Dataset Name for training.')
    parser.add_argument('--data', '-d', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'imagenet'])
    return parser


def train(data_iterator, monitor, config, comm):
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

    model = Model(config)
    if config['train']['solver'] == 'adam':
        solver = S.Adam()
    else:
        solver = S.momentum()
    solver.set_learning_rate(config['train']['learning_rate'])

    train_loader = data_iterator(config, comm, train=True)
    val_loader = data_iterator(config, comm, train=False)

    trainer = VQVAEtrainer(model, solver, train_loader, monitor_train_loss, 
    	monitor_train_recon, monitor_val_loss, monitor_val_recon, config, comm)

    if os.path.exists(config['model']['checkpoint']):
        trainer.load_checkpoint(config['model']['checkpoint'])

    for epoch in range(config['train']['num_epochs']):
        iteration = trainer.train(epoch)


if __name__ == '__main__':

	parser = make_parser()
	args = parser.parse_args()
	config = read_yaml(args.data + '.yaml')
	ctx = get_extension_context(config['extension_module'], device_id=config['device_id'])
	nn.set_auto_forward(True)

	if args.data == 'mnist':
		data_iterator = data_iterator_mnist
	elif args.data == 'imagenet':
		data_iterator = data_iterator_imagenet
	elif args.data =='cifar10':
		data_iterator = data_iterator_cifar10
	else:
		print('Dataset not recognized')
		exit(1)

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
