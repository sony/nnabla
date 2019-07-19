import nnabla as nn
import nnabla.functions as F
import numpy as np
import os
from tqdm import trange
import matplotlib.pyplot as plt 


class VQVAEtrainer(object):

	def __init__(self, model, solver, data_loader, val_data_loader, monitor_train_loss, monitor_train_recon,
		monitor_val_loss, monitor_val_recon, config, comm):
		self.model = model
		self.solver = solver
		self.data_loader = data_loader
		self.val_data_loader = val_data_loader
		self.monitor_train_loss = monitor_train_loss
		self.monitor_train_recon = monitor_train_recon
		self.monitor_val_loss = monitor_val_loss 
		self.monitor_val_recon = monitor_val_recon
		self.comm = comm

		self.iterations_per_epoch = int(np.ceil(self.data_loader.size/self.data_loader.batch_size/comm.n_procs))
		self.weight_decay = config['train']['weight_decay']
		self.learning_rate_decay_epochs = config['train']['learning_rate_decay_epochs']
		self.learning_rate_decay_factor = config['train']['learning_rate_decay_factor']

		self.train_recon_path = os.path.join(config['monitor']['path'], config['monitor']['train_recon'])
		self.val_recon_path = os.path.join(config['monitor']['path'], config['monitor']['val_recon'])
		os.makedirs(self.train_recon_path, exist_ok=True)
		os.makedirs(self.val_recon_path, exist_ok=True)

		self.dataset_name = config['dataset']['name']
		if self.dataset_name != 'imagenet':
			self.val_iterations_per_epoch = int(np.ceil(self.val_data_loader.size/self.val_data_loader.batch_size/comm.n_procs))
			self.data_variance = np.var(self.data_loader._data_source._images/255.0)
			
		else:
			self.data_variance = 1

	def save_checkpoint(self, path, epoch):
		file_name = os.path.join(path, 'epoch_'+str(epoch))
		os.makedirs(file_name, exist_ok=True)
		nn.save_parameters(os.path.join(file_name, 'params.h5'))
		self.solver.save_states(os.path.join(file_name, 'solver.h5'))

	def load_checkpoint(self,path):
		nn.load_parameters(os.path.join(path, 'params.h5'))
		self.solver.load_states(os.path.join(path, 'solver.h5'))

	def convert_to_var(self, img):
		if not np.all(img < 1):
			img = img/255.0
		img = (img - 0.5)/1
		return nn.Variable.from_numpy_array(img)

	def scale_back_var(self, img_var):
		img = img_var.d
		img = np.transpose(img, (0,2,3,1))
		img = (img - np.min(img))/np.ptp(img)
		return img

	def compute_loss(self, img_var):
		vq_loss, img_recon, perplexity = self.model(img_var)
		recon_loss = F.mean(F.squared_error(img_recon,img_var))/self.data_variance
		loss = recon_loss + vq_loss
		return loss, recon_loss, perplexity, img_recon

	def save_image(self, img, path):
		# img = (img - np.min(img))/np.ptp(img)	
		img = img*0.5 + 0.5
		plt.imshow(nn.monitor.tile_images(img))
		plt.savefig(path)
		plt.close()
		print('Saving reconstrutions in', path)

	def log_loss(self, epoch, loss, train=True):
		if train:
			self.monitor_train_loss.add(epoch, loss)
		else:
			self.monitor_val_loss.add(epoch, loss)

	def log_image(self, epoch, img_recon, train=True):
		if self.dataset_name == 'imagenet' or True:
			if train:
				self.save_image(img_recon.d, os.path.join(self.train_recon_path, 'epoch_{}.png'.format(epoch)))
			else:
				self.save_image(img_recon.d, os.path.join(self.val_recon_path, 'epoch_{}.png'.format(epoch)))
		else:
			if train:
				self.monitor_train_recon.add(epoch, self.scale_back_var(img_recon))
			else:
				self.monitor_val_recon.add(epoch, self.scale_back_var(img_recon))

	def train(self, epoch):
		pbar = trange(self.iterations_per_epoch//10, desc='Train at epoch '+str(epoch), disable=self.comm.rank > 0)
		epoch_loss = 0

		if epoch in self.learning_rate_decay_epochs:
			self.solver.set_learning_rate(self.solver.learning_rate()*self.learning_rate_decay_factor)

		for i in pbar:
			# import pdb; pdb.set_trace()
			data = self.data_loader.next()
			if self.dataset_name == 'imagenet':
				img_var = nn.Variable(data[0].shape)
				img_var.data = data[0]
			else:
				img_var = self.convert_to_var(data[0])
			loss, recon_loss, perplexity, img_recon = self.compute_loss(img_var)
			if i == 0:
				self.log_image(epoch, img_recon, train=True)

			pbar.set_description('Batch Loss: {}'.format(loss.d))
			epoch_loss += loss.d

			self.solver.set_parameters(nn.get_parameters(), reset=False, retain_state=True)
			self.solver.zero_grad()
			
			loss.backward(clear_buffer=True)

			params = [x.grad for x in nn.get_parameters().values()]
			self.comm.all_reduce(params, division=False, inplace=True)

			self.solver.weight_decay(self.weight_decay)
			self.solver.update()

		avg_epoch_loss = epoch_loss/self.iterations_per_epoch 
		self.log_loss(epoch, avg_epoch_loss, train=True)

	def validate(self, epoch):
		pbar = trange(self.val_iterations_per_epoch, desc='Validate at epoch '+str(epoch), disable=self.comm.rank > 0)
		epoch_loss = 0

		for i in pbar:
			data = self.val_data_loader.next()
			if self.dataset_name == 'imagenet':
				img_var = nn.Variable(data[0].shape)
				img_var.data = data[0]
			else:
				img_var = self.convert_to_var(data[0])
			loss, recon_loss, perplexity, img_recon = self.compute_loss(img_var)

			pbar.set_description('Batch Loss: {}'.format(loss.d))
			epoch_loss += loss.d

		avg_epoch_loss = epoch_loss/self.iterations_per_epoch 
		self.log_loss(epoch, avg_epoch_loss, train=False)
		self.log_image(epoch, img_recon, train=False)
