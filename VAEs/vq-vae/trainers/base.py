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

import nnabla as nn
import nnabla.functions as F

import os
import numpy as np
import matplotlib.pyplot as plt

class BaseTrainer(object):
	
	def __init__(self, solver, data_loader, val_loader, monitor_train_loss, 
		monitor_train_recon, monitor_val_loss, monitor_val_recon, config, comm, eval=False):
		
		self.comm = comm
		if not eval:
			self.solver = solver
			self.data_loader = data_loader
			self.val_data_loader = val_loader
			self.monitor_train_loss = monitor_train_loss
			self.monitor_train_recon = monitor_train_recon
			self.monitor_val_loss = monitor_val_loss 
			self.monitor_val_recon = monitor_val_recon
	
			self.iterations_per_epoch = int(np.ceil(self.data_loader.size/self.data_loader.batch_size/comm.n_procs))
			self.weight_decay = config['train']['weight_decay']
			self.learning_rate_decay_epochs = config['train']['learning_rate_decay_epochs']
			self.learning_rate_decay_factor = config['train']['learning_rate_decay_factor']

			self.batch_size = config['train']['batch_size']
			self.dataset_name = config['dataset']['name']
			if self.dataset_name != 'imagenet':
				self.val_iterations_per_epoch = int(np.ceil(self.val_data_loader.size/self.val_data_loader.batch_size/comm.n_procs))
	
	def save_checkpoint(self, path, epoch, msg='', pixelcnn=False):
		file_name = os.path.join(path, 'epoch_'+str(epoch))
		os.makedirs(file_name, exist_ok=True)
		if pixelcnn:
			nn.save_parameters(os.path.join(file_name, 'pixelcnn_params.h5'))
			self.solver.save_states(os.path.join(file_name, 'pixelcnn_solver.h5'))
		else:
			nn.save_parameters(os.path.join(file_name, 'params.h5'))
			self.solver.save_states(os.path.join(file_name, 'solver.h5'))
		print(msg)

	def load_checkpoint(self, path, msg='', pixelcnn=False, load_solver=True):
		if pixelcnn:
			nn.load_parameters(os.path.join(path, 'pixelcnn_params.h5'))
			if load_solver:
				self.solver.save_states(os.path.join(path, 'pixelcnn_solver.h5'))
		else:
			nn.load_parameters(os.path.join(path, 'params.h5'))
			if load_solver:
				self.solver.load_states(os.path.join(path, 'solver.h5'))
		print(msg)

	def convert_to_var(self, img):
		if not np.all(img < 1):
			img = img/255.0
		img = (img - 0.5)/0.5
		return nn.Variable.from_numpy_array(img)

	def save_image(self, img, filename):
		img = img*0.5 + 0.5
		plt.imshow(nn.monitor.tile_images(img.d))
		plt.axis('off')
		plt.savefig(filename, bbox_inches='tight')
		plt.close()
		print('Saving reconstrutions in', filename)
  
	def visualize_discrete_image(self, var, filename):
		assert var.ndim < 3, 'The discrete image should only consist of indices of the codebook vectors'
		if var.ndim==2 and var.shape[1] > 1:
			var = F.max(var, axis=1, only_index=True)
   
		var = F.reshape(var, [-1, 1] + self.latent_shape, inplace=True)
		var = var/self.num_embedding
  
		img = nn.monitor.tile_images(var.d)
		plt.imshow(img, cmap='magma')
		plt.axis('off')
		plt.savefig(filename, bbox_inches='tight')
		plt.close()
  
		print('Reconstruction saved at {}'.format(filename))
		
	def log_loss(self, epoch, loss, train=True):
		if train:
			self.monitor_train_loss.add(epoch, loss)
		else:
			self.monitor_val_loss.add(epoch, loss)
   