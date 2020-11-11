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

import numpy as np 
import matplotlib.pyplot as plt
import os 
from tqdm import trange

from .base import BaseTrainer

class TrainerPrior(BaseTrainer):

	def __init__(self, base_model, pixelcnn_model, solver, data_loader, val_loader, monitor_train_loss, 
		monitor_train_recon, monitor_val_loss, monitor_val_recon, config, comm, eval=False):
     
		super(TrainerPrior, self).__init__(solver, data_loader, val_loader, monitor_train_loss, monitor_train_recon,
                                     monitor_val_loss, monitor_val_recon, config, comm, eval)
     
		self.base_model = base_model
		self.prior = pixelcnn_model
  
		self.num_classes = config['prior']['num_classes']
		self.latent_shape = config['prior']['latent_shape']
		self.num_embedding = self.base_model.num_embedding
  
		self.latent_recon_path = os.path.join(config['monitor']['path'], config['monitor']['prior_recon'])
		os.makedirs(self.latent_recon_path, exist_ok=True)

		assert config['model']['checkpoint'] is not None, 'Prior needs to be trained over a learnt discretized latent space. Please specify checkpoint for a trained VAE'
		self.load_checkpoint(config['model']['checkpoint'], msg='VQVAE Checkpoint Loaded! {}'.format(config['model']['checkpoint']), load_solver=False)	

	def forward_pass(self, img_var, labels):
		enc_indices, quantized = self.base_model(img_var, return_encoding_indices=True, test=True)
		labels_var = nn.Variable(labels.shape)
		if isinstance(labels, nn.NdArray):
			labels_var.data  = labels
		else:
			labels_var.d = labels
		labels_var = F.one_hot(labels_var, shape=(self.num_classes,))
		enc_recon = self.prior(quantized, labels_var)
		loss = F.mean(F.softmax_cross_entropy(enc_recon, enc_indices))

		return loss, enc_indices, enc_recon

	def train(self, epoch):
		epoch_loss = 0
		pbar = trange(self.iterations_per_epoch, desc='Training at epoch {}'.format(epoch))
		for i in pbar:
			data = self.data_loader.next()
			if self.dataset_name == 'imagenet':
				img_var = nn.Variable(data[0].shape)
				img_var.data = data[0]
			else:
				img_var = self.convert_to_var(data[0])
			loss, enc_indices, enc_recon = self.forward_pass(img_var, data[1]) 

			epoch_loss += loss.d 
			pbar.set_description('Train Batch Loss {}'.format(loss.d))
			self.solver.set_parameters(nn.get_parameters(), reset=False, retain_state=True)
			self.solver.zero_grad()
			loss.backward(clear_buffer=True)

			self.solver.weight_decay(self.weight_decay)
			self.solver.update()

		avg_epoch_loss = epoch_loss/self.iterations_per_epoch
		self.log_loss(epoch, avg_epoch_loss, train=True)
		self.visualize_discrete_image(enc_indices, os.path.join(self.latent_recon_path, 'original_{}.png'.format(epoch)))
		self.visualize_discrete_image(enc_recon, os.path.join(self.latent_recon_path, 'recon_{}.png'.format(epoch)))

	
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
			loss, enc_indices, enc_recon = self.forward_pass(img_var, data[1]) 

			pbar.set_description('Batch Loss: {}'.format(loss.d))
			epoch_loss += loss.d

		avg_epoch_loss = epoch_loss/self.iterations_per_epoch
		self.log_loss(epoch, avg_epoch_loss, train=False)
		self.visualize_discrete_image(enc_indices, os.path.join(self.latent_recon_path, 'original_{}.png'.format(epoch)))
		self.visualize_discrete_image(enc_recon, os.path.join(self.latent_recon_path, 'recon_{}.png'.format(epoch)))
  
	def random_generate(self, num_images, path):
    		
		# Generate from the uniform prior of the base model
		indices = F.randint(low=0, high=self.num_embedding, shape=[num_images]+self.latent_shape)
		indices = F.reshape(indices, (-1,), inplace=True)
		quantized = F.embed(indices, self.base_model.vq.embedding_weight)
		quantized = F.transpose(quantized.reshape([num_images]+self.latent_shape + [quantized.shape[-1]]), (0,3,1,2))
  
		img_gen_uniform_prior = self.base_model(quantized, quantized_as_input=True, test=True)
  
		# Generate images using pixelcnn prior
		indices = nn.Variable.from_numpy_array(np.zeros(shape=[num_images]+self.latent_shape))
		labels = F.randint(low=0, high=self.num_classes, shape=(num_images,1))
		labels = F.one_hot(labels, shape=(self.num_classes,))

		# Sample from pixelcnn - pixel by pixel
		import torch # Numpy behavior is different and not giving correct output
		for i in range(self.latent_shape[0]):
			for j in range(self.latent_shape[1]):
				quantized = F.embed(indices.reshape((-1,)), self.base_model.vq.embedding_weight)
				quantized = F.transpose(quantized.reshape([num_images]+self.latent_shape + [quantized.shape[-1]]), (0,3,1,2))
				indices_sample = self.prior(quantized, labels)
				indices_prob = F.reshape(indices_sample, indices.shape+(indices_sample.shape[-1],), inplace=True)[:,i,j]
				indices_prob = F.softmax(indices_prob)

				indices_prob_tensor = torch.from_numpy(indices_prob.d)
				sample = indices_prob_tensor.multinomial(1).squeeze().numpy()
				indices[:,i,j] = sample
    
		print(indices.d)
		quantized = F.embed(indices.reshape((-1,)), self.base_model.vq.embedding_weight)
		quantized = F.transpose(quantized.reshape([num_images]+self.latent_shape + [quantized.shape[-1]]), (0,3,1,2))
  
		img_gen_pixelcnn_prior = self.base_model(quantized, quantized_as_input=True, test=True)

		self.save_image(img_gen_uniform_prior, os.path.join(path, 'generate_uniform.png'))
		self.save_image(img_gen_pixelcnn_prior, os.path.join(path, 'generate_pixelcnn.png'))

		print('Random labels generated for pixelcnn prior:', list(F.max(labels, axis=1, only_index=True).d))
  
