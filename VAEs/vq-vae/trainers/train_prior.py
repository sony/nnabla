import nnabla as nn
import nnabla.parametric_functions as PF 
import nnabla.functions as F

import numpy as np 
import matplotlib.pyplot as plt
import os 
from tqdm import trange

class TrainerPrior(object):

	def __init__(self, base_model, pixelcnn_model, solver, data_loader, val_loader, monitor_train_loss, 
		monitor_train_recon, monitor_val_loss, monitor_val_recon, config, comm):
		self.base_model = base_model
		self.prior = pixelcnn_model
		self.solver = solver
		self.data_loader = data_loader
		self.val_data_loader = val_loader
		self.monitor_train_loss = monitor_train_loss
		self.monitor_train_recon = monitor_train_recon
		self.monitor_val_loss = monitor_val_loss 
		self.monitor_val_recon = monitor_val_recon
		self.comm = comm
  
		self.num_classes = config['prior']['num_classes']
		self.latent_shape = config['prior']['latent_shape']
		self.num_embedding = self.base_model.num_embedding

		self.batch_size = config['train']['batch_size']
		self.iters_per_epoch = int(np.ceil(config['dataset']['train_size']/config['train']['batch_size']))
		self.weight_decay = config['train']['weight_decay']
  
		self.latent_recon_path = os.path.join(config['monitor']['path'], config['monitor']['prior_recon'])
		os.makedirs(self.latent_recon_path, exist_ok=True)

		assert config['model']['checkpoint'] is not None, 'Prior needs to be trained over a learnt discretized latent space. Please specify checkpoint for a trained VAE'
		self.load_base_params(config['model']['checkpoint'])
  
		self.dataset_name = config['dataset']['name']
		if self.dataset_name != 'imagenet':
			self.val_iterations_per_epoch = int(np.ceil(self.val_data_loader.size/self.val_data_loader.batch_size/comm.n_procs))

	def load_base_params(self, path):
		nn.load_parameters(os.path.join(path, 'params.h5'))
		print(f'VQVAE Checkpoint Loaded! {path}')

	def load_checkpoint(self, path):
		nn.load_parameters(os.path.join(path, 'pixelcnn_params.h5'))
		self.solver.save_states(os.path.join(path, 'pixelcnn_solver.h5'))
		print(f'Checkpoint Loaded! {path}')

	def save_checkpoint(self, path, epoch):
		file_name = os.path.join(path, 'epoch_'+str(epoch))
		os.makedirs(file_name, exist_ok=True)
		nn.save_parameters(os.path.join(file_name, 'pixelcnn_params.h5'))
		self.solver.save_states(os.path.join(file_name, 'pixelcnn_solver.h5'))
  
	def visualize_discrete_img(self, var, epoch, name):
		assert var.ndim < 3, 'The discrete image should only consist of indices of the codebook vectors'
		if var.ndim==2 and var.shape[1] > 1:
			var = F.max(var, axis=1, only_index=True)
   
		var = F.reshape(var, [self.batch_size, 1] + self.latent_shape, inplace=True)
		var = var/self.num_embedding
  
		img = nn.monitor.tile_images(var.d)
		plt.imshow(img, cmap='magma')
		plt.axis('off')
		filename = os.path.join(self.latent_recon_path, f'{name}_{epoch}.png')
		plt.savefig(filename, bbox_inches='tight')
		plt.close()
  
		print(f'Reconstruction saved at {filename}')	
  
	def save_image(self, img, path):
		img = img*0.5 + 0.5
		plt.imshow(nn.monitor.tile_images(img.d))
		plt.axis('off')
		plt.savefig(path, bbox_inches='tight')
		plt.close()
		print('Saving reconstrutions in', path)	

	def forward_pass(self, img_var, labels):
		enc_indices, quantized = self.base_model(img_var, return_encoding_indices=True)

		if self.dataset_name == 'imagenet':
			labels = nn.Variable(labels.shape).apply(data=labels)
		else:
			labels = nn.Variable.from_numpy_array(labels)
		labels = F.one_hot(labels, shape=(self.num_classes,))
		enc_recon = self.prior(quantized, labels)
		loss = F.mean(F.softmax_cross_entropy(enc_recon, enc_indices))

		return loss, enc_indices, enc_recon

	def convert_to_var(self, img):
		img = (img-0.5)/1
		img_var = nn.Variable.from_numpy_array(img)
		return img_var

	def train(self, epoch):
		epoch_loss = 0
		pbar = trange(self.iters_per_epoch, desc='Training at epoch {}'.format(epoch))
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

			if i == 0:
				avg_epoch_loss = epoch_loss/self.iters_per_epoch
				self.monitor_train_loss.add(epoch, avg_epoch_loss)
				self.visualize_discrete_img(enc_indices, epoch, 'original')
				self.visualize_discrete_img(enc_recon, epoch, 'recon')
	
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

		avg_epoch_loss = epoch_loss/self.val_iterations_per_epoch 
		self.monitor_val_loss.add(epoch, avg_epoch_loss)
		self.visualize_discrete_img(enc_indices, epoch, 'val_original')
		self.visualize_discrete_img(enc_recon, epoch, 'val_recon')
  
	def random_generate(self, num_images, path):
		
		# Generate from the uniform prior of the base model
		indices = F.randint(low=0, high=self.num_embedding, shape=[num_images]+self.latent_shape)
		indices = F.reshape(indices, (-1,), inplace=True)
		quantized = F.embed(indices, self.base_model.vq.embedding_weight)
		quantized = F.transpose(quantized.reshape([num_images]+self.latent_shape + [quantized.shape[-1]]), (0,3,1,2))
		
		img_gen_uniform_prior = self.base_model(quantized, quantized_as_input=True, test=True)
  
		# Generate images using pixelcnn prior
		indices = nn.Variable.from_numpy_array(np.zeros(shape=[num_images]+self.latent_shape))
		# indices = F.randint(low=0, high=self.num_embedding, shape=[num_images]+self.latent_shape)
		indices_before = indices.d
		# labels = F.randint(low=0, high=self.num_classes, shape=(num_images,1))
		labels = nn.Variable.from_numpy_array(np.ones((num_images,1))*3 )
		labels = F.one_hot(labels, shape=(self.num_classes,))

		# Sample from pixelcnn - pixel by pixel
		for i in range(self.latent_shape[0]):
			for j in range(self.latent_shape[1]):
				# for c in range(num_images):
				quantized = F.embed(indices.reshape((-1,)), self.base_model.vq.embedding_weight)
				quantized = F.transpose(quantized.reshape([num_images]+self.latent_shape + [quantized.shape[-1]]), (0,3,1,2))
				indices_sample = self.prior(quantized, labels)
				for c in range(indices.shape[0]):
					indices_prob = indices_sample.reshape(indices.shape+(indices_sample.shape[-1],))[c,i,j,:].d
					indices_prob = (indices_prob - indices_prob.min() )/(indices_prob.max()-indices_prob.min())
					indices_prob /= indices_prob.sum()
     
					sampled_idx = np.random.multinomial(1, indices_prob)
					indices[c,i,j] = sampled_idx.argmax()
					print(sampled_idx.argmax(), sampled_idx.max())
					# np.random.multinomial(self.num_embedding, indices_sample[c,i,j].d)
				# indices[:,i,j].d = F.max(indices_sample, axis=1, only_index=True).reshape(indices.shape)[:,i,j].d

		quantized = F.embed(indices.reshape((-1,)), self.base_model.vq.embedding_weight)
		quantized = F.transpose(quantized.reshape([num_images]+self.latent_shape + [quantized.shape[-1]]), (0,3,1,2))

		import pdb; pdb.set_trace()
		img_gen_pixelcnn_prior = self.base_model(quantized, quantized_as_input=True, test=True)

		self.save_image(img_gen_uniform_prior, os.path.join(path, 'generate_uniform.png'))
		self.save_image(img_gen_pixelcnn_prior, os.path.join(path, 'generate_pixelcnn.png'))

		print('Random labels generated for pixelcnn prior:', list(F.max(labels, axis=1, only_index=True).d))
  
