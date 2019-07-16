import nnabla as nn
import nnabla.parametric_functions as PF 
import nnabla.functions as F

import numpy as np 
import os 
from tqdm import trange

class TrainerPrior(object):

	def __init__(self, base_model, pixelcnn_model, solver, data_loader, logger, 
		config, comm, num_classes=10):
		self.base_model = base_model
		self.prior = pixelcnn_model
		self.solver = solver
		self.data_loader = data_loader
		self.logger = logger
		self.comm = comm
		self.num_classes = num_classes

		self.iters_per_epoch = int(np.ceil(config['dataset']['train_size']/config['train']['batch_size']))
		self.weight_decay = config['train']['weight_decay']

		self.data_variance = np.var(self.data_loader._data_source._images/255.0)
		print("Variance:", self.data_variance)

	def load_base_params(self, path):
		nn.load_parameters(os.path.join(path, 'params.h5'))

	def load_checkpoint(self, path):
		nn.load_parameters(os.path.join(path, 'params.h5'))
		self.solver.save_states(os.path.join(params, 'solver.h5'))

	def save_checkpoint(self, path):
		nn.save_parameters(os.path.join(path, 'params.h5'))
		self.solver.load_states(os.path.join(params, 'solver.h5'))

	def forward_pass(self, img_var, labels):
		latents = self.base_model.encoder(img_var, 0)
		latents = latents.get_unlinked_variable(need_grad=False)
		import pdb; pdb.set_trace()
		labels = nn.Variable.from_numpy_array(labels)
		labels = F.one_hot(labels, shape=(self.num_classes,))
		# import pdb; pdb.set_trace()
		latents_recon = self.prior(latents, labels)
		import pdb; pdb.set_trace()

		loss = F.softmax_cross_entropy(latents_recon, latents)

		return loss, latents_recon, latents

	def convert_to_var(self, img):
		img = (img-0.5)/1
		img_var = nn.Variable.from_numpy_array(img)
		return img_var

	def scale_back_var(self, img_var):
		img = img_var.d*0.5 + 1
		return img

	def train(self, epoch):
		epoch_loss = 0
		pbar = trange(self.iters_per_epoch, desc='Training at epoch {}'.format(epoch))
		for i in pbar:
			data = self.data_loader.next()
			img_var = self.convert_to_var(data[0])
			# import pdb; pdb.set_trace()
			loss, latents_recon, latents = self.forward_pass(img_var, data[1]) 

			epoch_loss += loss.d 
			pbar.set_description('Train Batch Loss {}'.format(loss.d))
			self.solver.set_parameters(nn.get_parameters(), reset=False, retain_state=True)
			self.solver.zero_grad()
			loss.backward(clear_buffer=True)

			self.solver.weight_decay(self.weight_decay)
			self.solver.update()

		avg_epoch_loss = epoch_loss/self.iters_per_epoch
		self.logger.add(epoch, avg_epoch_loss)

	def validate(self):
		pass