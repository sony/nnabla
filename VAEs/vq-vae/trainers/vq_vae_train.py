import nnabla as nn 
import nnabla.functions as F
import numpy as np
import os

class VQVAEtrainer(object):

	def __init__(self, model, solver, data_loader, logger, logger_recon, config, comm):
		self.model = model
		self.solver = solver
		self.data_loader = data_loader
		self.logger = logger
		self.logger_recon = logger_recon
		self.comm = comm

		self.iterations_per_epoch = int(np.ceil(config['dataset']['train_size']/config['train']['batch_size']))
		self.weight_decay = config['train']['weight_decay']

		self.data_variance = np.var(self.data_loader._data_source._images/255.0)
		print("Variance:", self.data_variance)

	def save_checkpoint(self, path, epoch):
		file_name = os.path.join(path, 'epoch_'+str(epoch))
		os.makedirs(file_name, exist_ok=True)
		nn.save_parameters(os.path.join(file_name, 'params.h5'))
		self.solver.save_states(os.path.join(file_name, 'solver.h5'))

	def load_checkpoint(self,path):
		nn.load_parameters(os.path.join(path, 'params.h5'))
		self.solver.load_states(os.path.join(path, 'solver.h5'))

	def convert_to_var(self, img):
		# Expected img: Numpy array with shape (N,3,32,32)
		if not np.all(img < 1):
			img = img/255.0
		img = (img - 0.5)/1
		return nn.Variable.from_numpy_array(img)

	def scale_back_var(self, img_var):
		img = img_var.d + 0.5
		img = (img - img.min(axis=(2,3)).reshape((img.shape[0:2]+(1,1))))/img.max(axis=(2,3)).reshape((img.shape[0:2]+(1,1)))
		return img

	def compute_loss(self, img_var, iteration):
		vq_loss, img_recon, perplexity = self.model(img_var, iteration)
		recon_loss = F.mean(F.squared_error(img_recon,img_var))/self.data_variance
		loss = recon_loss + vq_loss
		if np.isnan(loss.d):
			import pdb; pdb.set_trace()
		return loss, recon_loss, perplexity, img_recon

	def train(self, iteration):

		for i in range(self.iterations_per_epoch):
			data = self.data_loader.next()
			img_var = self.convert_to_var(data[0])
			loss, recon_loss, perplexity, img_recon = self.compute_loss(img_var, iteration)
			self.logger.add(iteration, loss.d)
			self.logger_recon.add(iteration, self.scale_back_var(img_recon))

			self.solver.set_parameters(nn.get_parameters(), reset=False, retain_state=True)
			self.solver.zero_grad()
			
			loss.backward(clear_buffer=True)

			params = [x.grad for x in nn.get_parameters().values()]
			self.comm.all_reduce(params, division=False, inplace=True)

			self.solver.weight_decay(self.weight_decay)
			self.solver.update()
			iteration += 1

		return iteration