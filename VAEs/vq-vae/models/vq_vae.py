import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I
import numpy as np

np.random.seed(1)

class ResidualStack(object):
	
	def __init__(self, in_channels, num_hidden, num_res_layers, rng=313):
		self.in_channels = in_channels
		self.num_hidden = num_hidden
		self.num_res_layers = num_res_layers
		self.rng = rng

	def __call__(self, x):
		out = x
		for i in range(self.num_res_layers):
			out = self.res_block(out, scope_name='res_block_'+str(i))
		return F.relu(out)

	def res_block(self, x, scope_name='res_block'):
		with nn.parameter_scope(scope_name):
			out = F.relu(x, inplace=True)
			out = PF.convolution(out, self.num_hidden, (3,3),
			 stride=(1,1), pad=(1,1), with_bias=False, name='conv_1', rng=self.rng)
			out = PF.batch_normalization(out, name='bn_1')
			out = F.relu(out, inplace=True)
			out = PF.convolution(out, self.num_hidden, (1,1),
			 stride=(1,1), with_bias=False, name='conv_2', rng=self.rng)
			out = PF.batch_normalization(out, name='bn_2')
		return x + out

class VectorQuantizer(object):

    def __init__(self, embedding_dim, num_embedding, commitment_cost, rng,
                 scope_name='vector_quantizer'):
        self.embedding_dim = embedding_dim
        self.num_embedding = num_embedding
        self.commitment_cost = commitment_cost
        self.rng = rng
        self.scope_name = scope_name

        # self.embedding_weight = nn.Variable.from_numpy_array(self.rng.uniform(-1/self.num_embedding, 1/self.num_embedding,
        # 	(self.num_embedding, self.embedding_dim)), need_grad = True)

        with nn.parameter_scope(scope_name):
            self.embedding_weight = nn.parameter.get_parameter_or_create('W', shape=(self.num_embedding, self.embedding_dim),
                                                                         initializer=I.UniformInitializer((-1./self.num_embedding, 1./self.num_embedding), rng=self.rng), need_grad=True)

    def __call__(self, x, iteration):
        x = F.transpose(x, (0, 2, 3, 1))
        x_flat = x.reshape((-1, self.embedding_dim))

        x_flat_squared = F.broadcast(
            F.sum(x_flat**2, axis=1, keepdims=True), (x_flat.shape[0], self.num_embedding))
        emb_wt_squared = F.transpose(
            F.sum(self.embedding_weight**2, axis=1, keepdims=True), (1, 0))

        distances = x_flat_squared + emb_wt_squared - 2 * \
            F.batch_matmul(x_flat, F.transpose(self.embedding_weight, (1, 0)))

        _, encoding_indices = F.min(
            distances, with_index=True, axis=1, keepdims=True)
        encoding_indices.need_grad = False
        quantized = F.embed(encoding_indices.reshape(
            encoding_indices.shape[:-1]), self.embedding_weight).reshape(x.shape)

        encodings = F.one_hot(encoding_indices, (self.num_embedding,))
        # quantized = F.batch_matmul(encodings, self.embedding_weight).reshape(x.shape)
        # import pdb; pdb.set_trace()
        e_latent_loss = F.mean(F.squared_error(
            quantized.get_unlinked_variable(need_grad=False), x))
        q_latent_loss = F.mean(F.squared_error(
            quantized, x.get_unlinked_variable(need_grad=False)))
        loss = q_latent_loss + self.commitment_cost*e_latent_loss

        quantized = x + (quantized - x).get_unlinked_variable(need_grad=False)
        # quantized.need_grad = False
        # import pdb; pdb.set_trace()
        avg_probs = F.mean(encodings, axis=0)
        perplexity = F.exp(-F.sum(avg_probs*F.log(avg_probs+1.0e-10)))
        if np.isnan(loss.d) or np.any(np.isnan(quantized.d)):
            import pdb
            pdb.set_trace()
        # import pdb
        # pdb.set_trace()

        return loss, F.transpose(quantized, (0, 3, 1, 2)), perplexity, encodings


class VectorQuantizerEMA(object):

	def __init__(self, embedding_dim, num_embedding, commitment_cost,rng, 
		decay=0.99, epsilon = 1e-5, scope_name = 'VectorQuantizerEMA'):

		self.embedding_dim = embedding_dim
		self.num_embedding = num_embedding
		self.decay = decay
		self.commitment_cost = commitment_cost
		self.rng = rng
		self.epsilon = epsilon

		# with nn.parameter_scope(scope_name):
		self.embedding_weight = nn.parameter.get_parameter_or_create('emb/W', shape=(self.num_embedding, self.embedding_dim),
			initializer=I.NormalInitializer(rng=self.rng))

		self.ema_cluster_size = nn.parameter.get_parameter_or_create('emb/cluster_size', shape = (self.num_embedding,), 
			initializer = I.ConstantInitializer(0))

		self.ema_w = nn.parameter.get_parameter_or_create('emb/ema_w', shape=(self.num_embedding, self.embedding_dim),
			initializer=I.NormalInitializer(rng=self.rng))			
		# import pdb; pdb.set_trace()

	def __call__(self, x, iteration, is_training=True):
		x = F.transpose(x, (0,2,3,1))
		x_flat = x.reshape((-1, self.embedding_dim))

		try:
			x_flat_squared = F.broadcast(F.sum(x_flat**2, axis=1, keepdims=True), (x_flat.shape[0], self.num_embedding))
			emb_wt_squared = F.transpose(F.sum(self.embedding_weight**2, axis=1, keepdims=True), (1,0))
			distances = x_flat_squared + emb_wt_squared - 2*F.batch_matmul(x_flat, F.transpose(self.embedding_weight, (1,0)))
		except:
			import pdb; pdb.set_trace()
		_, encoding_indices = F.min(distances, with_index=True, axis=1, keepdims=True)
		encoding_indices.need_grad = False
		# quantized = F.embed(encoding_indices.reshape(encoding_indices.shape[:-1]), self.embedding_weight).reshape(x.shape)
		encodings = F.one_hot(encoding_indices, (self.num_embedding,))
		# e_latent_loss = F.mean(F.squared_error(quantized.get_unlinked_variable(need_grad=False), x))

		try:
			if is_training:
				updated_ema_cluster_size = self.decay*self.ema_cluster_size + (1-self.decay)*F.sum(encodings, axis=0)
				dw = F.batch_matmul(F.transpose(encodings, (1,0)), x_flat)
				updated_ema_w = self.decay*self.ema_w+ (1-self.decay)*dw
				
				flag = F.sum(updated_ema_cluster_size).d
				updated_ema_cluster_size = ((self.ema_cluster_size + self.epsilon)/(flag + self.num_embedding +self.epsilon)*flag)
				# import pdb; pdb.set_trace()
				self.embedding_weight = updated_ema_w / updated_ema_cluster_size.reshape(self.ema_cluster_size.shape+(1,))
			# import pdb; pdb.set_trace()
		except:
			import pdb; pdb.set_trace()

		# import pdb; pdb.set_trace()
		quantized = F.embed(encoding_indices.reshape(encoding_indices.shape[:-1]), self.embedding_weight).reshape(x.shape)
		e_latent_loss = F.mean(F.squared_error(quantized.get_unlinked_variable(need_grad=False), x))

		loss = self.commitment_cost*e_latent_loss
		import pdb; pdb.set_trace()

		quantized = x + (quantized - x).get_unlinked_variable()
		avg_probs = F.mean(encodings, axis=0)
		perplexity = F.exp(-F.sum(avg_probs*F.log(avg_probs+1.0e-10)))
		if np.isnan(loss.d):
			import pdb; pdb.set_trace()
		# import pdb; pdb.set_trace()

		return loss, F.transpose(quantized, (0,3,1,2)), perplexity, encodings

class Model(object):

	def __init__(self, config, training = True):
		self.in_channels = config['model']['in_channels']
		self.num_hidden = config['model']['num_hidden']
		self.num_res_layers = config['model']['num_res_layers']
		self.rng = np.random.RandomState(config['model']['rng'])

		self.encoder_res_stack = ResidualStack(in_channels=self.num_hidden,
			num_hidden=self.num_hidden, num_res_layers=self.num_res_layers,
			rng=self.rng)

		self.decoder_res_stack = ResidualStack(in_channels=self.num_hidden,
			num_hidden=self.num_hidden, num_res_layers=self.num_res_layers,
			rng=self.rng)

		self.num_embedding = config['model']['num_embeddings']
		self.embedding_dim = config['model']['embedding_dim']
		self.commitment_cost = config['model']['commitment_cost']
		self.decay = config['model']['decay']

		self.training = training
		self.vq = VectorQuantizer(self.embedding_dim, self.num_embedding, self.commitment_cost, self.rng)

	def encoder(self, x, iteration):
		with nn.parameter_scope('encoder'):
			out = PF.convolution(x, self.num_hidden, (4,4), stride=(2,2),
				pad=(1,1), name = 'conv_1', rng=self.rng)
			out = PF.batch_normalization(out)
			out = F.relu(out)
			out = PF.convolution(out, self.num_hidden, (4,4), stride=(2,2),
				pad=(1,1), name = 'conv_2', rng=self.rng)

			out = self.encoder_res_stack(out)	
		if np.any(np.isnan(out.d)):
			import pdb; pdb.set_trace()		
		return out

	def decoder(self, x, iteration):
		with nn.parameter_scope('decoder'):
			# out = PF.convolution(x, self.num_hidden, (3,3), stride=(1,1),
			# 	pad=(1,1), name='conv_1', rng=self.rng)
			out = self.decoder_res_stack(x)
			out = F.relu(out)
			out = PF.deconvolution(out, self.num_hidden, (4,4), stride=(2,2),
				pad=(1,1), name='deconv_1', rng=self.rng)
			out = PF.batch_normalization(out)
			out = F.relu(out)
			out = PF.deconvolution(out, self.in_channels, (4,4), stride=(2,2),
				pad=(1,1), name='deconv_2', rng=self.rng) 
			out = F.tanh(out)
			if np.any(np.isnan(out.d)):
				import pdb; pdb.set_trace()

		return out

	def __call__(self, img, iteration):

		with nn.parameter_scope('vq_vae'):
			z = self.encoder(img, iteration)
			# import pdb; pdb.set_trace()
			z = PF.convolution(z, self.embedding_dim, (1,1), stride=(1,1))
			loss, quantized, perplexity, encodings = self.vq(z, iteration)
			img_recon = self.decoder(quantized, iteration)

		return loss, img_recon, perplexity 

