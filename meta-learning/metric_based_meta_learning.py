import numpy as np
import time

import numpy.random as rng
import numpy as np
import os
import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import manifold


import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I
import nnabla.solver as S


def conv_initializer(f_in, n_out, base_axis, kernel, mode):
    '''
    Conv initializer function
        This function returns various types of initialization for weights and bias parameters in convolution layer.

        Args:
            f_in (~nnabla.Variable): input variable.
            n_out (int) : number of output neurons per data.
            base_axis (int): dimensions up to base_axis are treated as the sample dimensions.
            kernel (tuple of int) : convolution kernel size.
            mode (str) : type of initialization to use.
        Returns:
            w (~nnabla.initializer.BaseInitializer): weight parameters
            b (~nnabla.initializer.BaseInitializer): bias parameters
    '''
    if mode == 'nnabla':
        # https://github.com/sony/nnabla/blob/master/python/src/nnabla/parametric_functions.py, line415, 417
        # https://github.com/sony/nnabla/blob/master/python/src/nnabla/initializer.py, line224. 121
        # uniform_lim_glorot = uniform(sqrt(6/(fin+fout)))
        n_input_plane = f_in.shape[base_axis]
        s = np.sqrt(6.0 / (n_input_plane * np.prod(kernel) + n_out))
        w = I.UniformInitializer([-s, s])
        b = I.ConstantInitializer(0)
        return w, b


def conv4(x, test=False, init_type='nnabla'):
    '''
    Embedding function
        This network is a typical embedding network for the one-shot learning benchmark task.
        Args:
            x (~nnabla.Variable) : input image.
            test (boolean) : whether test of training for
        Returns:
            h (~nnabla.Variable): embedding vector.

    '''
    h = x
    for i in range(4):
        w, b = conv_initializer(h, 64, 1, [3, 3], init_type)
        h = PF.convolution(h, 64, [3, 3], w_init=w, b_init=b, pad=[
                           1, 1], name='conv' + str(i))
        h = PF.batch_normalization(h, batch_stat=not test, name='bn' + str(i))
        #h = F.relu(h)
        # To avoid all 0 embedding vector for cosine similarity.
        if i != 3:
            h = F.relu(h)
        h = F.max_pooling(h, [2, 2])
    h = F.reshape(h, [h.shape[0], np.prod(h.shape[1:])])
    return h


def similarity(fq, fs, metric):
    '''
    Similarity function
        This function provides the various types of metrics to measure the similarity between the query and support.
        We provide euclidean distance and cosine similarity.

        Args:
            fq (~nnabla.Variable) : input query image.
            fs (~nnabla.Variable): input support image.
            metric (str): similarity metric to use.
        Returns:
            h (~nnabla.Variable): computed similarity.
    '''

    if metric == "euclid":  # euclid similarity of (n_query, n_support) tensor
        h = -F.sum((fs - fq) ** 2.0, axis=2)
    elif metric == "cosine":
        qs = F.sum(fq * fs, axis=2)
        qq = F.sum(fq ** 2.0, axis=2)
        ss = F.sum(fs ** 2.0, axis=2)
        h = qs * (ss ** -0.5) * (qq ** -0.5)
    return h


def net(n_class, xs, xq, init_type='nnabla', embedding='conv4', net_type='prototypical', distance='euclid', test=False):
    '''
    Similarity net function
        This function implements the network with settings as specified.

        Args:
            n_class (int): number of classes. Typical setting is 5 or 20.
            xs (~nnabla.Variable): support images.
            xq (~nnabla.Variable): query images.
            init_type (str, optional): initialization type for weights and bias parameters. See conv_initializer function.
            embedding(str, optional): embedding network.
            distance (str, optional): similarity metric to use. See similarity function.
            test (bool, optional): switch flag for training dataset and test dataset
        Returns:
            h (~nnabla.Variable): output variable indicating similarity between support and query.
    '''

    # feature embedding for supports and queries
    n_shot = xs.shape[0] / n_class
    n_query = xq.shape[0] / n_class
    if embedding == 'conv4':
        fs = conv4(xs, test, init_type)  # tensor of (n_support, fdim)
        fq = conv4(xq, test, init_type)  # tensor of (n_query, fdim)

    if net_type == 'matching':
        # This example does not include the full-context-embedding of matching networks.
        fs = F.reshape(fs, (1,) + fs.shape)  # (1, n_way, fdim)
        # (n_way*n_query, 1, fdim)
        fq = F.reshape(fq, (fq.shape[0], 1) + fq.shape[1:])
        h = similarity(fq, fs, distance)
        h = h - F.mean(h, axis=1, keepdims=True)
        if 1 < n_shot:
            h = F.minimum_scalar(F.maximum_scalar(h, -35), 35)
            h = F.softmax(h)
            h = F.reshape(h, (h.shape[0], n_class, n_shot))
            h = F.mean(h, axis=2)
            # Reverse to logit to use same softmax cross entropy
            h = F.log(h)
    elif net_type == 'prototypical':
        if 1 < n_shot:
            fs = F.reshape(fs, (n_class, n_shot) + fs.shape[1:])
            fs = F.mean(fs, axis=1)
        fs = F.reshape(fs, (1,) + fs.shape)  # (1, n_way, fdim)
        # (n_way*n_query, 1, fdim)
        fq = F.reshape(fq, (fq.shape[0], 1) + fq.shape[1:])
        h = similarity(fq, fs, distance)
        h = h - F.mean(h, axis=1, keepdims=True)

    return h


def augmentation(data):
    # This function generates augmented class and data
    data_1 = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_1[i][j] = np.rot90(data[i][j], 1)

    data_2 = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_2[i][j] = np.rot90(data[i][j], 2)

    data_3 = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_3[i][j] = np.rot90(data[i][j], 3)

    return np.concatenate((data, data_1, data_2, data_3))


def get_embeddings(batch, embedding):
    # This function outputs embedding vectors of inputted batch.
    x = nn.Variable(batch.shape)
    h = embedding(x, test=True)
    x.d = batch
    h.forward()
    return h.d


def get_tsne(u):
    # This function calculates tsne projection
    t0 = time.time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    v = tsne.fit_transform(u)
    t1 = time.time()
    print("t-SNE: %.2g sec" % (t1 - t0))
    return v


def plot_tsne(x, y, color, image_file):
    # This function plots the tsne visualization
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=color, cmap=plt.cm.Spectral, marker='.')
    plt.savefig(image_file)


def get_args():
    parser = argparse.ArgumentParser(description="Fewshot Learning")
    parser.add_argument("--n_class", "-nw", type=int, default=5)
    parser.add_argument("--n_shot", "-ns", type=int, default=1)
    parser.add_argument("--n_query", "-nq", type=int, default=5)
    parser.add_argument("--n_class_tr", "-nwt", type=int, default=60)
    parser.add_argument("--n_shot_tr", "-nst", type=int, default=1)
    parser.add_argument("--n_query_tr", "-nqt", type=int, default=5)
    parser.add_argument("--dataset", "-ds", type=str, default="omniglot")
    parser.add_argument("--dataset_root", "-dr", type=str, default=".")
    parser.add_argument("--init_type", "-i", type=str, default='nnabla')
    parser.add_argument("--embedding", "-e", type=str, default='conv4')
    parser.add_argument("--net_type", "-n", type=str, default='prototypical')
    parser.add_argument("--metric", "-d", type=str, default='euclid')
    parser.add_argument("--max_iteration", "-mi", type=int, default=20001)
    parser.add_argument("--iter_per_epoch", "-ei", type=int, default=100)
    parser.add_argument("--iter_per_valid", "-vi", type=int, default=1000)
    parser.add_argument("--n_episode_for_valid", "-ev", type=int, default=1000)
    parser.add_argument("--n_episode_for_test", "-et", type=int, default=1000)
    parser.add_argument("--lr_decay_interval", "-lrdi", type=int, default=2000)
    parser.add_argument("--lr_decay", "-lrd", type=float, default=0.5)
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        help="Extension modules. ex) 'cpu', 'cudnn'.")
    parser.add_argument("--device-id", "-gid", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cuda.cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--work-dir", "-w", type=str, default="tmp.result/")
    args = parser.parse_args()
    return args


def load_omniglot(dataset_root):
    x_train, _ = np.load(dataset_root + "/train.npy", allow_pickle=True)
    x_valid, _ = np.load(dataset_root + "/val.npy", allow_pickle=True)
    x = np.r_[x_train, x_valid]
    from nnabla.utils.image_utils import imresize
    shape_x = (1, 28, 28)
    x_resized = np.zeros([1623, 20, 28, 28])
    for xi, ri in zip(x, x_resized):
        for xij, rij in zip(xi, ri):
            rij[:] = imresize(xij, size=(shape_x[2], shape_x[1])) / 255.
    data = augmentation(x_resized)
    rng = np.random.RandomState(706)
    data = rng.permutation(data)
    data = data.reshape((1,) + data.shape).transpose(1, 2, 0, 3, 4)
    train_data = data[:4112]
    val_data = data[4112:4800]
    test_data = data[4800:]
    return train_data, val_data, test_data


class EpisodeGenerator:
    def __init__(self, n_class, n_shot, n_query, shape_x, dataset):
        """
        Create episode function
            Args:
                n_class (int): number of support classes, generally called n_way in one-shot litterateur
                n_shot (int): number of shots per class
                n_query (int): number of queries per class.
                shape_x : dimension of the input image.
                dataset(nd_array): nd_array of (class, sample, shape)
        """
        self.n_class = n_class
        self.n_shot = n_shot
        self.n_query = n_query
        self.shape_x = shape_x
        self.dataset = dataset
        self.rng = np.random.RandomState(706)

    def next(self):
        """
        Create episode function
            Args:
                n_class (int): number of support classes, generally called n_way in one-shot litterateur
                n_shot (int): number of shots per class
                n_query (int): number of queries per class.
                shape_x : dimension of the input image.
                dataset_mode (str, optional): data split to use.
            Returns:
                x_s (~nnabla.Variable): support images
                x_q (~nnabla.Variable): query images
                y_q (~nnabla.Variable): query class_id in support sets
        """
        # parameters
        n_class = self.n_class
        n_shot = self.n_shot
        n_query = self.n_query
        shape_x = self.shape_x

        # dataset selection
        dataset = self.dataset

        # memory allocations
        x_s = np.zeros((n_class * n_shot, ) + shape_x)
        x_q = np.zeros((n_class * n_query, ) + shape_x, dtype=np.int)
        y_q = np.zeros((n_class * n_query, 1), dtype=np.int)

        # episode classes
        support_class_ids = rng.choice(dataset.shape[0], n_class, False)

        # operate for each support class
        for i, support_class_id in enumerate(support_class_ids):

            # select support class
            xi = dataset[support_class_id]
            n_sample = xi.shape[0]

            if n_sample < n_shot + n_query:
                print("Error: too few samples.")
                exit()

            # sample indices for support and queries
            sample_ids = rng.choice(n_sample, n_shot + n_query, False)
            s_sample_ids = sample_ids[:n_shot]
            q_sample_ids = sample_ids[n_shot:]

            # support set
            for j in range(n_shot):
                si = i * n_shot + j
                x_s[si] = xi[s_sample_ids[j]]

            # query set
            for j in range(n_query):
                qi = i * n_query + j
                x_q[qi] = xi[q_sample_ids[j]]
                y_q[qi] = i

        return x_s, x_q, y_q


def train_and_eval():

    # Settings
    args = get_args()
    n_class = args.n_class
    n_shot = args.n_shot
    n_query = args.n_query
    n_class_tr = args.n_class_tr
    n_shot_tr = args.n_shot_tr
    if n_shot_tr == 0:
        n_shot_tr = n_shot
    n_query_tr = args.n_query_tr
    if n_query_tr == 0:
        n_query_tr = n_query

    dataset = args.dataset
    dataset_root = args.dataset_root

    init_type = args.init_type
    embedding = args.embedding
    net_type = args.net_type
    metric = args.metric

    max_iteration = args.max_iteration
    lr_decay_interval = args.lr_decay_interval
    lr_decay = args.lr_decay
    iter_per_epoch = args.iter_per_epoch
    iter_per_valid = args.iter_per_valid
    n_episode_for_valid = args.n_episode_for_valid
    n_episode_for_test = args.n_episode_for_test
    work_dir = args.work_dir

    # Set context
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Monitor outputs
    from nnabla.monitor import Monitor, MonitorSeries
    monitor = Monitor(args.work_dir)
    monitor_loss = MonitorSeries(
        "Training loss", monitor, interval=iter_per_epoch)
    monitor_valid_err = MonitorSeries(
        "Validation error", monitor, interval=iter_per_valid)
    monitor_test_err = MonitorSeries("Test error", monitor)
    monitor_test_conf = MonitorSeries("Test error confidence", monitor)

    # Output files
    param_file = work_dir + "params.h5"
    tsne_file = work_dir + "tsne.png"

    # Load data
    shape_x = (1, 28, 28)
    train_data, valid_data, test_data = load_omniglot(
        dataset_root + "/omniglot/data/")
    train_episode_generator = EpisodeGenerator(
        n_class_tr, n_shot_tr, n_query_tr, shape_x, train_data)
    valid_episode_generator = EpisodeGenerator(
        n_class, n_shot, n_query, shape_x, valid_data)
    test_episode_generator = EpisodeGenerator(
        n_class, n_shot, n_query, shape_x, test_data)

    # Build training model
    xs_t = nn.Variable((n_class_tr * n_shot_tr, ) + shape_x)
    xq_t = nn.Variable((n_class_tr * n_query_tr, ) + shape_x)
    hq_t = net(n_class_tr, xs_t, xq_t, init_type,
               embedding, net_type, metric, False)
    yq_t = nn.Variable((n_class_tr * n_query_tr, 1))
    loss_t = F.mean(F.softmax_cross_entropy(hq_t, yq_t))

    # Build evaluation model
    xs_v = nn.Variable((n_class * n_shot, ) + shape_x)
    xq_v = nn.Variable((n_class * n_query, ) + shape_x)
    hq_v = net(n_class, xs_v, xq_v, init_type,
               embedding, net_type, metric, True)
    yq_v = nn.Variable((n_class * n_query, 1))
    err_v = F.mean(F.top_n_error(hq_v, yq_v, n=1))

    # Setup solver
    solver = S.Adam(1.0e-3)
    solver.set_parameters(nn.get_parameters())
    learning_rate_decay_activate = True

    # Training loop
    train_losses = []
    best_err = 1.0
    for i in range(max_iteration):

        # Decay learning rate
        if learning_rate_decay_activate and ((i + 1) % lr_decay_interval == 0):
            solver.set_learning_rate(solver.learning_rate() * lr_decay)

        # Create an episode
        xs_t.d, xq_t.d, yq_t.d = train_episode_generator.next()

        # Training by the episode
        solver.zero_grad()
        loss_t.forward(clear_no_need_grad=True)
        loss_t.backward(clear_buffer=True)
        solver.update()
        train_losses.append(loss_t.d.copy())

        # Evaluation
        if (i + 1) % iter_per_valid == 0:
            train_loss = np.mean(train_losses)
            train_losses = []
            valid_errs = []
            for k in range(n_episode_for_valid):
                xs_v.d, xq_v.d, yq_v.d = valid_episode_generator.next()
                err_v.forward(clear_no_need_grad=True, clear_buffer=True)
                valid_errs.append(np.float(err_v.d.copy()))
            valid_err = np.mean(valid_errs)

            #monitor_loss.add(i + 1, train_loss)
            monitor_valid_err.add(i + 1, valid_err * 100)
            if valid_err < best_err:
                best_err = valid_err
                nn.save_parameters(param_file)

    # Final evaluation
    nn.load_parameters(param_file)
    v_errs = []
    for k in range(n_episode_for_test):
        xs_v.d, xq_v.d, yq_v.d = test_episode_generator.next()
        err_v.forward(clear_no_need_grad=True, clear_buffer=True)
        v_errs.append(np.float(err_v.d.copy()))
    v_err = np.mean(v_errs)
    v_err_conf = 1.96 * np.std(v_errs) / np.sqrt(n_episode_for_test)
    monitor_test_err.add(0, v_err * 100)
    monitor_test_conf.add(0, v_err_conf)

    # Visualization
    n_class = 50
    n_sample = 20
    batch = test_data[:n_class].reshape(n_class * n_sample, 1, 28, 28)
    label = []
    for i in range(n_class):
        label.extend(np.ones(n_sample) * (i % 50))
    u = get_embeddings(batch, conv4)
    v = get_tsne(u)
    plot_tsne(v[:, 0], v[:, 1], label, tsne_file)


if __name__ == '__main__':
    train_and_eval()
