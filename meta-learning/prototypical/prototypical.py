# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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
from _checkpoint_nnp_util import save_nnp
import nnabla.utils.save as save
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I
import nnabla.solver as S
from nnabla.monitor import Monitor, MonitorSeries


def conv4(x, test=False):
    '''
    Embedding function
        This network is a typical embedding network for the one-shot learning benchmark task.
        Args:
            x (~nnabla.Variable) : input images.
            test (boolean) : whether test or training
        Returns:
            h (~nnabla.Variable): embedding vector.

    '''
    h = x
    for i in range(4):
        h = PF.convolution(h, 64, [3, 3], pad=[1, 1], name='conv' + str(i))
        h = PF.batch_normalization(h, batch_stat=not test, name='bn' + str(i))
        h = F.relu(h)
        h = F.max_pooling(h, [2, 2])
    h = F.reshape(h, [h.shape[0], np.prod(h.shape[1:])])
    return h


def similarity(fq, fs, metric):
    '''
    Similarity function
        This function provides the various types of metrics to measure the similarity between the query and support.
        We provide Euclidean distance and cosine similarity.

        Args:
            fq (~nnabla.Variable) : input query images.
            fs (~nnabla.Variable): input support images.
            metric (str): similarity metric to use.
        Returns:
            h (~nnabla.Variable): computed similarity.
    '''

    # Euclidian-distance based similarity of (n_query, n_support)
    if metric == "euclid":
        h = -F.sum((fs - fq) ** 2.0, axis=2)
    elif metric == "cosine":
        qs = F.sum(fq * fs, axis=2)
        qq = F.sum(fq ** 2.0, axis=2)
        ss = F.sum(fs ** 2.0, axis=2)
        h = qs * (ss ** -0.5) * (qq ** -0.5)
    return h


def net(n_class, xs, xq, embedding='conv4', net_type='prototypical', distance='euclid', test=False):
    '''
    Similarity net function
        This function implements the network with settings as specified.

        Args:
            n_class (int): number of classes. Typical setting is 5 or 20.
            xs (~nnabla.Variable): support images.
            xq (~nnabla.Variable): query images.
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
        fs = conv4(xs, test)  # (n_support, fdim)
        fq = conv4(xq, test)  # (n_query, fdim)

    if net_type == 'matching':
        # This example does not include the full-context-embedding of matching networks.
        fs = F.reshape(fs, (1,) + fs.shape)
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
        fs = F.reshape(fs, (1,) + fs.shape)
        fq = F.reshape(fq, (fq.shape[0], 1) + fq.shape[1:])
        h = similarity(fq, fs, distance)
        h = h - F.mean(h, axis=1, keepdims=True)

    return h


def augmentation(data):
    # This function generates augmented class and data
    augmented_data = np.zeros((data.shape[0] * 4,) + data.shape[1:])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            augmented_data[i*4][j] = data[i][j]
            augmented_data[i*4+1][j] = np.rot90(data[i][j], 1)
            augmented_data[i*4+2][j] = np.rot90(data[i][j], 2)
            augmented_data[i*4+3][j] = np.rot90(data[i][j], 3)

    return augmented_data


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
    parser = argparse.ArgumentParser(
        description="Prototypical Networks for Few-shot Learning")

    # Meta-evaluation/meta-test settings
    parser.add_argument("--n_class", "-nw", type=int, default=5,
                        help='number of classes used in few-shot classification (N for N-way classification)')
    parser.add_argument("--n_shot", "-ns", type=int, default=1,
                        help='number of examples in each few-shot class (K for K-shot learning)')
    parser.add_argument("--n_query", "-nq", type=int, default=5,
                        help='number of examples for few-shot classification')

    # Meta learning settings
    parser.add_argument("--n_class_tr", "-nwt", type=int, default=60,
                        help='number of classes in each episode of meta-learning')
    parser.add_argument("--n_shot_tr", "-nst", type=int, default=1,
                        help='number of examples in each class of each episode')
    parser.add_argument("--n_query_tr", "-nqt", type=int, default=5,
                        help='number of examples in each episode')

    # Dataset settings
    parser.add_argument("--dataset", "-ds", type=str, default="omniglot",
                        help='Default dataset is omniglot')
    parser.add_argument("--dataset_root", "-dr", type=str, default="../data",
                        help='Default dataset root is ../data')

    # Network settings
    parser.add_argument("--embedding", "-e", type=str, default='conv4',
                        help='Benchmark embedding network with 4 layer convolutions')
    parser.add_argument("--net_type", "-n", type=str, default='prototypical',
                        help='prototypical and matching are available')
    parser.add_argument("--metric", "-d", type=str, default='euclid',
                        help='euclid and cosine are available')

    # Solver settings
    parser.add_argument("--max_iteration", "-mi", type=int, default=20000,
                        help='Maximum number of mini-batch iterations')
    parser.add_argument("--learning_rate", "-lr", type=float, default=1.0e-3,
                        help='Learning rate of meta-learning step')
    parser.add_argument("--lr_decay_interval", "-lrdi", type=int, default=2000,
                        help='Learning rate decay interval of mini-batch iterations')
    parser.add_argument("--lr_decay", "-lrd", type=float, default=0.5,
                        help='Decay rate of each learning rate decay interval')
    parser.add_argument("--iter_per_epoch", "-ei", type=int, default=100,
                        help='Number of iterations per an epoch')
    parser.add_argument("--iter_per_valid", "-vi", type=int, default=1000,
                        help='Number of iterations per a validation')
    parser.add_argument("--n_episode_for_valid", "-ev", type=int, default=1000,
                        help='Number of episodes for validation')
    parser.add_argument("--n_episode_for_test", "-et", type=int, default=1000,
                        help='Number of episodes for test')

    # Calculation settings
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        help="Extension modules. ex) 'cpu', 'cudnn'.")
    parser.add_argument("--device-id", "-gid", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cuda.cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')

    # Log settings
    parser.add_argument("--work-dir", "-w", type=str, default="tmp.result/",
                        help='Directory for monitor results')

    # Program mode train or test
    parser.add_argument("--train-or-test", "-tt", type=str, default='train',
                        help='Program mode: train or test')

    args = parser.parse_args()

    return args


def load_celeb_a(dataset_path):

    # We cached resized CelebA dataset as npy files
    # Shapes of Images : (64, 64, 3)
    # Number of IDs : 10177
    # Number of Images : 202586
    celeb_a_x = np.load(dataset_path + 'celeb_images.npy')
    celeb_a_y = np.load(dataset_path + 'celeb_labels.npy')
    celeb_a_x = celeb_a_x.transpose(0, 3, 1, 2)

    # Original setting for few-shot face identification
    # - Number of classes for training : 8000
    # - Number of classes for validation : 1000
    # - Number of classes for test : 1177
    n_train = 8000
    n_val = 1000
    train_x = celeb_a_x[celeb_a_y < n_train]
    train_y = celeb_a_y[celeb_a_y < n_train]
    valid_x = celeb_a_x[(n_train <= celeb_a_y) & (celeb_a_y < n_train + n_val)]
    valid_y = celeb_a_y[(n_train <= celeb_a_y) & (
        celeb_a_y < n_train + n_val)] - n_train
    test_x = celeb_a_x[n_train + n_val <= celeb_a_y]
    test_y = celeb_a_y[n_train + n_val <= celeb_a_y] - n_train - n_val
    train_y = train_y.reshape(train_y.shape[0], 1)
    valid_y = valid_y.reshape(valid_y.shape[0], 1)
    test_y = test_y.reshape(test_y.shape[0], 1)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def load_omniglot(dataset_root):

    # We cached omniglot dataset as npy files
    x_train, _ = np.load(dataset_root + "/train.npy", allow_pickle=True)
    x_valid, _ = np.load(dataset_root + "/val.npy", allow_pickle=True)
    x = np.r_[x_train, x_valid]

    # A common setting for benchmarking with Omniglot dataset
    # - Image shape: (1, 28, 28)
    # - Number of classes: 1623
    # - Number of images per class: 20
    shape_x = (1, 28, 28)
    x_resized = np.zeros([1623, 20, 28, 28])

    # Resize images following the benchmark setting
    from nnabla.utils.image_utils import imresize
    for xi, ri in zip(x, x_resized):
        for xij, rij in zip(xi, ri):
            rij[:] = imresize(xij, size=(shape_x[2], shape_x[1]),
                              interpolate="nearest") / 255.

    # Class augmentation following the benchmark setting
    rng = np.random.RandomState(706)
    data = augmentation(x_resized)
    data = rng.permutation(data)
    data = data.reshape((1,) + data.shape).transpose(1, 2, 0, 3, 4)

    # Divide dataset following the benchmark setting
    train_data = data[:4112]
    valid_data = data[4112:4800]
    test_data = data[4800:]

    def separate_data_to_xy(data):
        n_class = data.shape[0]
        n_shot = data.shape[1]
        x = data.reshape((n_class * n_shot,) + data.shape[2:])
        y = np.arange(n_class).reshape(n_class, 1).repeat(
            n_shot).reshape(n_class * n_shot, 1)
        return x, y

    train_x, train_y = separate_data_to_xy(train_data)
    valid_x, valid_y = separate_data_to_xy(valid_data)
    test_x, test_y = separate_data_to_xy(test_data)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


class EpisodeGenerator():
    def __init__(self, x, y, n_way, n_shot, n_query):
        """
        Initialization function for episode generator class
            Args:
                x (nd_array): nd_array of images (n_sample, image_shape)
                y (nd_array): nd_array of labels (n_sample, 1)
                n_way (int): number of support classes, generally called n_way in one-shot litterateur
                n_shot (int): number of shots per class
                n_query (int): number of queries per class.
        """
        hist, _ = np.histogram(y, bins=max(y[:, 0]) + 1)
        self.class_ids = np.where(n_shot + n_query <= hist)[0]
        self.x = x
        self.y = y
        self.shape_x = x.shape[1:]
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.s_x = np.zeros((n_way * n_shot, ) + x[0].shape)
        self.q_x = np.zeros((n_way * n_query,) + x[0].shape)
        self.q_y = np.zeros((n_way * n_query, 1))

    def next(self):
        """
        Function for providing an episode
            Returns:
                x_s (~nnabla.Variable): support images
                x_q (~nnabla.Variable): query images
                y_q (~nnabla.Variable): query class_id in support sets
        """
        ids = np.random.choice(len(self.class_ids),
                               self.n_way * (1 + self.n_query), False)
        k_ids = ids[:self.n_way]
        u_ids = ids[self.n_way:]
        s_ids = []
        q_ids = []
        i = 0
        for i in range(self.n_way):
            support_id = i
            support_class_id = self.class_ids[k_ids[i]]
            a_ids = np.array(np.where(support_class_id == self.y[:, 0]))[0]
            a_ids = np.random.choice(a_ids, self.n_shot + self.n_query, False)
            s_ids = a_ids[:self.n_shot]
            q_ids = a_ids[self.n_shot:]
            for j in range(self.n_shot):
                self.s_x[i * self.n_shot + j] = self.x[s_ids[j], :]
            for j in range(self.n_query):
                self.q_x[i * self.n_query + j] = self.x[q_ids[j], :]
                self.q_y[i * self.n_query + j] = support_id
        return self.s_x, self.q_x, self.q_y


def meta_train(args, shape_x, train_data, valid_data, test_data):

    # Build episode generators
    train_episode_generator = EpisodeGenerator(
        train_data[0], train_data[1], args.n_class_tr, args.n_shot_tr, args.n_query_tr)
    valid_episode_generator = EpisodeGenerator(
        valid_data[0], valid_data[1], args.n_class, args.n_shot, args.n_query)
    test_episode_generator = EpisodeGenerator(
        test_data[0], test_data[1], args.n_class, args.n_shot, args.n_query)

    # Build training model
    xs_t = nn.Variable((args.n_class_tr * args.n_shot_tr, ) + shape_x)
    xq_t = nn.Variable((args.n_class_tr * args.n_query_tr, ) + shape_x)
    hq_t = net(args.n_class_tr, xs_t, xq_t, args.embedding,
               args.net_type, args.metric, False)
    yq_t = nn.Variable((args.n_class_tr * args.n_query_tr, 1))
    loss_t = F.mean(F.softmax_cross_entropy(hq_t, yq_t))

    # Build evaluation model
    xs_v = nn.Variable((args.n_class * args.n_shot, ) + shape_x)
    xq_v = nn.Variable((args.n_class * args.n_query, ) + shape_x)
    hq_v = net(args.n_class, xs_v, xq_v, args.embedding,
               args.net_type, args.metric, True)
    yq_v = nn.Variable((args.n_class * args.n_query, 1))
    err_v = F.mean(F.top_n_error(hq_v, yq_v, n=1))

    # Setup solver
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Monitor outputs
    monitor = Monitor(args.work_dir)
    monitor_loss = MonitorSeries(
        "Training loss", monitor, interval=args.iter_per_epoch)
    monitor_valid_err = MonitorSeries(
        "Validation error", monitor, interval=args.iter_per_valid)
    monitor_test_err = MonitorSeries("Test error", monitor)
    monitor_test_conf = MonitorSeries("Test error confidence", monitor)

    # Output files
    param_file = args.work_dir + "/params.h5"
    tsne_file = args.work_dir + "/tsne.png"

    # Save NNP
    batch_size = 1
    contents = save_nnp({'x0': xs_v, 'x1': xq_v}, {
                          'y': hq_v}, batch_size)
    save.save(os.path.join(args.work_dir,
                           'MetricMetaLearning_epoch0.nnp'), contents, variable_batch_size=False)

    # Training loop
    train_losses = []
    best_err = 1.0
    for i in range(args.max_iteration):

        # Decay learning rate
        if (i + 1) % args.lr_decay_interval == 0:
            solver.set_learning_rate(solver.learning_rate() * args.lr_decay)

        # Create an episode
        xs_t.d, xq_t.d, yq_t.d = train_episode_generator.next()

        # Training by the episode
        solver.zero_grad()
        loss_t.forward(clear_no_need_grad=True)
        loss_t.backward(clear_buffer=True)
        solver.update()
        train_losses.append(loss_t.d.copy())

        # Evaluation
        if (i + 1) % args.iter_per_valid == 0:
            train_loss = np.mean(train_losses)
            train_losses = []
            valid_errs = []
            for k in range(args.n_episode_for_valid):
                xs_v.d, xq_v.d, yq_v.d = valid_episode_generator.next()
                err_v.forward(clear_no_need_grad=True, clear_buffer=True)
                valid_errs.append(np.float(err_v.d.copy()))
            valid_err = np.mean(valid_errs)

            monitor_loss.add(i + 1, loss_t.d.copy())
            monitor_valid_err.add(i + 1, valid_err * 100)
            if valid_err < best_err:
                best_err = valid_err
                nn.save_parameters(param_file)

    # Final evaluation
    nn.load_parameters(param_file)
    v_errs = []
    for k in range(args.n_episode_for_test):
        xs_v.d, xq_v.d, yq_v.d = test_episode_generator.next()
        err_v.forward(clear_no_need_grad=True, clear_buffer=True)
        v_errs.append(np.float(err_v.d.copy()))
    v_err_mean = np.mean(v_errs)
    v_err_std = np.std(v_errs)
    v_err_conf = 1.96 * v_err_std / np.sqrt(args.n_episode_for_test)
    monitor_test_err.add(0, v_err_mean * 100)
    monitor_test_conf.add(0, v_err_conf * 100)

    # Visualization
    n_class = 50
    n_sample = 20
    visualize_episode_generator = EpisodeGenerator(
        train_data[0], train_data[1], n_class, 0, n_sample)
    _, samples, labels = visualize_episode_generator.next()
    u = get_embeddings(samples, conv4)
    v = get_tsne(u)
    plot_tsne(v[:, 0], v[:, 1], labels[:, 0], tsne_file)

    # Save NNP
    contents = save_nnp({'x0': xs_v, 'x1': xq_v}, {
                          'y': hq_v}, batch_size)
    save.save(os.path.join(args.work_dir,
                           'MetricMetaLearning.nnp'), contents, variable_batch_size=False)


def meta_test(args, shape_x, test_data):

    # Build episode generators
    test_episode_generator = EpisodeGenerator(
        test_data[0], test_data[1], args.n_class, args.n_shot, args.n_query)

    # Build prototypical network
    xs_v = nn.Variable((args.n_class * args.n_shot, ) + shape_x)
    xq_v = nn.Variable((args.n_class * args.n_query, ) + shape_x)
    hq_v = net(args.n_class, xs_v, xq_v, args.embedding,
               args.net_type, args.metric, True)
    yq_v = nn.Variable((args.n_class * args.n_query, 1))
    err_v = F.mean(F.top_n_error(hq_v, yq_v, n=1))

    # Load parameters
    nn.load_parameters(args.work_dir + "/params.h5")

    # Evaluate error rate
    v_errs = []
    for k in range(args.n_episode_for_test):
        xs_v.d, xq_v.d, yq_v.d = test_episode_generator.next()
        err_v.forward(clear_no_need_grad=True, clear_buffer=True)
        v_errs.append(np.float(err_v.d.copy()))
    v_err_mean = np.mean(v_errs)
    v_err_std = np.std(v_errs)
    v_err_conf = 1.96 * v_err_std / np.sqrt(args.n_episode_for_test)

    # Monitor error rate
    monitor = Monitor(args.work_dir)
    monitor_test_err = MonitorSeries("Test error", monitor)
    monitor_test_conf = MonitorSeries("Test error confidence", monitor)
    monitor_test_err.add(0, v_err_mean * 100)
    monitor_test_conf.add(0, v_err_conf * 100)

    return v_err_mean, v_err_conf


def main():

    # Get settings
    args = get_args()

    # Set context
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Load data
    if args.dataset == "omniglot":
        dataset_path = args.dataset_root + "/omniglot/data"
        if not os.path.exists(dataset_path):
            print("\nSet dataset path with --dataset_root option\n")
            exit()
        shape_x = (1, 28, 28)
        train_data, valid_data, test_data = load_omniglot(
            dataset_path)

    elif args.dataset == "celeb_a":
        dataset_path = args.dataset_root + "/celeb_a/data/"
        if not os.path.exists(dataset_path):
            print("\nSet dataset path with --dataset_root option\n")
            exit()
        shape_x = (3, 64, 64)
        train_data, valid_data, test_data = load_celeb_a(
            '../data/celeb_a/data/')

    else:
        print("\nUse omniglot or celeb_a dataset\n")
        exit()

    if args.train_or_test == "train":
        meta_train(args, shape_x, train_data, valid_data, test_data)
    elif args.train_or_test == "test":
        meta_test(args, shape_x, test_data)


if __name__ == '__main__':
    main()
