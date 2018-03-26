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

#!/usr/bin/env python

import numpy as np
import sys
import os
import time

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solver as S
import nnabla.monitor as M

from nnabla.utils.data_source_loader import download


def load_ptbset(ptbfile):
    """
    Load Penn Treebank Corpus 

    """

    f = download(ptbfile)
    itow = {}  # index to word
    wtoi = {}  # word to index
    dataset = []

    # extract vocabraries from corpus
    for line in f:
        for w in line.split():
            # register the new word as an index number of first appearance
            if w not in wtoi:
                i = len(wtoi)
                wtoi[w] = i
                itow[i] = w
            # translate words into numbers
            dataset.append(wtoi[w])

    return itow, wtoi, dataset


class CategoricalSampler(object):
    """
    Categoricl Sampler
    - the sampler for getting negative samples

    """

    def calc_random_method_selection_rate(self, k, histogram, gamma):
        """
        Calculate 2 rondom type selection rate
           In this example, the sampler combines 2 random method
             - sample from dataset
             - sample from uniform random of n_category
        This operation intends to simulate the distribution of
        powered histogram.

        This function calculate the rate of 2 random method
        minimising the difference between real distribution and combined distribution

        Args:
            k: number of categgory
            histogram : histogram
            gamma : power of histogram in negative sampling

        Returns:
            rate of randomizing method
        """

        p = histogram
        p /= np.sum(p)

        q = np.power(histogram, gamma)
        q /= np.sum(q)

        c = 1.0 / k

        alpha = np.sum((p - q) * (p - q)) / np.sum((p - c) * (p - c))

        rate = (1 - alpha) / ((1 - alpha) + c)
        return rate

    def __init__(self, dataset, gamma=0.75):
        """
        Initialization

        Args:
            dataset: word corpus replaced id 

        """

        n_category = np.max(dataset)
        self.n_category = n_category
        self.dataset = np.array(dataset)

        # create histogram
        histogram = [0] * (n_category + 1)
        for wid in dataset:
            histogram[wid] += 1
        self.rate = self.calc_random_method_selection_rate(
            n_category, histogram, gamma)

    def sample(self, shape):
        """
        Extract samples

        Args:
            shape: sample shape

        Returns:
            values

        """
        if np.random.rand() < self.rate:
            indices = np.random.randint(
                low=0, high=len(self.dataset), size=shape)
            return self.dataset[indices]
        else:
            return np.random.randint(low=0, high=self.n_category, size=shape)


def create_minibatch(dataset, ids, sampler, half_window=3, n_negative=5):
    """
    Create minibatch

    Args:
        ids: indices
        sampler: sampler functions
        half_window: half window size
        n_negative: numper of negative samples

    Returns:
        list of x(word) y(context) t(positive, negative)

    """

    # loop
    xp, yp = [], []
    for i in ids:

        # positive-context
        wc = np.array(dataset[i - half_window + 1:i] +
                      dataset[i + 1:i + half_window])
        xp.append(wc)

        # positive-word
        wt = np.array(dataset[i]).repeat((half_window - 1) * 2)
        yp.append(wt)

    # positive-context
    xp = np.hstack(xp)

    # positive-word
    yp = np.hstack(yp)
    # yp = np.array(dataset)[ids].repeat((half_window - 1) * 2) # is possible but slower than above line

    # postive-label
    tp = np.ones(len(xp), dtype=np.int32)

    # negative-context
    xn = xp.repeat(n_negative)

    # negative-word
    yn = sampler.sample(len(xn))

    # negative-label
    tn = np.zeros(len(xn), dtype=np.int32)

    x = np.hstack([xp, xn])
    y = np.hstack([yp, yn])
    t = np.hstack([tp, tn])

    return [x, y, t]


class DataIteratorForEmbeddingLearning():
    def __init__(self, batchsize, half_window, n_negative, dataset):
        """
        Initialization

        Args:
            batchsize: batchsize
            half_window: half window length
            n_negative: number of negative samples
            dataset: corpus replaced with word ids

        """

        self.batchsize = batchsize
        self.half_window = half_window
        self.n_negative = n_negative
        self.dataset = dataset
        self.counter = 0
        self.datasize = len(dataset)
        self.sampler = CategoricalSampler(dataset)
        self.indices = np.random.permutation(
            self.datasize - 2 * self.half_window) + self.half_window
        self.n_batch = self.datasize // batchsize - 1

    def next(self, ):
        """
        Creating minibatch

        Returns:
            list of x(word_id) y(context) t(positive, negative)

        """

        # To ensure indices do not exceed datasize, if position reaches to end
        if self.datasize <= self.counter + self.batchsize:
            self.counter = 0
            self.indices = np.random.permutation(
                self.datasize - 2 * self.half_window) + self.half_window

        # Context indices
        # terminations are already cared
        ids = self.indices[self.counter:(self.counter + self.batchsize)]

        # Position increment
        self.counter += self.batchsize

        # Create minibatch
        return create_minibatch(self.dataset, ids, self.sampler, self.half_window, self.n_negative)


def output_similar_words(itow, k, similarity):
    """
    Function for outputting similar words

    Args:
        itow : dictionary of index to word
        k : query word id
        similarity: calculated similarity

    """
    n_result = 5  # number of search result to show
    print('query_word: id=%d, %s' % (k, itow[k]))
    count = 0

    # Enumerate similar words
    for i in (-similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if i == k:
            continue
        print('id=%d, %s: %f' % (i, itow[i], similarity[i]))
        count += 1
        if count == n_result:
            break


def get_args():
    """
    Get command line arguments.
    Arguments set the default values of command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension path. ex) cpu, cuda.cudnn.")
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cuda.cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--work_dir", "-m", type=str,
                        default="tmp.result.w2v/")

    parser.add_argument("--embed-dim", "-e", type=int, default=100)
    parser.add_argument("--batchsize", "-b", type=int, default=100)
    parser.add_argument("--half-window-length", "-wl", type=int, default=3)
    parser.add_argument("--n-negative-sample", "-ns", type=int, default=5)

    parser.add_argument("--learning-rate", "-l", type=float, default=1e-3)
    parser.add_argument("--max-epoch", "-i", type=int, default=20)
    parser.add_argument("--monitor-interval", "-v", type=int, default=1000)

    parser.add_argument("--max-check-words", "-mw", type=int, default=405)
    return parser.parse_args()


def main():

    # Get arguments
    args = get_args()
    data_file = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt"
    model_file = args.work_dir + "model.h5"

    # Load Dataset
    itow, wtoi, dataset = load_ptbset(data_file)

    # Get context.
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Create data provider
    n_word = len(wtoi)
    n_dim = args.embed_dim
    batchsize = args.batchsize
    half_window = args.half_window_length
    n_negative = args.n_negative_sample

    di = DataIteratorForEmbeddingLearning(
        batchsize=batchsize,
        half_window=half_window,
        n_negative=n_negative,
        dataset=dataset)

    # Create model
    # - Real batch size including context samples and negative samples
    size = batchsize * (1 + n_negative) * (2 * (half_window - 1))

    # Model for learning
    # - input variables
    xl = nn.Variable((size,))  # variable for word
    yl = nn.Variable((size,))  # variable for context

    # Embed layers for word embedding function
    # - f_embed : word index x to get y, the n_dim vector
    # --  for each sample in a minibatch
    hx = PF.embed(xl, n_word, n_dim, name="e1")  # feature vector for word
    hy = PF.embed(yl, n_word, n_dim, name="e1")  # feature vector for context
    hl = F.sum(hx * hy, axis=1)

    # -- Approximated likelihood of context prediction
    # pos: word context, neg negative samples
    tl = nn.Variable([size, ], need_grad=False)
    loss = F.sigmoid_cross_entropy(hl, tl)
    loss = F.mean(loss)

    # Model for test of searching similar words
    xr = nn.Variable((1,), need_grad=False)
    hr = PF.embed(xr, n_word, n_dim, name="e1")  # feature vector for test

    # Create solver
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Create monitor.
    monitor = M.Monitor(args.work_dir)
    monitor_loss = M.MonitorSeries(
        "Training loss", monitor, interval=args.monitor_interval)
    monitor_time = M.MonitorTimeElapsed(
        "Training time", monitor, interval=args.monitor_interval)

    # Do training
    max_epoch = args.max_epoch
    for epoch in range(max_epoch):

        # iteration per epoch
        for i in range(di.n_batch):

            # get minibatch
            xi, yi, ti = di.next()

            # learn
            solver.zero_grad()
            xl.d, yl.d, tl.d = xi, yi, ti
            loss.forward(clear_no_need_grad=True)
            loss.backward(clear_buffer=True)
            solver.update()

            # monitor
            itr = epoch * di.n_batch + i
            monitor_loss.add(itr, loss.d)
            monitor_time.add(itr)

    # Save model
    nn.save_parameters(model_file)

    # Evaluate by similarity
    max_check_words = args.max_check_words
    for i in range(max_check_words):

        # prediction
        xr.d = i
        hr.forward(clear_buffer=True)
        h = hr.d

        # similarity calculation
        w = nn.get_parameters()['e1/embed/W'].d
        s = np.sqrt((w * w).sum(1))
        w /= s.reshape((s.shape[0], 1))
        similarity = w.dot(h[0]) / s[i]

        # for understanding
        output_similar_words(itow, i, similarity)


if __name__ == '__main__':
    main()
