import sys, os

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.utils.save import save

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, type=str)
    parser.add_argument("--nnp_path", default="./test.nnp", type=str)

    return parser.parse_args()


def get_nnable_graph(network):
    pass


def network(x):
    with nn.parameter_scope("h1"):
        h = PF.convolution(x, 32, (2, 2))
        h = F.relu(h)

    with nn.parameter_scope("h2"):
        h = PF.convolution(h, 16, (2, 2))
        h = F.tanh(h)

    with nn.parameter_scope("fc1"):
        h = PF.affine(h, 10)

    return h


def network2(x):
    with nn.parameter_scope("h1_2"):
        h = PF.convolution(x, 64, (3, 3))
        h = F.tanh(h)

    with nn.parameter_scope("h2_2"):
        h = PF.convolution(h, 32, (4, 4))
        h = F.leaky_relu(h)

    with nn.parameter_scope("fc1_2"):
        h = PF.affine(h, 4)

    return h


def exec_save(args):
    print("====================do save====================")

    x = nn.Variable((4, 2, 32, 32))
    t = nn.Variable((4, 1))

    y = network(x)

    loss = F.softmax_cross_entropy(y, t)

    contents = {
        'networks': [
            {'name': 'network',
             'batch_size': x.shape[0],
             'outputs': {'y': y, },
             'names': {'x': x, }}],
        'executors': [
            {'name': 'runtime',
             'network': 'network',
             'data': ['x', ],
             'output': ['y', ]}]}

    save(args.nnp_path, contents, include_params=True)


def exec_load(args, test=True):
    print("====================do load====================")
    if test:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from .load import load_nnp_for_nnabla as load
    else:
        from nnabla.utils.load import load

    info = load(args.nnp_path)


if __name__ == '__main__':
    args = get_args()

    if args.mode == "save":
        exec_save(args)
    elif args.mode == "load":
        exec_load(args)
    else:
        raise NotImplementedError("mode must be save or load")
