import numpy as np
import random
import os,sys
from subprocess import call
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solver as S
from nnabla.contrib.context import extension_context
import nnabla.initializer as I

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension path. ex) cpu, cuda.cudnn.")
    parser.add_argument("--device-id", "-d", type=int, default=0)
    parser.add_argument("--work_dir", "-m", type=str,
                        default="tmp.result/")
    parser.add_argument("--save_dir", "-s", type=str,
                        default="params/")
    parser.add_argument("--embed-dim", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--half-window-length", "-wl", type=int, default=3)
    parser.add_argument("--n-negative-sample", "-ns", type=int, default=5)
    parser.add_argument("--learning-rate", "-l", type=float, default=1e-3)
    parser.add_argument("--max-iter", "-i", type=int, default=200000)
    parser.add_argument("--monitor-interval", "-v", type=int, default=1000)
    parser.add_argument("--max-check-words", "-mw", type=int, default=405)
    return parser.parse_args()

def perplexity(loss):
  perplexity = np.exp(loss)
  return perplexity

class LSTMCell(object):
  def __init__(self, state_size):
    self.state_size = state_size
  def __call__(self, x_t, h_t1, C_t1, dropout=False):
    X = F.concatenate(*(h_t1, x_t),axis=1)
    f_t = F.sigmoid(PF.affine(X,state_size , name='forget'))
    i_t = F.sigmoid(PF.affine(X,state_size , name='input'))
    o_t = F.sigmoid(PF.affine(X,state_size , name='output'))
    Ctilde_t = F.tanh(PF.affine(X,state_size, name='cell' ))
    C_t = f_t * C_t1 + i_t * Ctilde_t
    h_t = o_t * F.tanh(C_t)
    if dropout:
        h_t = F.dropout(h_t,0.5)
    return h_t, C_t

ctx = extension_context("cuda.cudnn", device_id=0)
nn.set_default_context(ctx)
nn.prefer_cached_array(True)

def get_data(fname):
    if not os.path.exists(fname):
      url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/'+fname
      call(['wget',url])

    words = open(fname).read().replace('\n', '<eos>').split()
    words_as_set = set(words)
    word_to_id = {w: i for i, w in enumerate(words_as_set)}
    id_to_word = {i: w for i, w in enumerate(words_as_set)}
    data = [word_to_id[w] for w in words]
    return data

train_data = get_data('ptb.train.txt')
val_data = get_data('ptb.valid.txt')
test_data = get_data('ptb.test.txt')

args = get_args()
state_size = 650
batch_size = args.batch_size
val_iter = 1000
num_iters = args.max_iter
num_steps = 20
num_layers = 4

lstm = []
for _ in range(num_layers):
    lstm.append(LSTMCell(state_size))
  

def get_one_example(data,num_steps):
    offset = np.random.randint(len(data) - num_steps - 1)
    return (data[offset:offset+num_steps], data[offset+num_steps])
   
def get_mini_batch(batch_size,data,num_steps):
    words, targets = [], []
    for _ in range(batch_size):
        w, t = get_one_example(data,num_steps)
        words.append(w)
        targets.append(t)
    return np.array(words).reshape([batch_size, num_steps]), np.array(targets).reshape([batch_size, 1])

from nnabla.monitor import Monitor, MonitorSeries
monitor = Monitor(args.work_dir)
monitor_loss = MonitorSeries("Training loss", monitor, interval=100)
monitor_perplexity = MonitorSeries("Training perplexity", monitor, interval=100)
monitor_vperplexity = MonitorSeries("Test perplexity", monitor, interval=100)



def model(batch_size, num_iters, lr, num_steps):

    n_words = 10000
    f_dim = state_size 

    x = nn.Variable([batch_size, num_steps])
    t = nn.Variable([batch_size, 1])
    h = PF.embed(x, n_words, f_dim, name='embed') 
    output = [nn.Variable([batch_size, state_size])] * num_layers
    state = [nn.Variable([batch_size, state_size])] * num_layers
     
    output_cur = nn.Variable([batch_size, state_size])
    state_cur = nn.Variable([batch_size, state_size])
    for i in range(num_steps):
        h_cur = F.slice(h, start=(0,i,0), stop=(batch_size,i+1,state_size))
        for j in range(num_layers):
            if j==0:
                with nn.parameter_scope("lstm" + str(j)) :
                    output[j], state[j] = lstm[j](h_cur, output_cur, state_cur)
            else:
                with nn.parameter_scope("lstm" + str(j)) :
                    output[j], state[j] = lstm[j](output[j-1], output[j], state[j])
        output_cur = output[-1]
        state_cur = state[-1]
    
    logits = PF.affine(output_cur, 10000, name='logits')
    loss = F.mean(F.softmax_cross_entropy(logits, t))
  
    solver = S.Adam(lr)
    solver.set_parameters(nn.get_parameters())

    for i in range(num_iters):
        x.d, t.d = get_mini_batch(batch_size,train_data, num_steps)
        output_cur.d = np.zeros([batch_size, state_size])
        state_cur.d = np.zeros([batch_size, state_size])
        perp = perplexity(loss.d)

        solver.zero_grad()
        loss.forward()
        loss.backward()
        monitor_perplexity.add(i, perp)
        solver.weight_decay(1e-5)
        solver.update()

        if i%1000==0:
            vper=0.0
            for j in range(val_iter):
                x.d, t.d = get_mini_batch(batch_size,val_data, num_steps)
                vper += perplexity(loss.d)
            monitor_vperplexity.add(i,vper/val_iter)
    return loss

model(batch_size, num_iters, args.learning_rate, num_steps)
nn.save_parameters(os.path.join(args.save_dir,'params.h5'))

