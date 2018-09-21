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


import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF


def augmentation(h, test, aug):
    '''
    Data augmentation by randomly shifting up to 2 pixels.

    Args:
        h (nnabla.Variable): Shape [B, C, H, W]
        test (bool): Only valid if aug is None, and the augmentation is applied if test=False.
        aug (None or bool): Whether the augmentation is applied or not.

    Returns:
        nnabla.Variable: Shape [B, C, H, W]

    '''
    if aug is None:
        aug = not test
    if aug:
        h = F.random_shift(h, (0, 0, 2, 2), seed=0)
    return h


def squash(h, axis=2, eps=1e-5):
    '''
    Squashing the capsules according to eq 1.

    Args:
        h (nnabla.Variable): Usually a shape of [B, #caps, C].
        axis (int): Channel axis.

    Returns:
        nnabla.Variable: The same dimensions as the given input.

    '''
    norm = F.sum(h ** 2, axis=axis, keepdims=True)
    return (norm / (1 + norm) / (norm ** 0.5 + eps)) * h


@PF.parametric_function_api('primary_capsule')
def primary_capsule(h, factor_capsules=32, out_channels=8, kernel=9, stride=2, fix_parameters=False):
    '''
    Takes Conv1 output and produces PrimaryCapsules.

    PrimaryCapsules are computed by using a single Convolution layer.

    Args:
        h (nnabla.Variable): A shape of [B, C, H, W].
        factor_capsules (int): Multiplication factor of output capsules. The output capsules will be ``factor_capsules x out_H x out_H`` where ``out_H`` and ``out_W`` are height and width of the output of the ``(kernel, kernel)`` Convolution with ``stride``. E.g. ``out_H = (H - (kernel - 1)) / stride``.
        out_channels (int): Number of units in each capsule of the output.
        kernel (int): Kernel size of the Convolution.
        stride (int): Stride of the Convolution.
        fix_parameters (bool): Fix parameters (Set need_grad=False).

    Returns:
        nn.Variable: A shape [B, factor_capsules x H' x W', out_channels].

    '''
    h = PF.convolution(h, out_channels * factor_capsules, (kernel, kernel),
                       stride=(stride, stride), fix_parameters=fix_parameters)
    num_capsules = factor_capsules * h.shape[2] * h.shape[3]
    h = F.reshape(h, [h.shape[0], out_channels, num_capsules])
    h = F.transpose(h, (0, 2, 1))
    return squash(h)


@PF.parametric_function_api('capsule')
def capsule_layer(u, num_j=10, out_channels=16, num_routing_iter=3, grad_dynamic_routing=False, fix_parameters=False):
    '''
    Takes PrimaryCapules output and produces DigitsCapsules.

    Args:
        u (nnabla.Variable): A shape of [B, in_capsules, in_channels]
        num_j (int): Number of output capsules.
        out_channels (int): Number of units in each capsule of the output.
        num_routing_iter (int): Dynamic routing iterations.
        grad_dynamic_routing (bool): If False, it doesn't compute gradients of
            dynamic routing coefficients as if they are given as
            hyperparameters.
        fix_parameters (bool): Fix parameters (Set need_grad=False).

    Returns:
        nn.Variable: A shape [B, num_j, out_channels].

    '''
    assert num_routing_iter > 0
    batch_size = u.shape[0]
    num_i = u.shape[1]  # 32 * 6 * 6
    in_channels = u.shape[2]

    # Routing u_hat = W u in eq 2.
    # Implementing with broadcast and batch_matmul. Maybe not efficient.

    # Create a parameter tensor
    # Note: Consider num input channels multiplied by num input capsules
    from nnabla.initializer import UniformInitializer, calc_uniform_lim_glorot
    from nnabla.parameter import get_parameter_or_create
    w_init = UniformInitializer(
        calc_uniform_lim_glorot(num_i * in_channels, out_channels))
    w_ij = get_parameter_or_create(
        "W", (1, num_j, num_i, in_channels, out_channels), w_init, not fix_parameters)
    # Tileing w_ij to [batch_size, num_j, num_i, in_channels, out_channels].
    w_ij_tiled = F.broadcast(w_ij, (batch_size,) + w_ij.shape[1:])
    # Tileing u to [batch_size, num_j, num_i, 1, in_channels].
    u = u.reshape((batch_size, 1, num_i, 1, in_channels))
    u_tiled = F.broadcast(u, (batch_size, num_j, num_i, 1, in_channels))
    # Apply batched matrix multiplication:
    # [1, in_channels] * [in_channels, out_channels] --> [1, out_channels]
    # u_hat shape: [batch_size, num_j, num_i, out_channels]
    u_hat = F.batch_matmul(u_tiled, w_ij_tiled).reshape(
        (batch_size, num_j, num_i, out_channels))

    # Dynamic Routing iteration doesn't compute gradients.
    # u_hat only used at the final step of computation of s.
    u_hat_no_grad = u_hat
    if not grad_dynamic_routing:
        u_hat_no_grad = F.identity(u_hat)
        u_hat_no_grad.need_grad = False

    # Dynamic routing described in Procedure 1.
    b = F.constant(0, (batch_size, num_j, num_i, 1))
    for r in range(num_routing_iter):
        # u_hat is only used in the last step.
        uh = u_hat_no_grad
        if r == num_routing_iter - 1:
            uh = u_hat

        # 4: Softmax in eq 3
        c = F.softmax(b, axis=1)
        # 5: Left of eq 2. s shape: [B, num_j, out_channels]
        s = F.sum(c * uh, axis=2)
        # 6: eq 1
        v = squash(s, axis=2)
        if r == num_routing_iter - 1:
            return u_hat, v
        # 7: Update by agreement
        b = b + F.sum(v.reshape((batch_size, num_j, 1, out_channels)) *
                      uh, axis=3, keepdims=True)


def capsule_net(x, test=False, aug=None, grad_dynamic_routing=False):
    '''
    Args:
        x (nnabla.Variable): A shape of [B, 1, 28, 28]. The values are in [0.0, 1.0].

    '''
    # Randomly shift up to 2 pixels.
    x = augmentation(x, test, aug)
    # Conv1 shape: [B, 256, 20, 20]
    c1 = F.relu(PF.convolution(x, 256, (9, 9), name='c1'))
    # PrimaryCapsules shape: [B, 8, 32 * 6 * 6]
    primary_caps = primary_capsule(c1, 32, 8, 9, 2)
    # DigitsCapsules shape: [B, 10, 16]
    u_hat, digits_caps = capsule_layer(
        primary_caps, 10, 16, 3, grad_dynamic_routing=grad_dynamic_routing)
    # Prediction (v-norm) shape: [B, 10]
    prediction = F.sum((digits_caps ** 2), axis=2) ** 0.5
    return c1, primary_caps, u_hat, digits_caps, prediction


def capsule_reconstruction(caps, t_onehot, noise=None):
    '''
    Reconstruct input image from capsules.

    The ``t_onehot`` is used to mask the capsule corresponding target label.

    Args:
        caps (nnabla.Variable): (A shape of [B, capsules, channels].
        t_onehot (nnabla.Variable): A shape of [B, capsules].
        noise (nnabla.Variable): A shape of [B, 1, channels]. This noise is injected to capsules.

    '''
    # Mask the Capsules by target label.
    # caps_vec shape: [B, capsules, channels]
    t_onehot = t_onehot.reshape(t_onehot.shape + (1,))
    if noise is not None:
        caps = caps + noise
    caps_vec = t_onehot * caps
    # MLP to reconstruct the input (values in [0.0, 1.0]).
    h = F.relu(PF.affine(caps_vec, 512, name='dec1'))
    h = F.relu(PF.affine(h, 1024, name='dec2'))
    h = F.sigmoid(PF.affine(h, (1, 28, 28), name='dec3'))
    return h


def capsule_loss(v_norm, t_onehot, recon=None, x=None, m_pos=0.9, m_neg=0.1, wn=0.5, wr=0.0005):
    '''
    Compute a margin loss given a length vector of  output capsules and one-hot labels, and.optionally computes a reconstruction loss.

    Margin loss is given in eq 4. Reconstruction loss is given in Sec 4.1.

    Args:
        v_norm (nnabla.Variable): A length vector of capsules. A shape of [B, capsules].
        t_onehot (nnabla.Variable): A shape of [B, capsules].
        recon (nnabla.Variable): Reconstruction output with a shape of [B, 1, 28, 28]. The values are in [0, 0.1].
        x (nnabla.Variable): Reconstruction target (i.e. input) with a shape of [B, 1, 28, 28]. The values are in [0, 0.1].
        m_pos (float): Margin of capsules corresponding targets.
        m_neg (float): Margin of capsules corresponding non-targets.
        wn (float): Weight of the non-target margin loss.
        wr (float): Weight of the reconstruction loss.

    Returns:
        nnabla.Variable: 0-dim

    '''
    # Classification loss
    lp = F.sum(t_onehot * F.relu(m_pos - v_norm) ** 2)
    ln = F.sum((1 - t_onehot) * F.relu(v_norm - m_neg) ** 2)
    lmargin = lp + wn * ln
    if recon is None or x is None:
        return lmargin / v_norm.shape[0]
    # Reconstruction loss
    lr = F.sum(F.squared_error(recon, x))
    # return (lmargin + wr * lr) / v_norm.shape[0]
    lmargin = lmargin / v_norm.shape[0]
    lmargin.persistent = True
    lreconst = (wr * lr) / v_norm.shape[0]
    lreconst.persistent = True
    return lmargin, lreconst, lmargin + lreconst
