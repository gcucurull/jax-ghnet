import math

import jax.numpy as np
from jax import lax, random
from jax.experimental import stax
from jax.experimental.stax import Relu, LogSoftmax
from jax.nn.initializers import glorot_normal, normal, uniform
import jax.nn as nn


def Dropout(rate):
    """
    Layer construction function for a dropout layer with given rate.
    This Dropout layer is modified from stax.experimental.Dropout, to use
    `is_training` as an argument to apply_fun, instead of defining it at
    definition time.

    Arguments:
        rate (float): Probability of keeping and element.
    """
    def init_fun(rng, input_shape):
        return input_shape, ()
    def apply_fun(params, inputs, is_training, **kwargs):
        rng = kwargs.get('rng', None)
        if rng is None:
            msg = ("Dropout layer requires apply_fun to be called with a PRNG key "
                   "argument. That is, instead of `apply_fun(params, inputs)`, call "
                   "it like `apply_fun(params, inputs, rng)` where `rng` is a "
                   "jax.random.PRNGKey value.")
            raise ValueError(msg)
        keep = random.bernoulli(rng, rate, inputs.shape)
        outs = np.where(keep, inputs / rate, 0)
        # if not training, just return inputs and discard any computation done
        out = lax.cond(is_training, outs, lambda x: x, inputs, lambda x: x)
        return out
    return init_fun, apply_fun


def GraphHighwayConvolution(out_dim: int, infusion: str = 'inner'):
    """
    Layer constructor function for a Graph Highway Convolution layer as the 
    one proposed in https://arxiv.org/abs/2004.04635 
    """
    assert infusion in ('inner', 'outer', 'raw')
    def init_fun(rng, input_shape):
        # TODO: maybe try glorot
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2, k3, k4, k5 = random.split(rng, num=5)
        stdv = 1. / math.sqrt(out_dim)
        W_init, b_init = uniform(stdv), uniform(stdv)
        # used for the gating function
        W_t = W_init(k1, (input_shape[-1], out_dim))
        b_t = b_init(k2, (out_dim,))
        # used for the homogenous representation
        Theta = W_init(k3, (input_shape[-1], out_dim))
        # projection used in the outer infusion
        W_h = W_init(k4, (input_shape[-1], out_dim))
        W_x = W_init(k5, (input_shape[-1], out_dim))
        return output_shape, (W_t, b_t, Theta, W_h, W_x)

    def apply_fun(params, input, adj, **kwargs):
        # TODO: we need the adj matrix without self loops
        # raised to power K
        # TODO: try bias in the projections

        # x, first_x = input
        x = input

        W_t, b_t, Theta, W_h, W_x = params
        gate = nn.sigmoid(np.dot(x, W_t) + b_t)

        F_hom = np.dot(x, Theta)
        F_hom = np.matmul(adj, F_hom)

        if infusion == 'inner':
            F_het = np.dot(x, Theta)
        elif infusion == 'outer':
            if x.shape[-1] != W_h.shape[-1]:
                F_het = np.dot(x, W_h)
            else:
                F_het = x
        elif infusion == 'raw':
            F_het = np.dot(first_x, W_x)

        out = gate*F_hom + (1 - gate)*F_het

        return out

    return init_fun, apply_fun

def GHNet(nhid, nclass, dropout):
    """
    GHNet implementation.
    """
    gc1_init, gc1_fun = GraphHighwayConvolution(nhid)
    _, drop_fun = Dropout(dropout)
    gc2_init, gc2_fun = GraphHighwayConvolution(nclass)

    init_funs = [gc1_init, gc2_init]

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_fun(params, x, adj, is_training=False, **kwargs):
        rng = kwargs.pop('rng', None)
        adj_1, adj_5 = adj

        x = gc1_fun(params[0], x, adj_1, rng=rng) # first conv has 1 hop
        x = nn.relu(x)
        x = drop_fun(None, x, is_training=is_training, rng=rng)
        x = gc2_fun(params[1], x, adj_5, rng=rng)
        x = nn.log_softmax(x)
        return x
    
    return init_fun, apply_fun