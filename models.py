import jax
import jax.numpy as np
from jax import lax, random
from jax.experimental import stax
from jax.experimental.stax import Relu, LogSoftmax
from jax.nn.initializers import glorot_normal, normal, glorot_uniform, zeros, ones
import jax.nn as nn

@jax.partial(jax.jit, static_argnums=(2))
def sp_matmul(A, B, shape):
    """
    Arguments:
        A: (N, M) sparse matrix represented as a tuple (indexes, values)
        B: (M,K) dense matrix
        shape: value of N
    Returns:
        (N, K) dense matrix
    """
    assert B.ndim == 2
    indexes, values = A
    rows, cols = indexes
    in_ = B.take(cols, axis=0)
    prod = in_*values[:, None]
    res = jax.ops.segment_sum(prod, rows, shape)
    return res

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


def GraphHighwayConvolution(out_dim: int, infusion: str = 'inner', 
                            dropout: float = 0.5, sparse: bool = False):
    """
    Layer constructor function for a Graph Highway Convolution layer as the 
    one proposed in https://arxiv.org/abs/2004.04635 
    """
    assert infusion in ('inner', 'outer', 'raw')
    _, drop_fun = Dropout(dropout)

    def matmul(A, B, shape):
        if sparse:
            return sp_matmul(A, B, shape)
        else:
            return np.matmul(A, B)

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2, k3, k4, k5 = random.split(rng, num=5)
        W_init, b_init = glorot_uniform(), zeros

        # used for the gating function
        W_t = W_init(k1, (input_shape[-1], out_dim))
        b_t = b_init(k2, (out_dim,))

        # used for the homogenous representation
        Theta = W_init(k3, (input_shape[-1], out_dim))

        # projection used in the outer infusion
        W_h = W_init(k4, (input_shape[-1], out_dim))
        # used only in the raw infusion
        W_x = W_init(k5, (1433, out_dim)) # hardcoded for Cora. should be an arg

        return output_shape, (W_t, b_t, Theta, W_h, W_x)

    def apply_fun(params, input, adj, activation=nn.relu, **kwargs):
        rng = kwargs.pop('rng', None)
        is_training = kwargs.pop('is_training', None)
        first_x, x = input # we need the first input for 'raw' infusion
        W_t, b_t, Theta, W_h, W_x = params

        x = drop_fun(None, x, is_training=is_training, rng=rng)

        # compute gate
        gate = nn.sigmoid(np.dot(x, W_t) + b_t)

        F_hom = np.dot(x, Theta)

        if infusion == 'inner':
            F_het = F_hom
        elif infusion == 'outer':
            F_het = np.dot(x, W_h) if x.shape[-1] != W_h.shape[-1] else x
        elif infusion == 'raw':
            F_het = np.dot(first_x, W_x)

        # k-hop convolution: adj is adj^k without self connections
        F_hom = matmul(adj, F_hom, F_hom.shape[0])
        F_hom = activation(F_hom)

        out = gate*F_hom + (1 - gate)*F_het

        return out

    return init_fun, apply_fun

def GHNet(nhid: int, nclass: int, dropout: float, infusion: str = 'inner', 
          sparse: bool = False):
    """
    GHNet implementation.
    """
    gc1_init, gc1_fun = GraphHighwayConvolution(nhid, infusion, dropout, sparse)
    gc2_init, gc2_fun = GraphHighwayConvolution(nclass, infusion, dropout, sparse)

    init_funs = [gc1_init, gc2_init]

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_fun(params, first_x, adj, is_training=False, **kwargs):
        rng = kwargs.pop('rng', None)
        k1, k2, k3, k4 = random.split(rng, 4)
        adj_1, adj_5 = adj

        x = gc1_fun(params[0], (first_x, first_x), adj_1, rng=k2, 
            is_training=is_training) 
        x = gc2_fun(params[1], (first_x, x), adj_5, activation=lambda x: x, 
            rng=k4, is_training=is_training)
        x = nn.log_softmax(x)
        return x
    
    return init_fun, apply_fun