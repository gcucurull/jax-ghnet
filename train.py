import argparse
import time

import jax
import jax.numpy as np
from jax import jit, grad, random
from jax.experimental import optimizers

from utils import load_data
from models import GHNet


@jit
def loss(params, batch):
    """
    The idxes of the batch indicate which nodes are used to compute the loss.
    """
    inputs, targets, adj, is_training, rng, idx = batch
    preds = predict_fun(params, inputs, adj, is_training=is_training, rng=rng)
    ce_loss = -np.mean(np.sum(preds[idx] * targets[idx], axis=1))
    l2_loss = 5e-4 * optimizers.l2_norm(params)**2 # tf doesn't use sqrt
    return ce_loss + l2_loss

@jit
def accuracy(params, batch):
    inputs, targets, adj, is_training, rng, idx = batch
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(predict_fun(params, inputs, adj, is_training=is_training, rng=rng), axis=1)
    return np.mean(predicted_class[idx] == target_class[idx])

@jit
def loss_accuracy(params, batch):
    inputs, targets, adj, is_training, rng, idx = batch
    preds = predict_fun(params, inputs, adj, is_training=is_training, rng=rng)
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(preds, axis=1)
    ce_loss = -np.mean(np.sum(preds[idx] * targets[idx], axis=1))
    acc = np.mean(predicted_class[idx] == target_class[idx])
    return ce_loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--infusion', type=str, default='inner')
    args = parser.parse_args()

    # Load data
    adj_1, features, labels, idx_train, idx_val, idx_test = load_data()
    adj_5 = np.linalg.matrix_power(adj_1, 5)
    adj = (adj_1, adj_5) # 1 hop and 5 hops adj matrices

    rng_key = random.PRNGKey(args.seed)
    dropout = args.dropout
    step_size = args.lr
    hidden = args.hidden
    num_epochs = args.epochs
    n_nodes = adj_1.shape[0]
    n_feats = features.shape[1]
    infusion = args.infusion

    init_fun, predict_fun = GHNet(nhid=hidden, 
                                nclass=labels.shape[1],
                                dropout=dropout,
                                infusion=infusion)
    input_shape = (-1, n_nodes, n_feats)
    rng_key, init_key = random.split(rng_key)
    _, init_params = init_fun(init_key, input_shape)

    opt_init, opt_update, get_params = optimizers.adam(step_size)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    opt_state = opt_init(init_params)

    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        batch = (features, labels, adj, True, rng_key, idx_train)
        opt_state = update(epoch, opt_state, batch)
        epoch_time = time.time() - start_time

        params = get_params(opt_state)
        eval_batch = (features, labels, adj, False, rng_key, idx_val)
        train_batch = (features, labels, adj, False, rng_key, idx_train)
        train_loss, train_acc = loss_accuracy(params, train_batch)
        val_loss, val_acc = loss_accuracy(params, eval_batch)
        print(f"Iter {epoch}/{num_epochs} ({epoch_time:.4f} s) train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        # new random key at each iteration, othwerwise dropout uses always the same mask 
        rng_key, _ = random.split(rng_key)
    
    # now run on the test set
    test_batch = (features, labels, adj, False, rng_key, idx_test)
    test_acc = accuracy(params, test_batch)
    print(f'Test set acc: {test_acc}')