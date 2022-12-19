import jax
import jax.numpy as jnp
from jax import nn
from jax.nn.initializers import glorot_normal, normal
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

@jax.jit
def linear(x, W, b):
    return jnp.dot(x, W) + b


@jax.jit
def Linear(params, inputs):
    outputs = linear(inputs, params["W"], params["b"])
    return outputs


@jax.jit
def MLP(params, inputs):
    linear1_out = Linear(params["linear1"], inputs)
    linear2_in = nn.relu(linear1_out)
    linear2_out = Linear(params["linear2"], linear2_in)
    logits = nn.softmax(linear2_out)
    return logits


@jax.jit
def categorical_cross_entropy_loss(logits, onehot_labels):
    return jnp.mean(-jnp.sum(onehot_labels * jnp.log(logits), axis=1))


@jax.jit
def SGD(params, grads, lr = 0.1):
    return jax.tree_map(lambda param, grad: param - lr * grad, params, grads)


@jax.jit
def train_batch(params, batch_X, batch_y):
    def loss_fn(params_, batch_X_):
        logits = MLP(params_, batch_X_)
        return categorical_cross_entropy_loss(logits, batch_y)
    grads = jax.jit(jax.grad(loss_fn))(params, batch_X)
    return SGD(params, grads)

@jax.jit
def train_one_epoch(rng, params, X_train, y_train, batch_size = 50):
    index = jax.random.permutation(rng, X_train.shape[0])
    num_batches = X_train.shape[0] // batch_size + 1
    for batch in range(num_batches):
        batch_index = index[batch * batch_size: (batch + 1) * batch_size]
        params = train_batch(params, X_train[batch_index], y_train[batch_index])
    return params


@jax.jit
def train(params, X_train, y_train, epochs: int):
    rng = jax.random.PRNGKey(42)
    params = jax.lax.fori_loop(0, epochs, lambda epoch_, params_: train_one_epoch(rng, params_, X_train, y_train), params)
    return params


@jax.jit
def accuracy(params, X_test, y_test):
    pred = MLP(params, X_test)
    return jnp.mean(y_test == jnp.argmax(pred, axis=1))


if __name__ == '__main__':
    rng = jax.random.PRNGKey(0)
    rng1, rng2 = jax.random.split(rng)
    rng1w, rng1b = jax.random.split(rng1)
    rng2w, rng2b = jax.random.split(rng2)
    params = {
        "linear1": {
            "W": glorot_normal()(rng1w, (4, 100)),
            "b": normal()(rng1b, (100,))
        },
        "linear2": {
            "W": glorot_normal()(rng2w, (100, 3)),
            "b": normal()(rng2b, (3,))
        }
    }
    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(jnp.array(iris_dataset['data']), jnp.array(iris_dataset['target']), test_size=0.25,  random_state=0)
    y_train = jnp.eye(3)[y_train]
    params = train(params, X_train, y_train, 100)
    acc = accuracy(params, X_test, y_test)
    print(acc)


