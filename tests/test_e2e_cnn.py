import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from tinyjax.eval import eval_jaxpr


class CNN(nn.Module):
  """A simple MNIST CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.silu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.silu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.silu(x)
    x = nn.Dense(features=10)(x)
    return x


def apply_model(cnn: CNN, params, images):
  return cnn.apply(params, images)


def test_cnn():
  bs = 5
  rng = jax.random.key(42)
  rng1, rng2 = jax.random.split(rng)
  cnn = CNN()
  params = cnn.init(rng1, jnp.ones([1, 28, 28, 1]))
  images = jax.random.uniform(rng2, (bs, 28, 28, 1))

  def partial_apply(images):
    return apply_model(cnn, params, images)

  closed_jaxpr = jax.make_jaxpr(partial_apply)(images)

  tg_out = eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, images)[0].numpy()
  jax_out = np.array(partial_apply(images))
  assert np.allclose(tg_out, jax_out, rtol=1e-2, atol=1e-4)
