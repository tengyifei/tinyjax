import jax.numpy as jnp
import numpy as np
from jax import make_jaxpr

from tinyjax.eval import eval_jaxpr


def func(first, second):
  return first * second


def test_mul_1d():
  first = jnp.array([1, 2, 3])
  second = jnp.array(4)
  jax_result = func(first, second)
  jaxpr = make_jaxpr(func)(first, second)
  tg_result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, first, second)[0].numpy()
  assert np.allclose(np.array(jax_result), tg_result)


def test_mul_2d():
  first = jnp.array([[1, 2, 3], [4, 5, 6]])
  second = jnp.array(4)
  jax_result = func(first, second)
  jaxpr = make_jaxpr(func)(first, second)
  tg_result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, first, second)[0].numpy()
  assert np.allclose(np.array(jax_result), tg_result)
