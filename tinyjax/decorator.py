import functools
from typing import Callable, List, Union, cast, overload

import jax
from jax.interpreters.partial_eval import dce_jaxpr
from tinygrad import Tensor, TinyJit
import jax.tree_util

from tinyjax.eval import eval_jaxpr


@overload
def tiny(fun: Callable[..., jax.Array]) -> Callable[..., Tensor]: ...


@overload
def tiny(fun: Callable[..., List[jax.Array]]) -> Callable[..., List[Tensor]]: ...


def tiny(
  fun: Callable,
) -> Union[
  Callable[..., Tensor],
  Callable[..., List[Tensor]],
]:
  """Wrap a JAX function into a Tinygrad function.

  The resulting function will wait until it is first invoked with some arguments,
  then use those arguments to trace the wrapped function to obtain a Jaxpr.
  Then it will transform the Jaxpr into a Tinygrad computation graph and
  evaluate it.

  Examples:

    >>> import jax.numpy as jnp
    >>> import tinygrad as tg
    >>> from tinyjax.decorator import tiny
    >>>
    >>> @tiny
    >>> def fun(first, second):
    >>>   temp = first + jnp.sin(second) * 3.0
    >>>   return jnp.sum(temp)
    >>>
    >>> fun(tg.Tensor([1, 2, 3]), tg.Tensor([4, 5, 6]))
  """

  @TinyJit
  def eval_tg(*args):
    jax_args = [
      jax.tree_util.tree_map(lambda p: jax.numpy.array(p.numpy()), v) for v in args
    ]
    closed_jaxpr = jax.make_jaxpr(fun)(*jax_args)
    opt_jaxpr, _ = dce_jaxpr(closed_jaxpr.jaxpr, [True] * len(closed_jaxpr.out_avals))
    outputs = eval_jaxpr(
      opt_jaxpr,
      closed_jaxpr.consts,
      *[i for v in args for i in jax.tree_util.tree_flatten(v)[0]],
    )
    outputs = [x.realize() for x in outputs]
    if len(outputs) == 1:
      return cast(Tensor, outputs[0])
    else:
      return [cast(Tensor, o) for o in outputs]

  @functools.wraps(fun)
  def wrapper(*args):
    return eval_tg(*args)

  return wrapper  # type: ignore
