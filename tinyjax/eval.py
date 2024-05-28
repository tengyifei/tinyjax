import jax
import numpy as np
import tinygrad as tg
from jax import core
from jax.core import Jaxpr, ShapedArray
from jax.util import safe_map

from tinyjax.ops import TG_REGISTRY


def eval_jaxpr(jaxpr: Jaxpr, consts, *args, debug: bool = False):
  assert type(debug) is bool

  # Mapping from variable -> value
  env = {}

  def read(var):
    # Literals are values baked into the Jaxpr
    if type(var) is core.Literal:
      return tg.Tensor(np.array(var.val))
    return env[var]

  def write(var, val):
    match val:
      case int() | float() | np.floating() | np.integer():
        env[var] = tg.Tensor(np.array(val))
      case jax.Array():
        env[var] = tg.Tensor(np.array(val))
      case _:
        env[var] = val
    if debug:
      print(f"Writing to {var}: {env[var].numpy()}")

  # Bind args and consts to environment
  safe_map(write, jaxpr.invars, args)
  safe_map(write, jaxpr.constvars, consts)

  # Loop through equations and evaluate primitives
  for eqn in jaxpr.eqns:
    # Read inputs to equation from environment
    invals = safe_map(read, eqn.invars)
    if eqn.primitive not in TG_REGISTRY:
      raise NotImplementedError(
        f"{eqn.primitive} does not have an implementation: {eqn}"
      )
    if debug:
      print(f"Processing {eqn}")
    outvals = TG_REGISTRY[eqn.primitive](*invals, **eqn.params)
    # Primitives may return multiple outputs or not
    if not eqn.primitive.multiple_results:
      outvals = [outvals]
    for var, val in zip(eqn.outvars, outvals):
      if isinstance(var.aval, ShapedArray):
        assert var.aval.shape == val.shape, f"{var.aval.shape} != {val.shape}"
    # Write the results of the primitive into the environment
    safe_map(write, eqn.outvars, outvals)

  # Read the final result of the Jaxpr from the environment
  return safe_map(read, jaxpr.outvars)
