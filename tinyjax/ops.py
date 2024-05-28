from typing import Any, Sequence

import numpy as np
import tinygrad as tg
import tinygrad.dtype
import tinygrad.function as tgf
from jax import lax
from jax._src.pjit import pjit_p
from jax._src.typing import Shape
from jax.core import ClosedJaxpr
from jax.typing import DTypeLike

TG_REGISTRY = {}


# mul
def tg_mul(a: tg.Tensor, b: tg.Tensor) -> tg.Tensor:
  return a.mul(b)


TG_REGISTRY[lax.mul_p] = tg_mul


# reshape
def tg_reshape(
  operand: tg.Tensor, new_sizes: Shape, dimensions: Sequence[int] | None = None
) -> tg.Tensor:
  shape = list(int(v) for v in new_sizes)
  if dimensions:
    permuted_shape = [0] * len(dimensions)
    for i, d in enumerate(dimensions):
      permuted_shape[d] = shape[i]
    shape = permuted_shape
  return operand.reshape(shape)


TG_REGISTRY[lax.reshape_p] = tg_reshape


# div
def tg_div(a: tg.Tensor, b: tg.Tensor) -> tg.Tensor:
  return a.div(b)


TG_REGISTRY[lax.div_p] = tg_div


# reduce_window_sum
def tg_reduce_window_sum(
  operand: tg.Tensor,
  window_dimensions: Shape,
  window_strides: Sequence[int],
  padding: Sequence[tuple[int, int]],
  base_dilation: Sequence[int] | None = None,
  window_dilation: Sequence[int] | None = None,
) -> tg.Tensor:
  # a:f32[3,14,14,32] = reduce_window_sum[
  #   base_dilation=(1, 1, 1, 1)
  #   padding=((0, 0), (0, 0), (0, 0), (0, 0))
  #   window_dilation=(1, 1, 1, 1)
  #   window_dimensions=(1, 2, 2, 1)
  #   window_strides=(1, 2, 2, 1)
  # ]
  assert padding == ((0, 0), (0, 0), (0, 0), (0, 0))
  assert base_dilation == (1, 1, 1, 1)
  window_dilation = window_dilation or [1, 1, 1, 1]
  pooled = operand._pool(
    k_=tuple(window_dimensions),
    stride=tuple(window_strides),
    dilation=tuple(window_dilation),
  )
  return pooled.sum(axis=tuple(range(0 - len(window_dimensions), 0)))


TG_REGISTRY[lax.reduce_window_sum_p] = tg_reduce_window_sum


# reduce_sum
def tg_reduce_sum(operand: tg.Tensor, axes: Sequence[int]) -> tg.Tensor:
  return operand._reduce(tgf.Sum, tuple(axes))


TG_REGISTRY[lax.reduce_sum_p] = tg_reduce_sum


# convert_element_type
def tg_convert_element_type(
  operand: Any, new_dtype: DTypeLike, weak_type: bool = False
) -> tg.Tensor:
  match operand:
    case int() | float() | np.floating() | np.integer():
      return tg.Tensor(np.array(operand).astype(new_dtype))
    case tg.Tensor():
      return operand.cast(convert_dtype(np.dtype(new_dtype)))
    case _:
      raise RuntimeError(f"Unsupported operand type {type(operand)}")


def convert_dtype(dtype: np.dtype) -> tinygrad.dtype.DType:
  match dtype:
    case np.float16:
      return tinygrad.dtypes.float16
    case np.float32:
      return tinygrad.dtypes.float32
    case np.int32:
      return tinygrad.dtypes.int32
    case np.int64:
      return tinygrad.dtypes.int64
    case _:
      raise RuntimeError(f"Unsupported dtype f{dtype}")


TG_REGISTRY[lax.convert_element_type_p] = tg_convert_element_type


# add
def tg_add(a: tg.Tensor, b: tg.Tensor) -> tg.Tensor:
  return a.add(b)


TG_REGISTRY[lax.add_p] = tg_add


# logistic
def tg_logistic(x: tg.Tensor) -> tg.Tensor:
  return x.sigmoid()


TG_REGISTRY[lax.logistic_p] = tg_logistic


# sin
def tg_sin(x: tg.Tensor) -> tg.Tensor:
  return x.sin()


TG_REGISTRY[lax.sin_p] = tg_sin


# dot_general
def tg_dot_general(
  lhs: tg.Tensor,
  rhs: tg.Tensor,
  dimension_numbers: lax.DotDimensionNumbers,
  precision: lax.PrecisionLike = None,
  preferred_element_type: DTypeLike | None = None,
) -> tg.Tensor:
  # a:f32[3,256] = dot_general[dimension_numbers=(([1], [0]), ([], []))] b c
  assert precision is None
  assert preferred_element_type is None

  ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims)) = (
    dimension_numbers
  )
  # Generate a set of unique labels for einsum
  labels = "abcdefghijklmnopqrstuvwxyz"

  # Assign labels to each dimension of lhs and rhs
  lhs_labels = [""] * len(lhs.shape)
  rhs_labels = [""] * len(rhs.shape)

  # Assign labels for contracting dimensions
  for i, (lhs_dim, rhs_dim) in enumerate(
    zip(lhs_contracting_dims, rhs_contracting_dims)
  ):
    label = labels[i]
    lhs_labels[lhs_dim] = label
    rhs_labels[rhs_dim] = label

  # Start index for batch and other dimensions
  start_idx = len(lhs_contracting_dims)

  # Assign labels for batch dimensions
  for i, (lhs_dim, rhs_dim) in enumerate(
    zip(lhs_batch_dims, rhs_batch_dims), start=start_idx
  ):
    label = labels[i]
    lhs_labels[lhs_dim] = label
    rhs_labels[rhs_dim] = label

  # Assign labels for remaining dimensions
  for i, label in enumerate(lhs_labels):
    if label == "":
      lhs_labels[i] = labels[start_idx]
      start_idx += 1

  for i, label in enumerate(rhs_labels):
    if label == "":
      rhs_labels[i] = labels[start_idx]
      start_idx += 1

  # Construct the einsum string
  lhs_subscripts = "".join(lhs_labels)
  rhs_subscripts = "".join(rhs_labels)
  result_subscripts = "".join([
    label
    for label in lhs_labels + rhs_labels
    if label not in lhs_labels or label not in rhs_labels
  ])

  formula = f"{lhs_subscripts},{rhs_subscripts}->{result_subscripts}"
  return tg.Tensor.einsum(formula, lhs, rhs)


TG_REGISTRY[lax.dot_general_p] = tg_dot_general


# conv_general_dilated
def tg_conv_general_dilated(
  lhs: tg.Tensor,
  rhs: tg.Tensor,
  window_strides: Sequence[int],
  padding: str | Sequence[tuple[int, int]],
  lhs_dilation: Sequence[int] | None = None,
  rhs_dilation: Sequence[int] | None = None,
  dimension_numbers: lax.ConvGeneralDilatedDimensionNumbers = None,
  feature_group_count: int = 1,
  batch_group_count: int = 1,
  precision: lax.PrecisionLike = None,
  preferred_element_type: DTypeLike | None = None,
) -> tg.Tensor:
  # t:f32[3,14,14,64] = conv_general_dilated[
  #    batch_group_count=1
  #    dimension_numbers=ConvDimensionNumbers(lhs_spec=(0, 3, 1, 2), rhs_spec=(3, 2, 0, 1), out_spec=(0, 3, 1, 2))
  #    feature_group_count=1
  #    lhs_dilation=(1, 1)
  #    padding=((1, 1), (1, 1))
  #    precision=None
  #    preferred_element_type=None
  #    rhs_dilation=(1, 1)
  #    window_strides=(1, 1)
  #  ]
  dim_nums = lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
  lhs = lhs.permute(dim_nums.lhs_spec)
  rhs = rhs.permute(dim_nums.rhs_spec)
  assert batch_group_count == 1
  assert feature_group_count == 1
  assert precision is None
  assert preferred_element_type is None
  assert lhs_dilation == (1, 1)
  assert not isinstance(padding, str)
  assert len(padding) == 2
  assert padding[0][0] == padding[0][1]
  assert padding[1][0] == padding[1][1]
  tg_padding = (int(padding[0][0]), int(padding[1][0]))
  assert window_strides == (1, 1)
  result = lhs.conv2d(
    rhs,
    dilation=tuple(int(x) for x in rhs_dilation),  # type: ignore
    padding=tg_padding,  # type: ignore
  )
  return permute_to_spec(result, existing=(0, 2, 3, 1), desired=dim_nums.out_spec)


def permute_to_spec(
  v: tg.Tensor, existing: Sequence[int], desired: Sequence[int]
) -> tg.Tensor:
  assert all(0 <= i <= 3 for i in existing)
  assert all(0 <= i <= 3 for i in desired)
  permute_idx = tuple(int(existing.index(i)) for i in desired)
  return v.permute(permute_idx)


TG_REGISTRY[lax.conv_general_dilated_p] = tg_conv_general_dilated


# pjit
def tg_pjit(*inputs, jaxpr: ClosedJaxpr, **kwargs):
  from .eval import eval_jaxpr

  return eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *inputs)


TG_REGISTRY[pjit_p] = tg_pjit
