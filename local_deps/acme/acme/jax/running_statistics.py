# Lint as: python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
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

"""Utility functions to compute running statistics."""

from typing import Optional, Tuple, Union

from acme import types
from acme.jax import utils
from acme.utils import tree_utils
import chex
import jax
import jax.numpy as jnp
import numpy as np
import tree


@chex.dataclass
class NestedMeanStd:
  """A container for running statistics (mean, std) of possibly nested data."""
  mean: types.NestedArray
  std: types.NestedArray


@chex.dataclass
class RunningStatisticsState(NestedMeanStd):
  """Full state of running statistics computation."""
  count: int
  summed_variance: types.NestedArray


def init_state(nest: types.Nest) -> RunningStatisticsState:
  """Initializes the running statistics for the given nested structure."""
  dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
  return RunningStatisticsState(
      count=0,
      mean=utils.zeros_like(nest, dtype=dtype),
      summed_variance=utils.zeros_like(nest, dtype=dtype),
      # Initialize with ones to make sure normalization works correctly
      # in the initial state.
      std=utils.ones_like(nest, dtype=dtype))


# TODO(b/187374660): annotate `batch_dims`.
def _validate_batch_shapes(batch: types.NestedArray,
                           reference_sample: types.NestedArray,
                           batch_dims) -> None:
  """Verifies shapes of the batch leaves against the reference sample.

  Checks that batch dimensions are the same in all leaves in the batch.
  Checks that non-batch dimensions for all leaves in the batch are the same
  as in the reference sample.

  Arguments:
    batch: the nested batch of data to be verified.
    reference_sample: the nested array to check non-batch dimensions.
    batch_dims: a Tuple of indices of batch dimensions in the batch shape.

  Returns:
    None.
  """
  batch_shape = tree.flatten(batch)[0].shape
  batch_dims = np.sort(np.array(batch_dims, dtype=np.int32))
  batch_shapes = np.take(batch_shape, batch_dims)
  shape_indices = batch_dims - np.arange(len(batch_dims))

  def validate_node_shape(reference_sample: jnp.ndarray,
                          batch: jnp.ndarray) -> None:
    expected_shape = tuple(
        np.insert(reference_sample.shape, shape_indices, batch_shapes))
    assert batch.shape == expected_shape, f'{batch.shape} != {expected_shape}'

  tree_utils.fast_map_structure(validate_node_shape, reference_sample, batch)


def update(state: RunningStatisticsState,
           batch: types.NestedArray,
           axis: Optional[Union[int, Tuple[int, ...]]] = None,
           std_min_value: float = 1e-6,
           std_max_value: float = 1e6,
           validate_shapes: bool = True) -> RunningStatisticsState:
  """Updates the running statistics with the given batch of data.

  Note: by default will use int32 for counts and float32 for accumulated
  variance. This results in an integer overflow after 2^31 data points and
  degrading precision after 2^24 batch updates or even earlier if variance
  updates have large dynamic range.
  To improve precision, consider setting jax_enable_x64 to True, see
  https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision

  Arguments:
    state: The running statistics before the update.
    batch: The data to be used to update the running statistics.
    axis: Axis or axes along which the aggregation is performed. The default,
      axis=None, will aggregate along all dimensions. The semantics is identical
      to that of the `axis` argument of Numpy aggregation functions, such as
      `numpy.sum`.
    std_min_value: Minimum value for the standard deviation.
    std_max_value: Maximum value for the standard deviation.
    validate_shapes: If true, the shapes of all leaves of the batch will be
      validated. Enabled by default. Doesn't impact performance when jitted.

  Returns:
    Updated running statistics.
  """
  batch_shape = tree.flatten(batch)[0].shape
  # If None, all dimensions are treated as batch dimensions.
  batch_dims = tuple(range(len(batch_shape))) if axis is None else axis
  batch_dims = ((batch_dims,) if isinstance(batch_dims, int) else batch_dims)
  # We use the original Numpy for all shape-related computations, as data volume
  # is very low here, no effect on performance when jitted, but larger set of
  # available operations compared to jax.numpy.
  # All other computations are done in jax.numpy for performance.
  batch_size = np.prod(np.take(batch_shape, batch_dims))
  count = state.count + batch_size

  # Validation is important. If the shapes don't match exactly, but are
  # compatible, arrays will be silently broadcasted resulting in incorrect
  # statistics. Disabled by default to not affect performance.
  if validate_shapes:
    _validate_batch_shapes(batch, state.mean, batch_dims)

  def _compute_node_statistics(
      mean: jnp.ndarray, summed_variance: jnp.ndarray,
      batch: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert isinstance(mean, jnp.ndarray), type(mean)
    assert isinstance(summed_variance, jnp.ndarray), type(summed_variance)
    # The mean and the sum of past variances are updated with Welford's
    # algorithm using batches (see https://stackoverflow.com/q/56402955).
    diff_to_old_mean = batch - mean
    mean_update = jnp.sum(diff_to_old_mean, axis=batch_dims) / count
    mean = mean + mean_update

    diff_to_new_mean = batch - mean
    variance_update = diff_to_old_mean * diff_to_new_mean
    variance_update = jnp.sum(variance_update, axis=batch_dims)
    summed_variance = summed_variance + variance_update
    return mean, summed_variance

  updated_stats = tree_utils.fast_map_structure(_compute_node_statistics,
                                                state.mean,
                                                state.summed_variance, batch)
  # map_structure_up_to is slow, so shortcut if we know the input is not
  # structured.
  if isinstance(state.mean, jnp.ndarray):
    mean, summed_variance = updated_stats
  else:
    # Reshape the updated stats from `nest(mean, summed_variance)` to
    # `nest(mean), nest(summed_variance)`.
    mean, summed_variance = [
        tree.map_structure_up_to(
            state.mean, lambda s, i=idx: s[i], updated_stats)
        for idx in range(2)
    ]

  def compute_std(summed_variance: jnp.ndarray) -> jnp.ndarray:
    assert isinstance(summed_variance, jnp.ndarray)
    # Summed variance can get negative due to rounding errors.
    summed_variance = jnp.maximum(summed_variance, 0)
    std = jnp.sqrt(summed_variance / count)
    std = jnp.clip(std, std_min_value, std_max_value)
    return std

  std = tree_utils.fast_map_structure(compute_std, summed_variance)

  return RunningStatisticsState(
      count=count, mean=mean, summed_variance=summed_variance, std=std)


def normalize(batch: types.NestedArray,
              mean_std: NestedMeanStd,
              max_abs_value: Optional[float] = None) -> types.NestedArray:
  """Normalizes data using running statistics."""

  def normalize_leaf(data: jnp.ndarray, mean: jnp.ndarray,
                     std: jnp.ndarray) -> jnp.ndarray:
    data = (data - mean) / std
    if max_abs_value is not None:
      # TODO(b/124318564): remove pylint directive
      data = jnp.clip(data, -max_abs_value, +max_abs_value)  # pylint: disable=invalid-unary-operand-type
    return data

  return tree_utils.fast_map_structure(normalize_leaf, batch, mean_std.mean,
                                       mean_std.std)
