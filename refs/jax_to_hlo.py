from ast import literal_eval
import importlib
import functools

from absl import app
from absl import flags
import jax.api
import jax.numpy as jnp
from jax.lib import xla_client


def jax_to_hlo(
        fn,
        input_shapes,
        constants=None
):
    """Converts a JAX function to an HLO module.
    Args:
      fn: Function to convert.
      input_shapes: List of tuples (arg name, xla_client.Shape),
        indicating the shapes of the arguments to fn.  The order of parameters in
        the resulting XLA program will match the order in this list.
      constants: Dict mapping function argument name to a Python value.  Specified
        arguments these values as compile-time constants.
    Returns:
      A tuple (serialized_hlo_proto, hlo_text).
    """
    if not constants:
        constants = {}

    overlapping_args = {arg_name for arg_name, _ in input_shapes} & set(
        constants.keys())
    if overlapping_args:
        raise ValueError(
            'Arguments appear in both `input_shapes` and `constants`: %s' %
            ', '.join(sorted(overlapping_args)))

    args = []
    for arg_name, shape in input_shapes:
        if not shape.is_array():
            raise ValueError('Shape %s is not an array, but currently only arrays '
                             'are supported (i.e., no tuples, nor tokens).' % str(shape))

        # Check that `shape` either doesn't have a layout or has the default layout.
        #
        # TODO(jlebar): This could be simpler if the Shape class exposed its layout,
        # or if Shape exposed a function to unconditionally use the default layout.
        shape_with_default_layout = xla_client.Shape.array_shape(
            shape.xla_element_type(),
            shape.dimensions()).with_major_to_minor_layout_if_absent()
        if (shape.with_major_to_minor_layout_if_absent() !=
                shape_with_default_layout):
            raise ValueError('Shape %s has a non-default layout, but only '
                             'the default layout is allowed.' % str(shape))

        args.append(jnp.zeros(shape.dimensions(), dtype=shape.numpy_dtype()))

    # Curry `constants` into the function.
    fn_curried = functools.partial(fn, **constants)

    # Wrapper that takes in args in the order of `input_shapes` and converts them
    # to kwargs for calling `fn`.
    def ordered_wrapper(*args):
        arg_names = [arg_name for arg_name, _ in input_shapes]
        return fn_curried(**dict(zip(arg_names, args)))

    comp = jax.api.xla_computation(ordered_wrapper)(*args)
    return (comp.as_serialized_hlo_module_proto(), comp.as_hlo_text())


def main(argv):
    if len(argv) != 1:
        raise app.UsageError('No positional arguments are accepted.')

    if not FLAGS.hlo_proto_dest and not FLAGS.hlo_text_dest:
        raise app.Error('At least one of --hlo_proto_dest and '
                        '--hlo_text_dest is required.')

    module_name, fn_name = FLAGS.fn.rsplit('.', 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)

    input_shapes = [(name, xla_client.Shape(shape_str))
                    for name, shape_str in literal_eval(FLAGS.input_shapes)]

    # Parse --constants and --evaled_constants.
    constants = {}
    for k, v in literal_eval(FLAGS.constants).items():
        if isinstance(v, list):
            v = jnp.asarray(v)
        constants[k] = v

    for k, v in literal_eval(FLAGS.evaled_constants).items():
        if isinstance(v, str):
            v = literal_eval(v)
        if isinstance(v, list):
            v = jnp.asarray(v)
        if k in constants:
            raise ValueError(
                'Argument appears in both --constants and --evaled_constants: %s' % k)
        constants[k] = v

    hlo_proto, hlo_text = jax_to_hlo(
        fn,
        input_shapes,
        constants
    )

    if FLAGS.hlo_proto_dest:
        with open(FLAGS.hlo_proto_dest, 'wb') as f:
            f.write(hlo_proto)

    if FLAGS.hlo_text_dest:
        with open(FLAGS.hlo_text_dest, 'w') as f:
            f.write(hlo_text)
