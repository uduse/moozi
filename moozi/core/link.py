import functools
import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Union,
)
from dataclasses import dataclass


def link(fn):
    if inspect.isclass(fn):
        return _link_class(fn)

    keys = inspect.signature(fn).parameters.keys()

    def _wrapper(tape):
        kwargs = {}
        for k in keys:
            kwargs[k] = tape[k]
        updates = fn(**kwargs)
        new_tape = tape.copy()
        if updates:
            new_tape.update(updates)
        return new_tape

    _wrapper.__orig_fn = fn

    return _wrapper


def _link_class(cls):
    @dataclass
    class _LinkClassWrapper:
        class_: type

        def __call__(self, *args, **kwargs):
            return link(self.class_(*args, **kwargs))

    return _LinkClassWrapper(cls)
