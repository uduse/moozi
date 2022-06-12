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

from moozi.core import Tape


@dataclass
class _AsyncLink:
    callable_obj: Any
    to_read: Union[List[str], str] = "auto"
    to_write: Union[List[str], str] = "auto"

    # validation_type: str = "once"

    _validated: bool = False

    async def __call__(self, tape: object):
        # TODO: slow, cache the parsing and validation results
        keys_to_read = self._get_keys_to_read(tape)
        tape_window = _AsyncLink._read_tape(tape, keys_to_read)

        updates = self.callable_obj(**tape_window)

        if inspect.isawaitable(updates):
            updates = await updates

        if not updates:
            updates = {}

        if not isinstance(updates, dict):
            raise TypeError("updates should either be a dictionary or `None`")

        self._validate_once(tape, updates)

        _AsyncLink._update_tape(tape, updates)

        return updates

    @staticmethod
    def _read_tape(tape, keys_to_read: List[str]):
        return {key: getattr(tape, key) for key in keys_to_read}

    @staticmethod
    def _update_tape(tape, updates: Dict[str, Any]):
        for key, val in updates.items():
            setattr(tape, key, val)

    @staticmethod
    def _tape_has_keys(tape, keys: List[str]) -> bool:
        return set(keys) <= set(tape.__dict__.keys())

    @staticmethod
    def _get_missing_keys(tape, keys: List[str]) -> List[str]:
        return list(set(keys) - set(tape.__dict__.keys()))

    def _validate_updates(self, tape, updates):
        if not isinstance(updates, dict):
            raise TypeError("updates should either be a dictionary or `None`")

        if not _AsyncLink._tape_has_keys(tape, list(updates.keys())):
            raise ValueError(
                "keys "
                + str(_AsyncLink._get_missing_keys(tape, list(updates.keys())))
                + " missing"
            )

        if self.to_write == "auto":
            pass
        elif isinstance(self.to_write, list):
            update_nothing = (not self.to_write) and (not updates)
            if update_nothing:
                pass
            elif self.to_write != list(updates.keys()):
                raise ValueError("write_view keys mismatch.")
        else:
            raise ValueError("`to_write` type not accepted.")

    def _validate_once(self, tape, updates):
        if self._validated:
            return
        else:
            self._validate_updates(tape, updates)
            self._validated = True

    @functools.cached_property
    def _wrapped_func_keys(self):
        return set(inspect.signature(self.callable_obj).parameters.keys())

    def _get_keys_to_read(self, tape):
        if self.to_read == "auto":
            keys = self._wrapped_func_keys
            keys = keys - {"self"}  ## TODO?
            if not _AsyncLink._tape_has_keys(tape, keys):
                raise ValueError(
                    f"{self._get_missing_keys(tape, keys)} not in {str(tape.__dict__.keys())})"
                )
        elif isinstance(self.to_read, list):
            keys = self.to_read
        else:
            raise ValueError("`to_read` type not accepted.")
        return keys


@dataclass
class _Link:
    callable_obj: Any
    to_read: Union[List[str], str] = "auto"
    to_write: Union[List[str], str] = "auto"

    _validated: bool = False

    def __call__(self, tape: object):
        keys_to_read = self._get_keys_to_read(tape)
        tape_window = _Link._read_tape(tape, keys_to_read)

        updates = self.callable_obj(**tape_window)

        self._validate_once(tape, updates)
        _Link._update_tape(tape, updates)

        return updates

    @staticmethod
    def _read_tape(tape, keys_to_read: List[str]):
        return {key: getattr(tape, key) for key in keys_to_read}

    @staticmethod
    def _update_tape(tape, updates: Dict[str, Any]):
        for key, val in updates.items():
            setattr(tape, key, val)

    @staticmethod
    def _tape_has_keys(tape, keys: List[str]) -> bool:
        return set(keys) <= set(tape.__dict__.keys())

    @staticmethod
    def _get_missing_keys(tape, keys: List[str]) -> List[str]:
        return list(set(keys) - set(tape.__dict__.keys()))

    def _validate_updates(self, tape, updates):
        if not isinstance(updates, dict):
            raise TypeError("updates should either be a dictionary or `None`")

        if not _AsyncLink._tape_has_keys(tape, list(updates.keys())):
            raise ValueError(
                "keys "
                + str(_AsyncLink._get_missing_keys(tape, list(updates.keys())))
                + " missing"
            )

        if self.to_write == "auto":
            pass
        elif isinstance(self.to_write, list):
            update_nothing = (not self.to_write) and (not updates)
            if update_nothing:
                pass
            elif self.to_write != list(updates.keys()):
                raise ValueError("write_view keys mismatch.")
        else:
            raise ValueError("`to_write` type not accepted.")

    def _validate_once(self, tape, updates):
        if self._validated:
            return
        else:
            self._validate_updates(tape, updates)
            self._validated = True

    @functools.cached_property
    def _wrapped_func_keys(self):
        return set(inspect.signature(self.callable_obj).parameters.keys())

    def _get_keys_to_read(self, tape):
        if self.to_read == "auto":
            keys = self._wrapped_func_keys
            keys = keys - {"self"}  ## TODO?
            if not _AsyncLink._tape_has_keys(tape, keys):
                raise ValueError(
                    f"{self._get_missing_keys(tape, keys)} not in {str(tape.__dict__.keys())})"
                )
        elif isinstance(self.to_read, list):
            keys = self.to_read
        else:
            raise ValueError("`to_read` type not accepted.")
        return keys


@dataclass
class _LinkClassWrapper:
    class_: type

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return link(self.class_(*args, **kwargs))


def link_async(*args, **kwargs):
    if len(args) == 1 and not kwargs and inspect.isclass(args[0]):
        return _LinkClassWrapper(args[0])
    elif len(args) == 1 and not kwargs and callable(args[0]):
        func = args[0]
        return _AsyncLink(func, to_read="auto", to_write="auto")
    else:
        func = functools.partial(_AsyncLink, *args, **kwargs)
        return func


def link(*args, **kwargs):
    if len(args) == 1 and not kwargs and inspect.isclass(args[0]):
        return _LinkClassWrapper(args[0])
    elif len(args) == 1 and not kwargs and callable(args[0]):
        func = args[0]
        return _Link(func, to_read="auto", to_write="auto")
    else:
        func = functools.partial(_Link, *args, **kwargs)
        return func


# TODO: deprecate this class
# @dataclass
# class Universe:
#     tape: Tape
#     laws: List[Callable]

#     def tick(self, times: Optional[int] = 1):
#         if times is None:
#             while 1:
#                 for law in self.laws:
#                     law(self.tape)
#                 if self.tape.interrupt_exit:
#                     break
#         else:
#             for _ in range(times):
#                 for law in self.laws:
#                     law(self.tape)
#                 if self.tape.interrupt_exit:
#                     break

#     def close(self):
#         for law in self.laws:
#             if hasattr(law, "close"):
#                 law.close()


@dataclass
class UniverseAsync:
    tape: Tape
    laws: List[Callable[[object], Awaitable]]

    async def tick(self, times: Optional[int] = 1):
        self.tape.signals = False

        if times is None:
            while 1:
                for law in self.laws:
                    await law(self.tape)
                    if self.tape.signals:
                        return
        else:
            for _ in range(times):
                for law in self.laws:
                    await law(self.tape)
                    if self.tape.signals:
                        return

    def close(self):
        for law in self.laws:
            if hasattr(law, "close"):
                law.close()


@dataclass
class BatchUniverse:
    tape: Tape
    laws: List[Callable[[object], Awaitable]]

    def tick(self):
        self.tape.signals = False
        while 1:
            for law in self.laws:
                law(self.tape)
                if self.tape.signals:
                    return

    def close(self):
        for law in self.laws:
            if hasattr(law, "close"):
                law.close()
