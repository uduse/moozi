import functools
import inspect
from typing import Any, Dict, List, Union
import attr

from absl import logging
from jax._src.numpy.lax_numpy import isin
from ray.cloudpickle.cloudpickle import instance


@attr.s(auto_attribs=True)
class Link:
    callable_obj: Any
    to_read: Union[List[str], str] = "auto"
    to_write: Union[List[str], str] = "auto"

    async def __call__(self, artifact: object):
        keys_to_read = self._get_keys_to_read(artifact)
        artifact_window = Link._read_artifact(artifact, keys_to_read)

        updates = self.callable_obj(**artifact_window)

        if inspect.isawaitable(updates):
            updates = await updates

        if not updates:
            updates = {}

        self._validate_updates(artifact, updates)
        Link._update_artifact(artifact, updates)

        return updates

    @staticmethod
    def _read_artifact(artifact, keys_to_read: List[str]):
        return {key: getattr(artifact, key) for key in keys_to_read}

    @staticmethod
    def _update_artifact(artifact, updates: Dict[str, Any]):
        for key, val in updates.items():
            setattr(artifact, key, val)

    @staticmethod
    def _artifact_has_keys(artifact, keys: List[str]) -> bool:
        return set(keys) <= set(artifact.__dict__.keys())

    @staticmethod
    def _get_missing_keys(artifact, keys: List[str]) -> List[str]:
        return list(set(keys) - set(artifact.__dict__.keys()))

    def _validate_updates(self, artifact, updates):
        if not isinstance(updates, dict):
            raise TypeError("updates should either be a dictionary or `None`")

        if not Link._artifact_has_keys(artifact, list(updates.keys())):
            raise ValueError(
                "keys "
                + str(Link._get_missing_keys(artifact, list(updates.keys())))
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

    def _wrapped_func_keys(self):
        return set(inspect.signature(self.callable_obj).parameters.keys())

    def _get_keys_to_read(self, artifact):
        if self.to_read == "auto":
            keys = self._wrapped_func_keys()
            keys = keys - {"self"}  ## TODO?
            if not Link._artifact_has_keys(artifact, keys):
                raise ValueError(f"{str(keys)} not in {str(artifact.__dict__.keys())})")
        elif isinstance(self.to_read, list):
            keys = self.to_read
        else:
            raise ValueError("`to_read` type not accepted.")
        return keys


@attr.s
class LinkClassWrapper:
    class_: type = attr.ib()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return link(self.class_(*args, **kwargs))


def link(*args, **kwargs):
    if len(args) == 1 and not kwargs and inspect.isclass(args[0]):
        return LinkClassWrapper(args[0])
    elif len(args) == 1 and not kwargs and callable(args[0]):
        func = args[0]
        return Link(func, to_read="auto", to_write="auto")
    else:
        func = functools.partial(Link, *args, **kwargs)
        return func


@attr.s
class Universe:
    artifact = attr.ib()
    laws = attr.ib()

    def tick(self, times=1):
        for _ in range(times):
            for law in self.laws:
                law(self.artifact)

    def close(self):
        for law in self.laws:
            if hasattr(law, "close"):
                law.close()


@attr.s
class UniverseAsync:
    artifact = attr.ib()
    laws = attr.ib()

    async def tick(self, times=1):
        for _ in range(times):
            for law in self.laws:
                await law(self.artifact)

    def close(self):
        for law in self.laws:
            if hasattr(law, "close"):
                law.close()
