import ray
from dataclasses import dataclass, field


@dataclass(repr=False)
class Config:
    ENV_FACTORY = "ENV_FACTORY"
    ENV_SPEC = "ENV_SPEC_FACTORY"
    ARTIFACT_FACTORY = "ARTIFACT_FACTORY"
    NUM_UNIVERSES = "NUM_UNIVERSES"
    

    store: dict = field(default_factory=dict)

    def get(self, name):
        assert name in dir(Config), f"{name} not in Config"
        try:
            return self.store[name]
        except KeyError:
            raise KeyError(f"key {name} not set")

    def set(self, name, val):
        assert name in dir(Config), f"{name} not in Config"
        self.store[name] = val


@dataclass(repr=False)
class ConfigProxy:
    handler: None

    def get(self, name):
        return ray.get(self.handler.get.remote(name))

    def set(self, name, val):
        return ray.get(self.handler.set.remote(name, val))


_config_global_handler = None


def get_config_proxy():
    # TODO: should only be called on driver
    global _config_global_handler
    if _config_global_handler is None:
        _config_global_handler = ray.remote(Config).remote()
    return ConfigProxy(_config_global_handler)
