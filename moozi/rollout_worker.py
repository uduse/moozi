from typing import Callable

import ray


class RolloutWorker:
    def __init__(
        self, universe_factory: Callable, name: str = "rollout_worker"
    ) -> None:
        self.universe = universe_factory()
        self.name = name
        print(f"{self.name} created")

        from loguru import logger

        logger.remove()
        logger.add(f"logs/rw.{self.name}.debug.log", level="DEBUG")
        logger.add(f"logs/rw.{self.name}.info.log", level="INFO")
        logger.info(
            f"RolloutWorker created, name: {self.name}, universe include {self.universe.tape.keys()}"
        )

    def run(self):
        return self.universe.run()

    def set(self, key, value):
        if isinstance(value, ray.ObjectRef):
            value = ray.get(value)
        self.universe.tape[key] = value
