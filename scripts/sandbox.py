# %%
from threading import Thread
from dataclasses import dataclass
from queue import Queue, Empty
from time import sleep
from ray import ObjectRef
import ray


@ray.remote
class C:
    def __init__(self) -> None:
        self._q: Queue = Queue()
        self._processed = []
        self._thread = Thread(target=self.processing_thread, daemon=True)
        self._thread.start()

    def add(self):
        print("adding")
        self._q.put(1)
        print("adding done")

    def get(self):
        print("get")
        while True:
            if len(self._processed) > 0:
                print("getting done")
                return self._processed.pop()
            sleep(0.1)

    def processing_thread(self):
        while True:
            print("processing")
            item = self._q.get(block=False)
            item = item + 1
            self._processed.append(item)


c = C.remote()
# %%
c.get.remote()

# %%
ray.add.remote()