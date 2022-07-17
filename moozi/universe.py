from moozi.laws import Law
from loguru import logger


class Universe:
    def __init__(self, tape, law: Law) -> None:
        assert isinstance(tape, dict)
        self.tape = tape
        self.law = law

    def tick(self):
        self.tape = self.law.apply(self.tape)

    def run(self):
        while True:
            self.tick()
            if self.tape["output"] is not None:
                break
        return self.flush()

    def flush(self):
        ret = self.tape["output"]
        self.tape["output"] = None
        return ret
