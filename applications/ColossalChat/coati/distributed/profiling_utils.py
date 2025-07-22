import os
import time


class CustomProfiler:
    def __init__(self, name, disabled=True):
        self.disabled = disabled
        if not disabled:
            self.name = name
            self.pid = os.getpid()
            self.file = open(f"{name}.prof", "w")

    def _log(self, message):
        if self.disabled:
            return
        current_time = time.time()
        self.file.write(f"{current_time} {self.name} {self.pid}:: {message}\n")
        self.file.flush()

    def log(self, message):
        if self.disabled:
            return
        current_time = time.time()
        self.file.write(f"[Log]: {current_time} {self.name} {self.pid}:: {message}\n")
        self.file.flush()

    def enter(self, event_name):
        self._log(f"Enter {event_name}")

    def exit(self, event_name):
        self._log(f"Exit {event_name}")

    def close(self):
        if self.disabled:
            return
        self.file.close()
        print(f"Profiler data written to {self.name}.prof")
