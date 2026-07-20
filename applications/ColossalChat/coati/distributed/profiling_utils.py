import os
import time


class CustomProfiler:
    def __init__(self, name, disabled=True):
        self.disabled = disabled
        if not disabled:
            safe_name = os.path.basename(name)
            if not safe_name or os.sep in safe_name or (os.altsep and os.altsep in safe_name):
                raise ValueError(f"Invalid profiler name: {name!r}")
            self.name = safe_name
            self.pid = os.getpid()
            self.file = open(os.path.join(".", self.name + ".prof"), "w")

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
