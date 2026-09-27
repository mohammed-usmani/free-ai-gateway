"""One gateway process per state file: cooldowns, realtime session tags and LiteLLM state live
in memory, so a second worker would split them (and lose voice turns). Scale out via Redis."""

import fcntl
import os


class SingleProcessLock:
    def __init__(self, state_path: str):
        self.path = f"{state_path}.lock"
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._fd)
            raise RuntimeError(f"another process holds {self.path}: run one gateway process per state "
                               f"file (uvicorn --workers 1); in-memory routing state can't be shared") from None

    def release(self):
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
