import os
import select
import termios
import tty


class TerminalKeys:
    def __init__(self, fd: int = 0) -> None:
        self.fd = fd
        self._saved = None

    def __enter__(self):
        if not os.isatty(self.fd):
            raise RuntimeError("interactive recording requires a TTY")
        self._saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def poll(self) -> str | None:
        if select.select([self.fd], [], [], 0)[0]:
            return os.read(self.fd, 1).decode(errors="ignore").lower()
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
