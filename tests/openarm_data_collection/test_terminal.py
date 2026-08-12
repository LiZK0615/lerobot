import os

from lerobot.openarm_data_collection.terminal import TerminalKeys


def test_non_tty_is_rejected():
    read_fd, write_fd = os.pipe()
    try:
        try:
            with TerminalKeys(read_fd): pass
        except RuntimeError as error:
            assert "TTY" in str(error)
    finally:
        os.close(read_fd); os.close(write_fd)
