import json

import pytest

from lerobot.openarm_data_collection.ros_receiver import RosSnapshotReceiver, decode_recording_snapshot


def payload(sequence=1):
    vector = {"values": [0.0] * 16, "received_monotonic_ns": 90, "ros_stamp_ns": 80}
    return {
        "version": 1, "sequence": sequence, "sent_monotonic_ns": 100,
        "state": vector, "action": vector, "command": None
    }


class FakeSocket:
    def __init__(self, packets):
        self.packets = list(packets)
    def setblocking(self, value): pass
    def bind(self, address): self.address = address
    def recvfrom(self, size):
        if not self.packets: raise BlockingIOError
        return self.packets.pop(0), ("127.0.0.1", 1)
    def close(self): pass


def encode(value): return json.dumps(value).encode()


def test_decoder_rejects_wrong_length_and_non_finite_values():
    value = payload()
    value["state"]["values"] = [0.0] * 15
    with pytest.raises(ValueError, match="16"):
        decode_recording_snapshot(encode(value), 4096)


def test_receiver_returns_newest_sequence():
    sock = FakeSocket([encode(payload(8)), encode(payload(9))])
    receiver = RosSnapshotReceiver(socket_factory=lambda: sock)
    assert receiver.poll().sequence == 9
    receiver.close()
