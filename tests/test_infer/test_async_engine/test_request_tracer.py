import pytest

from colossalai.inference.core.async_engine import Tracer
from colossalai.inference.struct import Sequence


class SampleEvent:
    def __init__(self):
        self.flag = False

    def set(self):
        self.flag = True

    def clear(self):
        self.flag = False


def test_request_tracer():
    tracker = Tracer()
    tracker.new_requests_event = SampleEvent()
    stream_1 = tracker.add_request(1)
    assert tracker.new_requests_event.flag
    new = tracker.get_new_requests()
    assert not tracker.new_requests_event.flag
    assert len(new) == 1
    assert new[0]["request_id"] == 1
    assert not stream_1.finished

    stream_2 = tracker.add_request(2)
    stream_3 = tracker.add_request(3)
    assert tracker.new_requests_event.flag
    new = tracker.get_new_requests()
    assert not tracker.new_requests_event.flag
    assert len(new) == 2
    assert new[0]["request_id"] == 2
    assert new[1]["request_id"] == 3
    assert not stream_2.finished
    assert not stream_3.finished

    # request_ids must be unique
    with pytest.raises(KeyError):
        tracker.add_request(1)
    assert not tracker.new_requests_event.flag

    tracker.abort_request(1)
    new = tracker.get_new_requests()
    assert not new

    stream_4 = tracker.add_request(4)
    tracker.abort_request(4)
    assert tracker.new_requests_event.flag
    new = tracker.get_new_requests()
    assert not new
    assert stream_4.finished

    stream_5 = tracker.add_request(5)
    assert tracker.new_requests_event.flag
    tracker.process_finished_request(Sequence(2, "output", [], 4, [], 0, 0))
    new = tracker.get_new_requests()
    assert not tracker.new_requests_event.flag
    assert len(new) == 1
    assert new[0]["request_id"] == 5
    assert stream_2.finished
    assert not stream_5.finished


if __name__ == "__main__":
    test_request_tracer()
