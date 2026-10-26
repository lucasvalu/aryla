import logging

from aryla.observability import flow, setup_logging


def test_flow_log_is_silent_by_default():
    setup_logging()
    assert not flow.isEnabledFor(logging.INFO)


def test_verbose_enables_flow_log():
    setup_logging(verbose=True)
    assert flow.isEnabledFor(logging.INFO)
    setup_logging()
