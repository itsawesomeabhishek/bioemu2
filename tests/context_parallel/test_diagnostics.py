import logging

import pytest
import torch

from bioemu2.context_parallel.diagnostics import attach_trace_hooks
from bioemu2.context_parallel.utils.session import _CP_LOGGER_NAME, TRACE_LEVEL


@pytest.mark.parametrize(
    ("level", "should_trace"),
    [(TRACE_LEVEL, True), (logging.DEBUG, False)],
)
def test_trace_logs_only_matching_modules_at_trace(level, should_trace, caplog):
    triangle_cls = type("Triangle", (torch.nn.Module,), {"forward": lambda self, x: x + 1})
    other_cls = type("NotMatched", (torch.nn.Module,), {"forward": lambda self, x: x * 2})
    model = torch.nn.Module()
    model.a = triangle_cls()
    model.b = other_cls()
    value = torch.tensor([1.0, 2.0])
    with caplog.at_level(level, logger=_CP_LOGGER_NAME):
        attach_trace_hooks(model)
        caplog.clear()
        torch.testing.assert_close(model.a(value), value + 1)
        torch.testing.assert_close(model.b(value), value * 2)
    assert caplog.messages == (["enter a (Triangle)", "exit  a (Triangle)"] if should_trace else [])
