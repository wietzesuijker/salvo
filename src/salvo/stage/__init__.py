"""Data-availability gate. Opt-in: callers pass an explicit manifest."""

from salvo.stage.gate import assert_data_available
from salvo.stage.gate import assert_data_available as gate

__all__ = ["assert_data_available", "gate"]
