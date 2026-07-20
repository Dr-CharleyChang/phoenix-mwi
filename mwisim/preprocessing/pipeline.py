"""Composable preprocessing stages for canonical measured-data records."""
from __future__ import annotations

from collections.abc import Iterable

from ..core.interfaces import Preprocessor
from ..core.registry import register
from ..data.schema import MeasurementSet


def require_measurement_set(data) -> MeasurementSet:
    """Return data after enforcing the Phase-2 measured-data contract."""
    if not isinstance(data, MeasurementSet):
        raise TypeError(
            "measured-data preprocessing requires a MeasurementSet; "
            f"received {type(data).__name__}"
        )
    return data


@register("preprocessor", "pipeline")
class PreprocessingPipeline(Preprocessor):
    """Apply an ordered sequence of preprocessing stages.

    Each stage receives the immutable record returned by the previous stage. This makes
    order explicit: gain calibration followed by reference subtraction is generally not
    interchangeable with the reverse order when gains differ between scans.
    """

    def __init__(self, stages: Iterable[Preprocessor]):
        self.stages = tuple(stages)
        for index, stage in enumerate(self.stages):
            if not isinstance(stage, Preprocessor):
                raise TypeError(
                    f"stage {index} must implement Preprocessor; "
                    f"received {type(stage).__name__}"
                )

    def apply(self, data, **kwargs) -> MeasurementSet:
        record = require_measurement_set(data)
        for index, stage in enumerate(self.stages):
            record = require_measurement_set(stage.apply(record, **kwargs))
        return record


__all__ = ["PreprocessingPipeline", "require_measurement_set"]
