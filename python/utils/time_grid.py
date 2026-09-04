from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HalfYearTimeGrid:
    max_time: float = 10.0
    interval_width: float = 0.5

    def __post_init__(self) -> None:
        if self.max_time <= 0 or self.interval_width <= 0:
            raise ValueError("max_time and interval_width must be positive")
        bins = self.max_time / self.interval_width
        if not np.isclose(bins, round(bins)):
            raise ValueError("max_time must be an exact multiple of interval_width")

    @property
    def num_event_bins(self) -> int:
        return int(round(self.max_time / self.interval_width))

    @property
    def tail_index(self) -> int:
        return self.num_event_bins

    @property
    def boundaries(self) -> np.ndarray:
        return np.linspace(0.0, self.max_time, self.num_event_bins + 1)

    @property
    def interval_endpoints(self) -> np.ndarray:
        return self.boundaries[1:]

    def event_bin(self, times: np.ndarray) -> np.ndarray:
        values = np.asarray(times, dtype=float)
        indices = np.ceil(values / self.interval_width).astype(int) - 1
        return np.clip(indices, 0, self.num_event_bins - 1)

    def horizon_bin(self, horizons: np.ndarray) -> np.ndarray:
        values = np.asarray(horizons, dtype=float)
        if np.any(values <= 0) or np.any(values > self.max_time):
            raise ValueError("horizons must lie in (0, max_time]")
        scaled = values / self.interval_width
        if not np.allclose(scaled, np.round(scaled)):
            raise ValueError("horizons must coincide with interval endpoints")
        return np.round(scaled).astype(int) - 1

    def administrative_target(
        self, times: np.ndarray, events: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observed_time = np.asarray(times, dtype=float)
        observed_event = np.asarray(events, dtype=int)
        if observed_time.shape != observed_event.shape:
            raise ValueError("times and events must have matching shapes")
        censored_time = np.minimum(observed_time, self.max_time)
        censored_event = observed_event.copy()
        censored_event[observed_time > self.max_time] = 0
        target_bin = self.event_bin(censored_time)
        target_bin[censored_event == 0] = self.tail_index
        return censored_time, censored_event, target_bin
