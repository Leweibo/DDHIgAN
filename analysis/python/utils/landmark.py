from __future__ import annotations


def validate_landmark_horizons(
    landmark: float,
    horizons: list[float],
    max_time: float,
) -> None:
    if landmark < 0 or max_time <= 0:
        raise ValueError("landmark must be non-negative and max_time must be positive")
    if landmark >= max_time:
        raise ValueError("landmark must be earlier than the model maximum time")
    invalid = [h for h in horizons if h <= 0 or landmark + h > max_time + 1e-9]
    if invalid:
        raise ValueError(
            f"landmark {landmark} plus horizon(s) {invalid} exceeds model maximum {max_time}"
        )
