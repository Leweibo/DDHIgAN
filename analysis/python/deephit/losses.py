from __future__ import annotations

import torch


def _step_survival(
    times: torch.Tensor,
    survival: torch.Tensor,
    query: torch.Tensor,
    *,
    before: bool,
) -> torch.Tensor:
    side = "left" if before else "right"
    indices = torch.searchsorted(times, query, right=(side == "right")) - 1
    values = survival.new_ones(query.shape)
    valid = indices >= 0
    if valid.any():
        values[valid] = survival[indices[valid]]
    return values


def differentiable_ipcw_brier_loss(
    predicted_survival: torch.Tensor,
    target_time: torch.Tensor,
    target_event: torch.Tensor,
    interval_width: float,
    censoring_times: torch.Tensor,
    censoring_survival: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    min_censoring_survival: float = 1e-6,
) -> torch.Tensor:
    """Graf IPCW integrated Brier loss at fixed yearly horizons 1--10."""
    if predicted_survival.ndim != 2 or predicted_survival.size(1) * interval_width < 10.0:
        raise ValueError("IPCW Brier loss requires survival predictions through 10 years")
    if not torch.isfinite(predicted_survival).all():
        raise ValueError("IPCW Brier predictions must be finite")
    if censoring_times.ndim != 1 or censoring_survival.shape != censoring_times.shape:
        raise ValueError("invalid training-set censoring distribution")
    horizons = torch.arange(
        1.0, 10.0 + 0.5, 1.0,
        dtype=predicted_survival.dtype,
        device=predicted_survival.device,
    )
    bins = torch.round(horizons / interval_width).long() - 1
    predictions = predicted_survival[:, bins]
    base_weight = (
        torch.ones_like(target_time) if sample_weight is None else sample_weight
    )
    scores = []
    for column, horizon in enumerate(horizons):
        event_before = target_event.bool() & (target_time <= horizon)
        event_free = target_time > horizon
        weights = torch.zeros_like(target_time, dtype=predicted_survival.dtype)
        if event_before.any():
            g_event = _step_survival(
                censoring_times,
                censoring_survival,
                target_time[event_before],
                before=True,
            )
            if torch.any(g_event <= min_censoring_survival):
                raise ValueError("training-set censoring survival is too low at event times")
            weights[event_before] = 1.0 / g_event
        if event_free.any():
            g_horizon = _step_survival(
                censoring_times,
                censoring_survival,
                horizon.reshape(1),
                before=False,
            )[0]
            if g_horizon <= min_censoring_survival:
                raise ValueError("training-set censoring survival is too low at a Brier horizon")
            weights[event_free] = 1.0 / g_horizon
        if not torch.any(weights > 0):
            scores.append(predictions[:, column].sum() * 0.0)
            continue
        observed_survival = event_free.to(predicted_survival.dtype)
        squared_error = (observed_survival - predictions[:, column]) ** 2
        scores.append((base_weight * weights * squared_error).sum() / base_weight.sum())
    loss = torch.trapz(torch.stack(scores), horizons) / (horizons[-1] - horizons[0])
    if not torch.isfinite(loss):
        raise ValueError("IPCW Brier loss is non-finite")
    return loss


def negative_log_likelihood_values(
    event_pmf: torch.Tensor,
    tail_probability: torch.Tensor,
    target_time: torch.Tensor,
    target_event: torch.Tensor,
    target_bin: torch.Tensor,
    interval_width: float,
) -> torch.Tensor:
    epsilon = 1e-8
    event_mask = target_event.bool()
    likelihood = torch.empty_like(target_time, dtype=event_pmf.dtype)
    if event_mask.any():
        event_index = target_bin[event_mask].long().clamp(0, event_pmf.size(1) - 1)
        likelihood[event_mask] = event_pmf[event_mask].gather(
            1, event_index.unsqueeze(1)
        ).squeeze(1)
    if (~event_mask).any():
        interval_end = (
            torch.arange(1, event_pmf.size(1) + 1, device=event_pmf.device)
            * interval_width
        )
        after_censor = interval_end.unsqueeze(0) > target_time[~event_mask].unsqueeze(1)
        likelihood[~event_mask] = (
            (event_pmf[~event_mask] * after_censor).sum(1)
            + tail_probability[~event_mask]
        )
    return -torch.log(likelihood.clamp_min(epsilon))


def negative_log_likelihood(
    event_pmf: torch.Tensor,
    tail_probability: torch.Tensor,
    target_time: torch.Tensor,
    target_event: torch.Tensor,
    target_bin: torch.Tensor,
    interval_width: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    epsilon = 1e-8
    loss = negative_log_likelihood_values(
        event_pmf, tail_probability, target_time, target_event, target_bin,
        interval_width,
    )
    if sample_weight is None:
        return loss.mean()
    return (loss * sample_weight).sum() / sample_weight.sum().clamp_min(epsilon)


def time_dependent_ranking_loss(
    cif: torch.Tensor,
    target_time: torch.Tensor,
    target_event: torch.Tensor,
    target_bin: torch.Tensor,
    sigma: float,
    patient_ids: list[str] | None = None,
    sample_weight: torch.Tensor | None = None,
    pairing: str = "legacy",
) -> torch.Tensor:
    """Dynamic-DeepHit ranking loss on a common residual-time scale.

    ``original_patient_balanced`` compares each event prefix to other patients
    at its residual event horizon, excludes a patient's other landmark
    prefixes, and applies the existing per-patient prefix weights.  ``legacy``
    retains historical behavior for archived configurations.
    """
    if pairing not in {"legacy", "original_patient_balanced", "original_ddh_exact"}:
        raise ValueError(f"unsupported ranking pairing: {pairing}")
    if pairing in {"original_patient_balanced", "original_ddh_exact"} and patient_ids is None:
        raise ValueError(f"{pairing} ranking requires patient_ids")
    if pairing == "original_ddh_exact":
        ids = [str(patient_id) for patient_id in patient_ids]
        if len(ids) != len(set(ids)):
            raise ValueError("original_ddh_exact ranking requires one prefix per patient")
    event_mask = target_event.bool()
    if not event_mask.any():
        return cif.sum() * 0.0
    event_bins = target_bin[event_mask].long().clamp(0, cif.size(1) - 1)
    event_times = target_time[event_mask]
    event_risk = cif[event_mask].gather(1, event_bins.unsqueeze(1)).squeeze(1)
    comparator_risk = cif[:, event_bins].transpose(0, 1)
    comparable = target_time.unsqueeze(0) > event_times.unsqueeze(1)
    if pairing == "original_patient_balanced":
        ids = [str(patient_id) for patient_id in patient_ids]
        event_ids = [patient_id for patient_id, is_event in zip(ids, event_mask.tolist()) if is_event]
        different_patient = torch.tensor(
            [[other_id != event_id for other_id in ids] for event_id in event_ids],
            device=target_time.device,
            dtype=torch.bool,
        )
        comparable = comparable & different_patient
    valid_event = comparable.any(1)
    if not valid_event.any():
        return cif.sum() * 0.0
    penalties = torch.exp(
        -((event_risk.unsqueeze(1) - comparator_risk) / sigma)
    )
    if pairing == "original_ddh_exact":
        # Official Dynamic-DeepHit: mean over all j in a minibatch for each
        # event i (including non-comparable zeros), then sum over event i.
        return (penalties * comparable.to(dtype=penalties.dtype)).mean(dim=1).sum()
    pair_weight = comparable.to(dtype=penalties.dtype)
    event_weight = None
    if pairing == "original_patient_balanced" and sample_weight is not None:
        event_weight = sample_weight[event_mask]
        pair_weight = pair_weight * event_weight.unsqueeze(1) * sample_weight.unsqueeze(0)
    weighted = penalties * pair_weight
    per_event = weighted.sum(1) / pair_weight.sum(1).clamp_min(1e-8)
    if event_weight is None:
        return per_event[valid_event].mean()
    valid_weights = event_weight[valid_event]
    return (per_event[valid_event] * valid_weights).sum() / valid_weights.sum().clamp_min(1e-8)


def longitudinal_prediction_loss(
    prediction: torch.Tensor,
    x_values: torch.Tensor,
    x_missing: torch.Tensor,
    sequence_mask: torch.Tensor,
    longitudinal_start: int,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    target = x_values[
        :, 1:, longitudinal_start : longitudinal_start + prediction.size(-1)
    ]
    observed = 1.0 - x_missing[:, 1:]
    valid_transition = sequence_mask[:, :-1] * sequence_mask[:, 1:]
    weight = observed * valid_transition.unsqueeze(-1)
    if sample_weight is not None:
        weight = weight * sample_weight[:, None, None]
    denominator = weight.sum()
    if denominator.item() == 0:
        return prediction.sum() * 0.0
    return (((prediction - target) ** 2) * weight).sum() / denominator


def dynamic_deephit_loss(
    output: dict[str, torch.Tensor],
    x_values: torch.Tensor,
    x_missing: torch.Tensor,
    sequence_mask: torch.Tensor,
    target_time: torch.Tensor,
    target_event: torch.Tensor,
    target_bin: torch.Tensor,
    longitudinal_start: int,
    interval_width: float,
    likelihood_weight: float,
    ranking_weight: float,
    longitudinal_weight: float,
    sigma: float,
    sample_weight: torch.Tensor | None = None,
    patient_ids: list[str] | None = None,
    ranking_pairing: str = "legacy",
    ipcw_brier_weight: float = 0.0,
    censoring_times: torch.Tensor | None = None,
    censoring_survival: torch.Tensor | None = None,
    min_censoring_survival: float = 1e-6,
    ipcw_time: torch.Tensor | None = None,
    ipcw_event: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    likelihood = negative_log_likelihood(
        output["event_pmf"], output["tail_probability"], target_time,
        target_event, target_bin, interval_width, sample_weight,
    )
    ranking = time_dependent_ranking_loss(
        output["cif"], target_time, target_event, target_bin, sigma,
        patient_ids=patient_ids,
        sample_weight=sample_weight,
        pairing=ranking_pairing,
    )
    longitudinal = longitudinal_prediction_loss(
        output["longitudinal_prediction"], x_values, x_missing,
        sequence_mask, longitudinal_start, sample_weight,
    )
    ipcw_brier = likelihood * 0.0
    if ipcw_brier_weight:
        if censoring_times is None or censoring_survival is None:
            raise ValueError("IPCW Brier loss requires a training-set censoring distribution")
        ipcw_brier = differentiable_ipcw_brier_loss(
            output["survival"],
            target_time if ipcw_time is None else ipcw_time,
            target_event if ipcw_event is None else ipcw_event,
            interval_width,
            censoring_times,
            censoring_survival,
            sample_weight,
            min_censoring_survival,
        )
    total = (
        likelihood_weight * likelihood
        + ranking_weight * ranking
        + longitudinal_weight * longitudinal
        + ipcw_brier_weight * ipcw_brier
    )
    return {
        "total": total,
        "likelihood": likelihood,
        "ranking": ranking,
        "longitudinal": longitudinal,
        "ipcw_brier": ipcw_brier,
    }
