import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from deephit.data_loader import DynamicDeepHitDataset, collate_fn
from deephit.train import build_model
from utils.data_utils import get_fold_split, load_norm_stats
from utils.landmark import validate_landmark_horizons
from utils.probabilities import project_probability_roundoff
from utils.time_grid import HalfYearTimeGrid
from utils.query_conditioning import log_survival_from_logits, conditional_log_survival


def predict_landmark(model, loader, landmark, horizons, grid, device, residual_grid=None,
                     conditioning="legacy"):
    if conditioning not in {"legacy", "query_survival", "unconditioned_diagnostic"}:
        raise ValueError("unknown prediction conditioning mode")
    modern = conditioning != "legacy"
    if modern and (not np.isfinite(landmark) or not 0 <= landmark <= 5):
        raise ValueError("candidate queries must lie in [0, 5]")
    residual_grid = np.asarray(
        (list(range(1, 11)) if modern else [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        if residual_grid is None else residual_grid,
        dtype=float,
    )
    if np.any(residual_grid <= 0) or np.any(residual_grid > grid.max_time):
        raise ValueError("residual curve times must be within the model horizon")
    if modern and (np.any(~np.isfinite(residual_grid)) or np.any(residual_grid < 1)
                   or np.any(residual_grid > 10)
                   or any(not np.isfinite(h) or h < 1 or h > 10 or h != int(h) for h in horizons)):
        raise ValueError("candidate curves span 1–10 years; named horizons must be integer years")
    result = {
        "patient_id": [], "true_time": [], "residual_time": [],
        "true_event": [], "Tstart": [], "risk_score": [],
    }
    if modern:
        result.update(query_time=[], last_observation_time=[], prediction_delay=[])
    for horizon in horizons:
        result[f"cond_surv_{int(horizon)}y"] = []
    for residual_time in residual_grid:
        result[f"resid_surv_{residual_time:.1f}y"] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_multimodal = batch.get("x_multimodal")
            if x_multimodal is not None:
                x_multimodal = x_multimodal.to(device)
            model_options = {"x_multimodal": x_multimodal,
                             "query_time": batch["query_time"].to(device)}
            if "x_time" in batch:
                model_options.update(x_time=batch["x_time"].to(device),
                                     x_delta=batch["x_delta"].to(device))
            output = model(
                batch["x_values"].to(device),
                batch["x_missing"].to(device),
                batch["sequence_mask"].to(device),
                **model_options,
            )
            survival = output["survival"].cpu().numpy()
            cif = output["cif"].cpu().numpy()
            residual_time = batch["evaluation_time"].numpy()
            patient_ids = batch["patient_id"]
            result["patient_id"].extend(patient_ids)
            result["residual_time"].extend(residual_time)
            result["true_time"].extend(residual_time + landmark)
            result["true_event"].extend(batch["evaluation_event"].numpy())
            result["Tstart"].extend(np.full(len(patient_ids), landmark))
            if modern:
                query = batch["query_time"].numpy().astype(np.float64)
                last = batch["last_observation_time"].numpy().astype(np.float64)
                delay = query - last
                if not np.isfinite(last).all() or np.any(last < 0) or not np.all(query == landmark):
                    raise ValueError("invalid biopsy-anchored query/history metadata")
                log_s = log_survival_from_logits(output["logits"].cpu().numpy(), grid.num_event_bins)
                times = [10.0, *horizons, *residual_grid]
                conditional = conditional_log_survival(log_s, grid.boundaries, delay, times)
                if conditioning == "unconditioned_diagnostic":
                    conditional = conditional_log_survival(log_s, grid.boundaries, np.zeros_like(delay), times)
                result["query_time"].extend(query)
                result["last_observation_time"].extend(last)
                result["prediction_delay"].extend(delay)
                result["risk_score"].extend(-np.expm1(conditional[:, 0]))
                for index, horizon in enumerate(horizons):
                    result[f"cond_surv_{int(horizon)}y"].extend(np.exp(conditional[:, index + 1]))
                for index, time_value in enumerate(residual_grid):
                    result[f"resid_surv_{time_value:.1f}y"].extend(np.exp(conditional[:, 1 + len(horizons) + index]))
                continue
            result["risk_score"].extend(cif[:, -1])
            for horizon in horizons:
                result[f"cond_surv_{int(horizon)}y"].extend(
                    survival[:, int(grid.horizon_bin(horizon))]
                )
            for time_value in residual_grid:
                result[f"resid_surv_{time_value:.1f}y"].extend(
                    survival[:, int(grid.horizon_bin(time_value))]
                )
    frame = pd.DataFrame(result)
    probability_columns = [
        "risk_score",
        *(f"cond_surv_{int(horizon)}y" for horizon in horizons),
        *(f"resid_surv_{time_value:.1f}y" for time_value in residual_grid),
    ]
    projected, audit = project_probability_roundoff(
        frame[probability_columns].to_numpy(dtype=float)
    )
    frame.loc[:, probability_columns] = projected
    frame.attrs["probability_roundoff"] = audit
    return frame


def prediction_landmark_label(landmark, conditioning):
    """Preserve arbitrary query precision in candidate filenames; legacy names stay fixed."""
    return str(int(landmark)) if conditioning == "legacy" else format(float(landmark), ".17g")


def main():
    parser = argparse.ArgumentParser(description="Formal Dynamic-DeepHit prediction")
    parser.add_argument("--config", default="python/deephit/config_minimal_core.yaml")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--outcome", choices=["ESKD", "Drop50"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--landmarks")
    parser.add_argument("--horizons", default="3,5,10")
    parser.add_argument("--residual-grid")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--unconditioned-diagnostic", action="store_true",
                        help="Use candidate weights without elapsed-survival conditioning; requires separate output")
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    requested_device = cfg["training"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    conditioning = cfg.get("prediction", {}).get("conditioning", "legacy")
    if conditioning != "legacy":
        if conditioning != "query_survival" or not cfg["data"].get("preserve_source_time"):
            raise ValueError("query conditioning requires source-precision history")
        if checkpoint.get("config") != cfg:
            raise ValueError("candidate checkpoint configuration does not match requested configuration")
    if args.unconditioned_diagnostic:
        if conditioning != "query_survival" or args.output_dir is None:
            raise ValueError("diagnostic requires a conditioned candidate and separate output directory")
        if args.output_dir.resolve() == Path(cfg["output"]["prediction_dir"]).resolve():
            raise ValueError("diagnostic cannot overwrite conditioned predictions")
        conditioning = "unconditioned_diagnostic"

    mcfg = cfg["model"]
    development = cfg["training"].get("split_protocol") == "continuous_time_development"
    if development:
        if args.fold is not None or args.split != "test":
            raise ValueError("development prediction uses the locked fold-1 validation set without --fold")
    elif args.fold is None:
        raise ValueError("cross-validation prediction requires --fold")
    prediction_name = cfg["output"].get("prediction_name", mcfg["name"])
    norm_stats_model_name = cfg["output"].get("norm_stats_model_name", mcfg["name"])
    grid = HalfYearTimeGrid(
        max_time=float(mcfg["max_residual_time"]),
        interval_width=float(mcfg["interval_width"]),
    )
    configured_landmarks = cfg["data"].get(
        "evaluation_landmarks", cfg["data"].get("landmarks", [0, 1, 3, 5])
    )
    landmarks = (
        [float(value) for value in args.landmarks.split(",")]
        if args.landmarks else [float(value) for value in configured_landmarks]
    )
    horizons = [float(value) for value in args.horizons.split(",")]
    residual_grid = (
        [float(value) for value in args.residual_grid.split(",")]
        if args.residual_grid else None
    )
    for horizon in horizons:
        grid.horizon_bin(horizon)
    if development:
        patient_ids = get_fold_split(
            cfg["data"]["splits_dir"], cfg["development"]["validation_partition"], "test"
        )
        stats_label = "development"
        prediction_split = "development_validation"
    else:
        patient_ids = get_fold_split(cfg["data"]["splits_dir"], args.fold, args.split)
        stats_label = f"fold_{args.fold}"
        prediction_split = f"fold{args.fold}_{args.split}"
    stats_path = Path(cfg["output"].get("norm_stats_dir", cfg["data"]["splits_dir"])) / (
        f"{stats_label}_{norm_stats_model_name}_{args.outcome}_norm_stats.json"
    )
    norm_stats = load_norm_stats(stats_path)
    output_dir = args.output_dir or Path(cfg["output"].get("prediction_dir", "results/predictions"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for landmark in landmarks:
        validate_landmark_horizons(0.0, horizons, grid.max_time)
        validate_landmark_horizons(
            landmark, horizons, float(cfg["data"]["absolute_max_time"])
        )
        dataset_options = {}
        if mcfg["architecture"] != "formal_dynamic_deephit":
            dataset_options["include_time_delta"] = True
        dataset_options["lazy_prefixes"] = bool(cfg["data"].get("lazy_prefixes", False))
        dataset_options["reject_excess_visits"] = bool(
            cfg["data"].get("reject_excess_visits", False)
        )
        dataset = DynamicDeepHitDataset(
            processed_dir=cfg["data"]["processed_dir"],
            patient_ids=patient_ids,
            static_cols=cfg["data"]["static_cols"],
            long_cols=cfg["data"]["long_cols"],
            outcome=args.outcome,
            max_visits=int(cfg["data"]["max_visits"]),
            time_grid=grid,
            norm_stats=norm_stats,
            evaluation_landmarks=[landmark],
            include_actual_queries=False,
            min_history_time=cfg["data"].get("min_history_time"),
            anchor_nearest_t0=bool(cfg["data"].get("anchor_nearest_t0", False)),
            append_missing_query_row=False,
            evaluation_history_window=float(cfg["data"].get("landmark_history_window", 0.0)),
            multimodal_embedding_path=cfg["data"].get("multimodal_embedding_path"),
            multimodal_cols=cfg["data"].get("multimodal_cols"),
            input_time_scale=cfg["data"].get("input_time_scale"),
            preserve_source_time=bool(cfg["data"].get("preserve_source_time", False)),
            actual_query_end_tolerance=float(cfg["data"].get("actual_query_end_tolerance", 0.0)),
            **dataset_options,
        )
        loader = DataLoader(
            dataset, batch_size=int(cfg["training"]["batch_size"]),
            shuffle=False, collate_fn=collate_fn,
        )
        predictions = predict_landmark(
            model, loader, landmark, horizons, grid, device,
            residual_grid=residual_grid, conditioning=conditioning,
        )
        output_path = output_dir / (
            f"{prediction_name}_{args.outcome}_{prediction_split}_"
            f"landmark{prediction_landmark_label(landmark, conditioning)}.csv"
        )
        predictions.to_csv(output_path, index=False)
        print(f"Saved {output_path} (n={len(predictions)})")
        audit = predictions.attrs["probability_roundoff"]
        print(
            "Probability roundoff projection: "
            f"adjusted_values={audit['adjusted_values']} "
            f"adjusted_rows={audit['adjusted_rows']} "
            f"max_adjustment={audit['maximum_absolute_adjustment']:.9g}"
        )


if __name__ == "__main__":
    main()
