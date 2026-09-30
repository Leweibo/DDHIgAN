import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from deephit.data_loader import (
    DynamicDeepHitDataset,
    PatientMultiplicityPrefixBatchSampler,
    PatientUniquePrefixBatchSampler,
    collate_fn,
)
from deephit.bootstrap_refit import load_roster
from deephit.losses import dynamic_deephit_loss, negative_log_likelihood_values
from deephit.model import FormalDynamicDeepHit
from utils.data_utils import (
    compute_dynamic_norm_stats,
    get_fold_split,
    load_longitudinal_data,
    save_norm_stats,
)
from utils.reproducibility import effective_fold_seed, seed_everything, seed_worker
from utils.survival_data import split_patient_ids
from utils.time_grid import HalfYearTimeGrid
from utils.metrics import estimate_censoring_distribution


def log_stage(message: str) -> None:
    print(f"[Stage] {message}", flush=True)


def enforce_external_confirmations(cfg: dict) -> None:
    """Fail before data/GPU work when an isolated model's human gates are absent."""
    data = cfg.get("data", {})
    confirmation_path = data.get("training_confirmation_path")
    if confirmation_path:
        snapshot_path = Path(data.get("release_manifest_path", data["snapshot_validation_path"]))
        confirmation = json.loads(Path(confirmation_path).read_text(encoding="utf-8"))
        snapshot_sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        confirmed_sha = confirmation.get(
            "release_manifest_sha256", confirmation.get("snapshot_sha256")
        )
        if (
            confirmation.get("status") != "HUMAN_CONFIRMED_FOR_TRAINING"
            or confirmed_sha != snapshot_sha
        ):
            raise ValueError("snapshot human confirmation is absent, invalid, or SHA-mismatched")
    formal_path = cfg.get("training", {}).get("formal_result_confirmation_path")
    evidence_path = cfg.get("training", {}).get("formal_result_evidence_path")
    if formal_path or evidence_path:
        if not formal_path or not evidence_path:
            raise ValueError("formal refit gate requires confirmation and evidence paths")
        confirmation = json.loads(Path(formal_path).read_text(encoding="utf-8"))
        evidence_sha = hashlib.sha256(Path(evidence_path).read_bytes()).hexdigest()
        if (
            confirmation.get("status") != "FORMAL_RESULTS_CONFIRMED_FOR_REFIT"
            or confirmation.get("evidence_sha256") != evidence_sha
        ):
            raise ValueError("formal-result confirmation is absent, invalid, or SHA-mismatched")
    data = cfg.get("data", {})
    if data.get("sequence_source") == "recent_followup":
        required_static = ["age_at_query", "gender"]
        required_long = ["CREA", "CystatinC", "ALB", "log_PRO24H"]
        if (
            data.get("static_cols") != required_static
            or data.get("long_cols") != required_long
            or data.get("max_visits") != 64
            or data.get("history_window_years") not in (3, 5)
            or "max_query_time" in cfg.get("training", {})
        ):
            raise ValueError("recent-follow-up model contract is not locked")
        if not all(
            "private_ddhigan_recent" in str(path)
            for path in cfg.get("output", {}).values()
        ):
            raise ValueError("recent-follow-up patient-level training artifacts must remain private")
        validation = json.loads(Path(data["snapshot_validation_path"]).read_text(encoding="utf-8"))
        if (
            validation.get("status") != "READY_FOR_HUMAN_CONFIRMATION"
            or validation.get("window_years") != data["history_window_years"]
            or validation.get("patients") != 9948
            or validation.get("max_visits") != 64
            or validation.get("date_fields_in_model_sequence_table") is not False
            or validation.get("biopsy_fields_in_model_sequence_table") is not False
        ):
            raise ValueError("recent-follow-up snapshot contract is invalid")
        fold_manifest_path = Path(data["splits_dir"]) / "fold_manifest.json"
        fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
        if (
            fold_manifest.get("seed") != 316
            or fold_manifest.get("contracts", {}).get("test_folds_disjoint") is not True
            or fold_manifest.get("contracts", {}).get("test_union_n") != 9948
            or validation.get("fixed_fold_manifest_sha256")
            != hashlib.sha256(fold_manifest_path.read_bytes()).hexdigest()
        ):
            raise ValueError("recent-follow-up fixed split manifest is invalid or changed")
        if set(fold_manifest.get("folds", {})) != {str(fold) for fold in range(5)}:
            raise ValueError("recent-follow-up fixed split manifest has an invalid fold roster")
        for fold in range(5):
            recorded = fold_manifest["folds"][str(fold)]
            for subset in ("train", "test"):
                split_path = Path(data["splits_dir"]) / f"fold_{fold}_{subset}.csv"
                if (
                    recorded.get(f"{subset}_sha256") != hashlib.sha256(split_path.read_bytes()).hexdigest()
                    or not isinstance(recorded.get(f"{subset}_n"), int)
                ):
                    raise ValueError("recent-follow-up fixed split file differs from its frozen manifest")
        mainline_path = Path(data["physv13_mainline_manifest_path"])
        start_confirmation = json.loads(
            Path(data["exploration_start_confirmation_path"]).read_text(encoding="utf-8")
        )
        mainline = json.loads(mainline_path.read_text(encoding="utf-8"))
        if (
            mainline.get("status") != "FORMAL_RESULTS_HUMAN_CONFIRMATION_REQUIRED"
            or start_confirmation.get("status") != "PHYSV13_FORMAL_MAINLINE_COMPLETE_AND_RESOURCES_IDLE_CONFIRMED"
            or start_confirmation.get("physv13_manifest_sha256")
            != hashlib.sha256(mainline_path.read_bytes()).hexdigest()
            or start_confirmation.get("aggregate_integrity_verified") is not True
            or start_confirmation.get("resources_idle_confirmed") is not True
            or start_confirmation.get("reader_facing_promotion") is not False
        ):
            raise ValueError("recent-follow-up physv13 completion/resource gate is invalid")


def build_model(cfg: dict) -> torch.nn.Module:
    mcfg = cfg["model"]
    static_dim = len(cfg["data"]["static_cols"])
    long_dim = len(cfg["data"]["long_cols"])
    multimodal_cfg = mcfg.get("multimodal", {})
    configured_multimodal_cols = cfg["data"].get("multimodal_cols") or []
    multimodal_dim = int(multimodal_cfg.get(
        "input_dim", len(configured_multimodal_cols)
    ))
    architecture = mcfg.get("architecture")
    formal_kwargs = dict(
        value_dim=1 + static_dim + long_dim,
        missing_dim=long_dim,
        longitudinal_dim=long_dim,
        hidden_dim=int(mcfg["gru_hidden"]),
        rnn_layers=int(mcfg.get("rnn_layers", 1)),
        attention_hidden=int(mcfg["attention_hidden"]),
        event_hidden=list(mcfg["event_hidden"]),
        num_event_bins=int(mcfg["num_event_bins"]),
        dropout=float(mcfg.get("dropout", 0.2)),
        multimodal_dim=multimodal_dim,
        multimodal_hidden=multimodal_cfg.get("hidden_dim"),
        multimodal_route_count=int(multimodal_cfg.get("route_count", 0)),
        multimodal_route_embedding_dim=int(multimodal_cfg.get("route_embedding_dim", 0)),
        multimodal_route_hidden=multimodal_cfg.get("route_hidden_dim"),
        multimodal_fusion_dim=multimodal_cfg.get("fusion_dim"),
        multimodal_fusion_mode=multimodal_cfg.get("fusion_mode", "static_concat"),
        multimodal_gate_hidden=multimodal_cfg.get("gate_hidden_dim"),
        multimodal_route_dropout=float(multimodal_cfg.get("route_dropout", 0.0)),
        multimodal_all_text_dropout=float(multimodal_cfg.get("all_text_dropout", 0.0)),
        multimodal_residual_gate_bias=float(multimodal_cfg.get("residual_gate_bias", 0.0)),
    )
    if architecture == "formal_dynamic_deephit":
        return FormalDynamicDeepHit(**formal_kwargs)
    if architecture == "formal_dynamic_deephit_timedelta":
        from deephit.model import TimeDeltaDynamicDeepHit

        return TimeDeltaDynamicDeepHit(**formal_kwargs)
    if architecture in {"formal_dynamic_deephit_cde", "formal_dynamic_deephit_cde_ibs"}:
        from deephit.model import ContinuousTimeDynamicDeepHit

        if multimodal_dim:
            raise ValueError("continuous-time candidates are clinical-only")
        reference_count = sum(
            parameter.numel() for parameter in FormalDynamicDeepHit(**formal_kwargs).parameters()
        )
        hidden_choices = [int(value) for value in mcfg.get("cde_hidden_candidates", [32, 48, 64])]
        if hidden_choices != [32, 48, 64]:
            raise ValueError("CDE hidden candidates are locked to [32, 48, 64]")
        candidates = [
            ContinuousTimeDynamicDeepHit(
                value_dim=1 + static_dim + long_dim,
                missing_dim=long_dim,
                longitudinal_dim=long_dim,
                hidden_dim=hidden,
                event_hidden=list(mcfg["event_hidden"]),
                num_event_bins=int(mcfg["num_event_bins"]),
                dropout=float(mcfg.get("dropout", 0.2)),
                solver=str(mcfg.get("cde_solver", "rk4")),
                step_size=float(mcfg.get("cde_step_size", 1.0)),
            )
            for hidden in hidden_choices
        ]
        selected = min(
            candidates,
            key=lambda model: (
                abs(sum(p.numel() for p in model.parameters()) - reference_count),
                model.hidden_dim,
            ),
        )
        selected.reference_parameter_count = reference_count
        selected.parameter_count_delta = (
            sum(parameter.numel() for parameter in selected.parameters()) - reference_count
        )
        return selected
    raise ValueError(f"unsupported Dynamic-DeepHit architecture: {architecture}")


def loss_kwargs(cfg: dict) -> dict:
    mcfg = cfg["model"]
    return {
        "longitudinal_start": 1 + len(cfg["data"]["static_cols"]),
        "interval_width": float(mcfg["interval_width"]),
        "likelihood_weight": float(mcfg["loss"]["likelihood"]),
        "ranking_weight": float(mcfg["loss"]["ranking"]),
        "longitudinal_weight": float(mcfg["loss"]["longitudinal"]),
        "sigma": float(mcfg["loss"]["sigma"]),
        "ranking_pairing": str(mcfg["loss"].get("ranking_pairing", "legacy")),
        "ipcw_brier_weight": float(mcfg["loss"].get("ipcw_brier", 0.0)),
        "min_censoring_survival": float(
            mcfg["loss"].get("min_censoring_survival", 1e-6)
        ),
    }


def run_epoch(
    model, loader, device, cfg, optimizer=None, burn_in=False,
    censoring_times=None, censoring_survival=None, likelihood_only=False,
):
    training = optimizer is not None
    model.train(training)
    totals = {
        name: 0.0
        for name in ("total", "likelihood", "ranking", "longitudinal", "ipcw_brier")
    }
    non_blocking = device.type == "cuda"
    context = torch.enable_grad() if training else torch.no_grad()
    likelihood_numerator = 0.0
    likelihood_denominator = 0.0
    with context:
        for batch in loader:
            x_values = batch["x_values"].to(device, non_blocking=non_blocking)
            x_missing = batch["x_missing"].to(device, non_blocking=non_blocking)
            sequence_mask = batch["sequence_mask"].to(device, non_blocking=non_blocking)
            target_time = batch["target_time"].to(device, non_blocking=non_blocking)
            target_event = batch["target_event"].to(device, non_blocking=non_blocking)
            target_bin = batch["target_bin"].to(device, non_blocking=non_blocking)
            evaluation_time = batch["evaluation_time"].to(
                device, non_blocking=non_blocking
            )
            evaluation_event = batch["evaluation_event"].to(
                device, non_blocking=non_blocking
            )
            sample_weight = batch["sample_weight"].to(device, non_blocking=non_blocking)
            if training and bool(cfg["training"].get("original_ddh_faithful", False)):
                sample_weight = torch.ones_like(sample_weight)
            x_multimodal = batch.get("x_multimodal")
            if x_multimodal is not None:
                x_multimodal = x_multimodal.to(device, non_blocking=non_blocking)
            query_time = batch["query_time"].to(device, non_blocking=non_blocking)
            x_time = batch.get("x_time")
            x_delta = batch.get("x_delta")
            if x_time is not None:
                x_time = x_time.to(device, non_blocking=non_blocking)
                x_delta = x_delta.to(device, non_blocking=non_blocking)
            if training:
                optimizer.zero_grad(set_to_none=True)
            model_options = {
                "x_multimodal": x_multimodal,
                "query_time": query_time,
            }
            if x_time is not None:
                model_options.update(x_time=x_time, x_delta=x_delta)
            output = model(x_values, x_missing, sequence_mask, **model_options)
            if likelihood_only:
                values = negative_log_likelihood_values(
                    output["event_pmf"], output["tail_probability"], target_time,
                    target_event, target_bin, float(cfg["model"]["interval_width"]),
                )
                likelihood_numerator += float((values * sample_weight).sum())
                likelihood_denominator += float(sample_weight.sum())
                continue
            losses = dynamic_deephit_loss(
                output, x_values, x_missing, sequence_mask,
                target_time, target_event, target_bin,
                sample_weight=sample_weight,
                patient_ids=batch["patient_id"],
                censoring_times=censoring_times,
                censoring_survival=censoring_survival,
                ipcw_time=evaluation_time,
                ipcw_event=evaluation_event,
                **loss_kwargs(cfg)
            )
            objective = losses["longitudinal"] if burn_in else losses["total"]
            if training:
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            for name in totals:
                totals[name] += float(losses[name].detach())
    if likelihood_only:
        value = likelihood_numerator / max(likelihood_denominator, 1e-8)
        return {name: value if name in {"total", "likelihood"} else 0.0 for name in totals}
    denominator = max(len(loader), 1)
    return {name: value / denominator for name, value in totals.items()}


def main():
    from torch.utils.tensorboard import SummaryWriter

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="python/deephit/config_minimal_core.yaml")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--outcome", choices=["ESKD", "Drop50"], required=True)
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    enforce_external_confirmations(cfg)
    mcfg = cfg["model"]
    model_name = mcfg["name"]
    base_seed = int(cfg["training"].get("seed", 20260607))
    split_protocol = str(cfg["training"].get("split_protocol", "cross_validation"))
    development = split_protocol == "continuous_time_development"
    full_refit = split_protocol == "full_refit_fixed_epochs"
    bootstrap_refit = split_protocol == "bootstrap_refit_fixed_epochs"
    fixed_refit = full_refit or bootstrap_refit
    bootstrap_multiplicity = None
    bootstrap_summary = None
    bootstrap_replicate = None
    if fixed_refit:
        if args.fold is not None:
            raise ValueError(f"{split_protocol} forbids --fold")
        locked_epochs = int(cfg["training"].get("locked_fixed_epochs", 48))
        if locked_epochs <= 0 or int(cfg["training"].get("epochs", -1)) != locked_epochs:
            raise ValueError(
                f"{split_protocol} epochs must equal locked_fixed_epochs={locked_epochs}"
            )
        if "validation_fraction" in cfg["training"]:
            raise ValueError(f"{split_protocol} forbids validation_fraction")
        if "patience" in cfg["training"]:
            raise ValueError(f"{split_protocol} forbids patience/early stopping")
        if bootstrap_refit:
            bootstrap_replicate = int(cfg["training"].get("bootstrap_replicate", -1))
            if not 0 <= bootstrap_replicate < 1000:
                raise ValueError("bootstrap_replicate must be within 0..999")
            if int(cfg["training"].get("bootstrap_replicates", -1)) != 1000:
                raise ValueError("bootstrap_refit_fixed_epochs is locked to 1,000 replicates")
            if int(cfg["training"].get("bootstrap_seed", -1)) != 316 or base_seed != 316:
                raise ValueError("bootstrap and algorithm seeds are locked to 316")
            if float(mcfg["loss"].get("ipcw_brier", 0.0)) != 0.0:
                raise ValueError("locked bootstrap refit requires ipcw_brier=0")
        run_seed = base_seed
        run_label = (
            f"bootstrap_{bootstrap_replicate:04d}" if bootstrap_refit else "full_refit"
        )
    elif development:
        if args.fold is not None:
            raise ValueError("development candidates forbid --fold, including lockbox fold 0")
        run_seed = base_seed
        run_label = "development"
    else:
        if args.fold is None:
            raise ValueError("cross-validation training requires --fold")
        run_seed = effective_fold_seed(base_seed, args.fold)
        run_label = f"fold{args.fold}"
    deterministic = bool(cfg["training"].get("deterministic", True))
    seed_everything(run_seed, deterministic=deterministic)

    requested_device = cfg["training"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not deterministic:
        torch.backends.cudnn.benchmark = True
    print(f"[Seed] base={base_seed} run={run_label} effective={run_seed}", flush=True)
    print(f"[Device] requested={requested_device} using={device}", flush=True)
    print(f"[Deterministic] enabled={deterministic}", flush=True)

    log_stage("loading fold split")
    recent_followup = cfg["data"].get("sequence_source") == "recent_followup"
    if recent_followup:
        if args.outcome != "ESKD":
            raise ValueError("recent-follow-up branch supports rechecked ESKD only")
        from deephit.recent_followup import load_recent_followup_patient_outcomes

        patient_outcomes = load_recent_followup_patient_outcomes(cfg["data"]["processed_dir"])
        formal_outcomes = patient_outcomes.rename(columns={"eskd_status": "event_status"})
        # Event time is never used to construct a recent-follow-up tensor; it
        # exists only for the inherited patient-level split stratifier.
        formal_outcomes["event_time"] = 1.0
        long_df = None
        outcomes_df = formal_outcomes.copy()
        time_col, status_col = "event_time", "event_status"
    else:
        long_df, outcomes_df = load_longitudinal_data(cfg["data"]["processed_dir"])
        time_col = f"{args.outcome.lower()}_time"
        status_col = f"{args.outcome.lower()}_status"
        formal_outcomes = outcomes_df[["patient_id", time_col, status_col]].rename(
            columns={time_col: "event_time", status_col: "event_status"}
        )
    if fixed_refit:
        expected_patients_value = cfg["training"].get("expected_patients")
        if expected_patients_value is None and cfg["data"].get("release_manifest_path"):
            release = json.loads(Path(cfg["data"]["release_manifest_path"]).read_text(encoding="utf-8"))
            expected_patients_value = release.get("counts", {}).get("baseline_patients")
        expected_patients = int(expected_patients_value or len(formal_outcomes))
        fold_sets = []
        for fold in range(5):
            ids = get_fold_split(cfg["data"]["splits_dir"], fold, "test")
            if len(ids) != len(set(ids)):
                raise ValueError(f"fold {fold} test set contains duplicate patients")
            fold_sets.append(set(map(str, ids)))
        for left in range(5):
            for right in range(left + 1, 5):
                if fold_sets[left] & fold_sets[right]:
                    raise ValueError(f"test folds {left} and {right} overlap")
        train_ids = sorted(set().union(*fold_sets))
        outcome_ids = set(formal_outcomes["patient_id"].astype(str))
        longitudinal_ids = (
            set(formal_outcomes["patient_id"].astype(str))
            if recent_followup else set(long_df["patient_id"].astype(str))
        )
        if len(train_ids) != expected_patients:
            raise ValueError(
                f"five-fold test union has {len(train_ids)} patients; expected {expected_patients}"
            )
        if set(train_ids) != outcome_ids or set(train_ids) != longitudinal_ids:
            raise ValueError("full-refit patient union must equal outcome and longitudinal patient sets")
        if bootstrap_refit:
            roster_path = cfg["training"].get("bootstrap_roster_path")
            if not roster_path:
                raise ValueError("bootstrap_refit_fixed_epochs requires bootstrap_roster_path")
            bootstrap_multiplicity, bootstrap_summary = load_roster(
                roster_path, train_ids, bootstrap_replicate,
            )
            train_ids = sorted(bootstrap_multiplicity)
        validation_ids = []
    elif development:
        contract = cfg.get("development", {})
        train_partitions = list(contract.get("train_partitions", []))
        validation_partition = contract.get("validation_partition")
        lockbox_partition = contract.get("lockbox_partition")
        if train_partitions != [2, 3, 4] or validation_partition != 1 or lockbox_partition != 0:
            raise ValueError("continuous-time development partitions must be train=2,3,4 val=1 lockbox=0")
        train_ids = [
            patient_id
            for partition in train_partitions
            for patient_id in get_fold_split(cfg["data"]["splits_dir"], partition, "test")
        ]
        validation_ids = get_fold_split(
            cfg["data"]["splits_dir"], validation_partition, "test"
        )
        lockbox_ids = set(get_fold_split(
            cfg["data"]["splits_dir"], lockbox_partition, "test"
        ))
        if (
            len(train_ids) != len(set(train_ids))
            or set(train_ids) & set(validation_ids)
            or lockbox_ids & (set(train_ids) | set(validation_ids))
        ):
            raise ValueError("continuous-time development partitions overlap")
    else:
        outer_train_ids = get_fold_split(
            cfg["data"]["splits_dir"], args.fold, "train"
        )
        outcome_strata = (
            formal_outcomes.assign(patient_id=formal_outcomes["patient_id"].astype(str))
            .set_index("patient_id")["event_status"]
            .reindex([str(value) for value in outer_train_ids])
            .fillna(-1)
            .astype(int)
            .tolist()
        )
        train_ids, validation_ids = split_patient_ids(
            outer_train_ids,
            validation_fraction=float(cfg["training"].get("validation_fraction", 0.2)),
            seed=run_seed,
            strata=outcome_strata,
        )
    log_stage(f"loading longitudinal/outcome data train_patients={len(train_ids)} validation_patients={len(validation_ids)} stratified_by={status_col}")
    if recent_followup:
        from deephit.recent_followup import compute_recent_followup_norm_stats

        log_stage("computing recent-follow-up normalization stats from date-free private sequences")
        norm_stats = compute_recent_followup_norm_stats(
            cfg["data"]["processed_dir"], train_ids, cfg["data"]["static_cols"],
            cfg["data"]["long_cols"], patient_multiplicity=bootstrap_multiplicity,
        )
    else:
        log_stage(f"computing normalization stats long_rows={len(long_df)} outcome_rows={len(outcomes_df)}")
        norm_stats = compute_dynamic_norm_stats(
            long_df, formal_outcomes, cfg["data"]["static_cols"],
            cfg["data"]["long_cols"], train_ids,
            min_visit_time=cfg["data"].get("min_history_time"),
            anchor_nearest_t0=bool(cfg["data"].get("anchor_nearest_t0", False)),
            max_visit_time=cfg["training"].get("max_query_time"),
            patient_multiplicity=bootstrap_multiplicity,
        )
    stats_label = (
        run_label if fixed_refit else "development" if development else f"fold_{args.fold}"
    )
    stats_path = Path(cfg["output"].get("norm_stats_dir", cfg["data"]["splits_dir"])) / (
        f"{stats_label}_{model_name}_{args.outcome}_norm_stats.json"
    )
    save_norm_stats(norm_stats, stats_path)

    grid = HalfYearTimeGrid(
        max_time=float(mcfg["max_residual_time"]),
        interval_width=float(mcfg["interval_width"]),
    )
    if grid.num_event_bins != int(mcfg["num_event_bins"]):
        raise ValueError("num_event_bins must equal max_residual_time / interval_width")
    if recent_followup:
        from deephit.recent_followup import RecentFollowupDataset

        dataset_class = RecentFollowupDataset
        dataset_kwargs = {
            "processed_dir": cfg["data"]["processed_dir"],
            "static_cols": cfg["data"]["static_cols"],
            "long_cols": cfg["data"]["long_cols"],
            "max_visits": int(cfg["data"]["max_visits"]),
            "time_grid": grid, "norm_stats": norm_stats,
        }
    else:
        dataset_class = DynamicDeepHitDataset
        dataset_kwargs = {
            "processed_dir": cfg["data"]["processed_dir"],
            "static_cols": cfg["data"]["static_cols"],
            "long_cols": cfg["data"]["long_cols"],
            "outcome": args.outcome,
            "max_visits": int(cfg["data"]["max_visits"]),
            "time_grid": grid,
            "norm_stats": norm_stats,
            "evaluation_landmarks": cfg["training"].get(
                "training_landmarks", cfg["data"].get("landmarks", [])
            ),
            "include_actual_queries": bool(cfg["training"].get("include_actual_queries", True)),
            "max_query_time": cfg["training"].get("max_query_time"),
            "input_time_scale": cfg["data"].get("input_time_scale"),
            "preserve_source_time": bool(cfg["data"].get("preserve_source_time", False)),
            "actual_query_end_tolerance": float(cfg["data"].get("actual_query_end_tolerance", 0.0)),
            "min_history_time": cfg["data"].get("min_history_time"),
            "anchor_nearest_t0": bool(cfg["data"].get("anchor_nearest_t0", False)),
            "multimodal_embedding_path": cfg["data"].get("multimodal_embedding_path"),
            "multimodal_cols": cfg["data"].get("multimodal_cols"),
            "include_time_delta": mcfg["architecture"] != "formal_dynamic_deephit",
            "lazy_prefixes": bool(cfg["data"].get("lazy_prefixes", False)),
            "reject_excess_visits": bool(cfg["data"].get("reject_excess_visits", False)),
        }
    log_stage("building training prefixes")
    train_ds = dataset_class(patient_ids=train_ids, **dataset_kwargs)
    if fixed_refit:
        val_ds = None
        log_stage(f"{run_label} training prefixes={len(train_ds)} validation=disabled")
    else:
        log_stage(f"building validation prefixes train_prefixes={len(train_ds)}")
        val_ds = dataset_class(patient_ids=validation_ids, **dataset_kwargs)
    query_roster_audit = {}
    query_preflight = cfg["data"].get("query_roster_preflight_path")
    if query_preflight:
        from utils.query_roster import query_roster_sha256
        expected = json.loads(Path(query_preflight).read_text())["splits"][args.fold]
        for label, dataset in [("train", train_ds), ("validation", val_ds)]:
            if dataset is None:
                raise ValueError("query roster gate requires the locked internal validation split")
            key = label + "_query_roster_sha256"
            actual = query_roster_sha256(dataset.patient_ids, dataset.query_time.tolist())
            if actual != expected[key] or len(dataset) != expected[label + "_prefixes"]:
                raise ValueError("actual dataset query roster differs from preflight before fitting")
            query_roster_audit[key] = actual
        log_stage("training and validation query rosters match preflight")
    censoring_times = None
    censoring_survival = None
    if float(mcfg["loss"].get("ipcw_brier", 0.0)):
        train_time = train_ds.evaluation_time.numpy().astype(float)
        train_event = train_ds.evaluation_event.numpy().astype(int)
        if not train_event.any():
            raise ValueError("IPCW Brier training set is entirely censored")
        censoring = estimate_censoring_distribution(train_time, train_event)
        censoring_times = torch.as_tensor(
            censoring.times, dtype=torch.float32, device=device
        )
        censoring_survival = torch.as_tensor(
            censoring.survival, dtype=torch.float32, device=device
        )
    log_stage(
        "constructing dataloaders "
        + ("validation=disabled" if fixed_refit else f"validation_prefixes={len(val_ds)}")
    )
    generator = torch.Generator().manual_seed(run_seed)
    num_workers = int(cfg["training"].get("num_workers", 0))
    loader_kwargs = {
        "collate_fn": collate_fn,
        "worker_init_fn": seed_worker,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            cfg["training"].get("persistent_workers", True)
        )
        loader_kwargs["prefetch_factor"] = int(cfg["training"].get("prefetch_factor", 4))
    if bool(cfg["training"].get("original_ddh_faithful", False)):
        batch_size = int(cfg["training"]["batch_size"])
        batch_sampler = (
            PatientMultiplicityPrefixBatchSampler(
                train_ds.patient_ids, train_ds.query_time, bootstrap_multiplicity,
                batch_size, run_seed,
            )
            if bootstrap_refit else
            PatientUniquePrefixBatchSampler(
                train_ds.patient_ids, train_ds.query_time, batch_size, run_seed,
                selection="cycle",
            )
        )
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, **loader_kwargs)
        val_loader = None if fixed_refit else DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs,
        )
        sampler_mode = (
            "bootstrap_multiplicity_patient_unique_layers"
            if bootstrap_refit else
            "patient_unique_cycle/all_prefix_patient_weighted_validation"
        )
    else:
        loader_kwargs["batch_size"] = int(cfg["training"]["batch_size"])
        train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs)
        val_loader = None if fixed_refit else DataLoader(
            val_ds, shuffle=False, **loader_kwargs
        )
        sampler_mode = "all_prefixes"
    print(
        "[DataLoader] "
        f"batch_size={cfg['training']['batch_size']} sampler={sampler_mode} num_workers={num_workers} "
        f"pin_memory={loader_kwargs['pin_memory']} "
        f"persistent_workers={loader_kwargs.get('persistent_workers', False)} "
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'none')}",
        flush=True,
    )

    log_stage("building model")
    model = build_model(cfg).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_{args.outcome}_{run_label}.pt"
    if fixed_refit and checkpoint_path.exists():
        raise FileExistsError(
            f"refusing to overwrite or resume fixed full-refit checkpoint: {checkpoint_path}"
        )
    tensorboard_root = Path(
        cfg["output"].get("tensorboard_dir", checkpoint_dir.parent / "tensorboard")
    )
    log_dir = tensorboard_root / f"{model_name}_{run_label}_{args.outcome}"
    writer = SummaryWriter(log_dir=str(log_dir))

    log_stage(f"starting epochs parameters={parameter_count}")
    checkpoint_metric = str(cfg["training"].get("checkpoint_metric", "total"))
    if checkpoint_metric not in {"total", "likelihood"}:
        raise ValueError("training.checkpoint_metric must be total or likelihood")
    best_val = float("inf")
    best_epoch = None
    patience = 0
    burn_in_epochs = int(cfg["training"].get("burn_in_epochs", 0))
    started_at = time.time()
    stopped_after_epoch = 0
    for epoch in range(int(cfg["training"]["epochs"])):
        burn_in = epoch < burn_in_epochs
        train_metrics = run_epoch(
            model, train_loader, device, cfg, optimizer, burn_in,
            censoring_times, censoring_survival,
        )
        if fixed_refit:
            val_metrics = None
            if epoch + 1 == int(cfg["training"]["epochs"]):
                best_epoch = epoch + 1
                best_val = None
                # The fixed-epoch checkpoint intentionally contains inference
                # weights only: no optimizer state and no validation selector.
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "base_seed": base_seed,
                    "effective_seed": run_seed,
                    "parameter_count": parameter_count,
                    "norm_stats_path": str(stats_path),
                    "split_protocol": split_protocol,
                    "fixed_epochs": int(cfg["training"]["epochs"]),
                    "bootstrap_summary": bootstrap_summary,
                }, checkpoint_path)
        else:
            val_metrics = run_epoch(
                model, val_loader, device, cfg,
                censoring_times=censoring_times,
                censoring_survival=censoring_survival,
                likelihood_only=True,
            )
            val_objective = val_metrics[checkpoint_metric]
            if burn_in:
                patience = 0
            elif val_objective < best_val:
                best_val = val_objective
                best_epoch = epoch + 1
                patience = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val,
                    "checkpoint_metric": checkpoint_metric,
                    "config": cfg,
                    "base_seed": base_seed,
                    "effective_seed": run_seed,
                    "parameter_count": parameter_count,
                    "norm_stats_path": str(stats_path),
                    "split_protocol": split_protocol,
                    "censoring_times": censoring_times,
                    "censoring_survival": censoring_survival,
                }, checkpoint_path)
            elif not burn_in:
                patience += 1
        metric_sets = [("train", train_metrics)]
        if val_metrics is not None:
            metric_sets.append(("val", val_metrics))
        for split, metrics in metric_sets:
            for name, value in metrics.items():
                writer.add_scalar(f"Loss/{split}_{name}", value, epoch + 1)
        stopped_after_epoch = epoch + 1
        if stopped_after_epoch == 1 or stopped_after_epoch % 10 == 0:
            print(
                f"Epoch {stopped_after_epoch}: train={train_metrics['total']:.5f} "
                + (
                    f"fixed_epoch_target={locked_epochs} burn_in={burn_in}"
                    if fixed_refit else f"val={val_metrics['total']:.5f} burn_in={burn_in}"
                )
            )
        if (
            not fixed_refit and not burn_in
            and patience >= int(cfg["training"]["patience"])
        ):
            print(f"Early stopping at epoch {stopped_after_epoch}")
            break

    writer.close()
    if best_epoch is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint")
    summary_dir = Path(
        cfg["output"].get("training_dir", checkpoint_dir.parent / "training")
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{model_name}_{args.outcome}_{run_label}.json"
    with open(summary_path, "w") as handle:
        json.dump({
            **query_roster_audit,
            "model_name": model_name,
            "architecture": mcfg["architecture"],
            "outcome": args.outcome,
            "fold": args.fold,
            "split_protocol": split_protocol,
            "effective_seed": run_seed,
            "parameter_count": parameter_count,
            "train_patients": len(train_ids),
            "train_patient_set_sha256": hashlib.sha256("\n".join(sorted(map(str, train_ids))).encode()).hexdigest(),
            "validation_patient_set_sha256": hashlib.sha256("\n".join(sorted(map(str, validation_ids))).encode()).hexdigest(),
            "bootstrap_patient_slots": (
                None if bootstrap_summary is None else bootstrap_summary["slots"]
            ),
            "bootstrap_summary": bootstrap_summary,
            "validation_patients": len(validation_ids),
            "train_prefixes": len(train_ds),
            "validation_prefixes": 0 if val_ds is None else len(val_ds),
            "validation_selection": "none" if fixed_refit else checkpoint_metric,
            "early_stopping": False if fixed_refit else True,
            "best_epoch": best_epoch,
            "stopped_after_epoch": stopped_after_epoch,
            "checkpoint_metric": checkpoint_metric,
            "best_val_loss": best_val,
            "elapsed_seconds": time.time() - started_at,
            "checkpoint": str(checkpoint_path),
            "norm_stats": str(stats_path),
            "config_path": args.config,
        }, handle, indent=2)
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training summary: {summary_path}")


if __name__ == "__main__":
    main()
