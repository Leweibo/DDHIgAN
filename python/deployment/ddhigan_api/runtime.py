from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

from python.deephit.model import FormalDynamicDeepHit
from python.utils.time_grid import HalfYearTimeGrid

from .calibration import calibration_uncertainty_interval, recalibrate
from .preprocessing import (
    distribution_warnings,
    late_followup_extrapolation_warning,
    prepare_model_inputs,
)


def build_inference_model(config: dict):
    model = config["model"]
    data = config["data"]
    if model.get("architecture") != "formal_dynamic_deephit" or model.get("multimodal"):
        raise ValueError("deployment supports only the locked clinical FormalDynamicDeepHit")
    return FormalDynamicDeepHit(
        value_dim=1 + len(data["static_cols"]) + len(data["long_cols"]),
        missing_dim=len(data["long_cols"]), longitudinal_dim=len(data["long_cols"]),
        hidden_dim=int(model["gru_hidden"]), rnn_layers=int(model.get("rnn_layers", 1)),
        attention_hidden=int(model["attention_hidden"]), event_hidden=list(model["event_hidden"]),
        num_event_bins=int(model["num_event_bins"]), dropout=float(model.get("dropout", .2)),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(bundle: Path) -> dict[str, str]:
    expected = {}
    for line in (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    required = {
        "model_weights.pt", "normalization.json", "calibrator.json",
        "model_config.json", "reference_ranges.json", "input_contract.json",
        "provenance.json", "software_versions.json",
    }
    ensemble = {
        "ensemble_weights.pt", "ensemble_normalizations.json",
        "ensemble_calibrators.json", "ensemble_manifest.json",
    }
    if set(expected) not in (required, required | ensemble):
        raise ValueError("deployment SHA manifest file set mismatch")
    for name, digest in expected.items():
        if file_sha256(bundle / name) != digest:
            raise ValueError(f"deployment SHA mismatch: {name}")
    return expected


class DDHIgANRuntime:
    def __init__(self, bundle: str | Path):
        import torch

        self.torch = torch
        torch.set_num_threads(int(os.environ.get("DDHIGAN_TORCH_THREADS", "1")))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self._inference_lock = threading.Lock()
        self.bundle = Path(bundle)
        self.api_release = self.bundle.resolve().parent.name
        self.hashes = verify_bundle(self.bundle)
        self.config = json.loads((self.bundle / "model_config.json").read_text())
        self.norm = json.loads((self.bundle / "normalization.json").read_text())
        self.calibrator = json.loads((self.bundle / "calibrator.json").read_text())
        self.predictive_manifest = None
        self.ensemble_states = None
        self.ensemble_norms = None
        self.ensemble_calibrators = None
        uncertainty = self.calibrator.get("uncertainty", {})
        draws = uncertainty.get("draws", [])
        optimizer = uncertainty.get("optimizer", {})
        calibration_uncertainty_valid = (
            uncertainty.get("status") != "COMPLETE"
            or uncertainty.get("uncertainty_version") != "provisional_pooled_oof_calibration_bootstrap_v1"
            or uncertainty.get("scope") != "calibration_estimation_only_conditioned_on_frozen_raw_risk"
            or uncertainty.get("iterations") != 2000
            or uncertainty.get("seed") != 316
            or uncertainty.get("patients") != 9948
            or uncertainty.get("patient_equal") is not True
            or uncertainty.get("resampling_unit") != "patient"
            or uncertainty.get("full_refit_predictions_used") is not False
            or uncertainty.get("ddhigan_models_retrained") != 0
            or uncertainty.get("quantiles") != [0.025, 0.975]
            or optimizer.get("name") != "L-BFGS-B"
            or optimizer.get("parameterization") != "bounded_direct_alpha_beta"
            or optimizer.get("constant_weight_normalization") is not True
            or optimizer.get("max_line_search_steps") != 100
            or optimizer.get("objective_changed") is not False
            or len(draws) != 2000
            or [row.get("iteration") for row in draws] != list(range(2000))
            or not all(
                np.isfinite([row.get("alpha"), row.get("beta")]).all() and row["beta"] > 0
                for row in draws
            )
        ) is False
        if "ensemble_manifest.json" not in self.hashes and not calibration_uncertainty_valid:
            raise ValueError("approved calibration uncertainty artifact is missing or invalid")
        self.ranges = json.loads((self.bundle / "reference_ranges.json").read_text())
        self.provenance = json.loads((self.bundle / "provenance.json").read_text())
        self.model = build_inference_model(self.config).cpu()
        state = torch.load(self.bundle / "model_weights.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.point_state = {key: value.detach().clone() for key, value in state.items()}
        self.model.eval()
        self.grid = HalfYearTimeGrid(
            float(self.config["model"]["max_residual_time"]),
            float(self.config["model"]["interval_width"]),
        )
        if "ensemble_manifest.json" in self.hashes:
            self.predictive_manifest = json.loads((self.bundle / "ensemble_manifest.json").read_text())
            weights = torch.load(self.bundle / "ensemble_weights.pt", map_location="cpu", weights_only=True)
            norms = json.loads((self.bundle / "ensemble_normalizations.json").read_text())
            calibrators = json.loads((self.bundle / "ensemble_calibrators.json").read_text())
            deployment_contract = self.config.get("deployment", {})
            uncertainty_contract = deployment_contract.get("predictive_uncertainty", {})
            expected_uncertainty_version = uncertainty_contract.get(
                "version", "ddhigan_bootstrap_refit_predictive_uncertainty_v1"
            )
            expected_refit_epochs = int(deployment_contract.get("refit_epochs", 48))
            if (
                self.predictive_manifest.get("status") != "COMPLETE"
                or self.predictive_manifest.get("version") != expected_uncertainty_version
                or self.predictive_manifest.get("quantiles") != [0.025, 0.975]
                or self.predictive_manifest.get("replicates") != 1000
                or self.predictive_manifest.get("patients") != 9948
                or self.predictive_manifest.get("bootstrap_seed") != 316
                or self.predictive_manifest.get("algorithm_seed") != 316
                or self.predictive_manifest.get("epochs") != expected_refit_epochs
                or self.predictive_manifest.get("patient_level_artifacts_in_bundle") is not False
                or weights.get("format") != "ddhigan_bootstrap_refit_state_dicts_v1"
                or weights.get("replicates") != 1000
                or len(weights.get("state_dicts", [])) != 1000
                or norms.get("format") != "ddhigan_bootstrap_refit_normalizations_v1"
                or calibrators.get("format") != "ddhigan_bootstrap_refit_oob_calibrators_v1"
                or [row.get("replicate") for row in norms.get("replicates", [])] != list(range(1000))
                or [row.get("replicate") for row in calibrators.get("replicates", [])] != list(range(1000))
            ):
                raise ValueError("predictive uncertainty ensemble is missing or invalid")
            self.ensemble_states = weights["state_dicts"]
            self.ensemble_norms = [row["stats"] for row in norms["replicates"]]
            self.ensemble_calibrators = calibrators["replicates"]

    def _raw_yearly(self, prepared: dict) -> np.ndarray:
        with self.torch.inference_mode():
            output = self.model(
                self.torch.from_numpy(prepared["x_values"]),
                self.torch.from_numpy(prepared["x_missing"]),
                self.torch.from_numpy(prepared["sequence_mask"]),
                query_time=self.torch.from_numpy(prepared["query_time"]),
            )
        raw = output["cif"][0].cpu().numpy()
        return np.asarray([[raw[int(self.grid.horizon_bin(year))] for year in range(1, 11)]])

    def predict(self, request) -> dict:
        if request.schema_version == "2.0":
            if len(request.visits) > 256:
                raise ValueError("visit count exceeds the 256-visit v2 contract")
            supported = self.config.get("deployment", {}).get("max_supported_query_time")
            if supported is None:
                supported = self.provenance.get("max_supported_query_time")
            if supported is None:
                raise ValueError("v2 bundle is missing its frozen query-time support limit")
            if request.query_time_years > float(supported) + 1e-8:
                raise ValueError("query time exceeds the frozen training support range")
        if request.schema_version == "3.0":
            deployment = self.config.get("deployment", {})
            window = deployment.get("history_window_years")
            if window not in (3, 5) or deployment.get("schema_version") != "3.0":
                raise ValueError("v3 bundle does not declare a locked recent-follow-up window")
            if len(request.visits) > 64 or int(self.config["data"].get("max_visits", -1)) != 64:
                raise ValueError("v3 bundle/request violates the 64-visit recent-follow-up contract")
            if sum(visit.intervisit_gap_years for visit in request.visits) > float(window) + 1e-8:
                raise ValueError("recent visit history exceeds this bundle's frozen window")
        with self._inference_lock:
            prepared = prepare_model_inputs(request, self.norm, int(self.config["data"]["max_visits"]))
            self.model.load_state_dict(self.point_state, strict=True)
            yearly = self._raw_yearly(prepared)
            calibrated = recalibrate(
                yearly, float(self.calibrator["alpha"]), float(self.calibrator["beta"])
            )[0]
            if self.predictive_manifest is not None:
                sampled = np.empty((1000, 10), dtype=float)
                for index, (state, norm, calibrator) in enumerate(zip(
                    self.ensemble_states, self.ensemble_norms, self.ensemble_calibrators,
                )):
                    self.model.load_state_dict(state, strict=True)
                    replicate_input = prepare_model_inputs(
                        request, norm, int(self.config["data"]["max_visits"]),
                    )
                    replicate_raw = self._raw_yearly(replicate_input)
                    sampled[index] = recalibrate(
                        replicate_raw, float(calibrator["alpha"]), float(calibrator["beta"]),
                    )[0]
                self.model.load_state_dict(self.point_state, strict=True)
                lower, upper = np.quantile(sampled, self.predictive_manifest["quantiles"], axis=0)
                if (
                    not np.isfinite(sampled).all() or np.any((sampled < 0) | (sampled > 1))
                    or np.any(np.diff(sampled, axis=1) < -1e-7)
                    or np.any(lower > upper) or np.any(np.diff(lower) < -1e-7)
                    or np.any(np.diff(upper) < -1e-7)
                ):
                    raise ValueError("predictive uncertainty interval is invalid")
                interval_fields = {
                    "predictive_uncertainty_lower": lower,
                    "predictive_uncertainty_upper": upper,
                }
            else:
                uncertainty = self.calibrator["uncertainty"]
                quantiles = tuple(float(value) for value in uncertainty["quantiles"])
                lower, upper = calibration_uncertainty_interval(yearly, uncertainty["draws"], quantiles)
                interval_fields = {
                    "calibration_uncertainty_lower": lower[0],
                    "calibration_uncertainty_upper": upper[0],
                }
        ood = distribution_warnings(request, self.ranges)
        late_warning = late_followup_extrapolation_warning(request)
        risks = {str(year): float(calibrated[year - 1]) for year in range(1, 11)}
        result = {
            "schema_version": request.schema_version,
            "future_risk_curve": [{
                "years": year, "risk": risks[str(year)],
                **{name: float(values[year - 1]) for name, values in interval_fields.items()},
            } for year in range(1, 11)],
            "risk_3y": risks["3"], "risk_5y": risks["5"], "risk_10y": risks["10"],
            "warnings": {
                "missing": prepared["missing_fields"], "distribution": ood,
                "out_of_distribution": bool(ood) or late_warning is not None,
                "late_followup_extrapolation": late_warning,
            },
            "history_processing": prepared["history_processing"],
            "model_version": self.provenance["model_version"],
            "api_release": self.api_release,
            "calibrator_version": self.calibrator["calibrator_version"],
            "weights_sha256": self.hashes["model_weights.pt"],
        }
        if self.predictive_manifest is not None:
            result["predictive_uncertainty"] = {
                "version": self.predictive_manifest["version"], "level": 0.95,
                "scope": self.predictive_manifest["scope"],
                "display_label": self.predictive_manifest["display_label"],
                "replicates": 1000,
            }
        else:
            uncertainty = self.calibrator["uncertainty"]
            result["calibration_uncertainty"] = {
                "version": uncertainty["uncertainty_version"],
                "level": float(uncertainty["confidence_level"]),
                "scope": uncertainty["scope"],
                "display_label": uncertainty["display_label"],
            }
        if request.schema_version == "2.0":
            result["history_complete_since_t0"] = True
            if request.query_time_years >= 10:
                result["warnings"]["late_extrapolation"] = (
                    "晚期未来10年风险为外推、当前队列未充分验证"
                )
        if request.schema_version == "3.0":
            result["recent_query_anchor"] = "last supplied valid follow-up visit"
            result["recent_history_window_years"] = int(
                self.config["deployment"]["history_window_years"]
            )
        return result

    def model_info(self) -> dict:
        contract = json.loads((self.bundle / "input_contract.json").read_text())
        return {
            "model_version": self.provenance["model_version"],
            "api_release": self.api_release,
            "calibrator_version": self.calibrator["calibrator_version"],
            "predictive_uncertainty_version": None if self.predictive_manifest is None else self.predictive_manifest["version"],
            "predictive_uncertainty_scope": None if self.predictive_manifest is None else self.predictive_manifest["scope"],
            "weights_sha256": self.hashes["model_weights.pt"],
            "contract": contract,
            "request_contract": {
                "schema_version": "1.0",
                "accepted_visits": {"minimum": 1, "maximum": 256},
                "encoded_visits": {"maximum": 30, "selection": "most_recent_30_visits_before_or_at_query"},
                "query_time_years": {"minimum": 0, "maximum": None, "finite": True},
                "internally_validated_query_time_years": [0, 5],
                "late_followup_use": "research extrapolation beyond internal validation",
                "identifier_fields_prohibited": True,
                "dates_prohibited": True,
            },
            "intended_use": "research clinical pilot; not a substitute for clinical judgment",
        }
