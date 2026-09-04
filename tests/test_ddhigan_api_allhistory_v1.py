import copy
import os
import unittest

import numpy as np
try:
    from python.deployment.ddhigan_api import app as app_module
except ModuleNotFoundError:
    app_module = None
from python.deployment.ddhigan_api.preprocessing import (
    LATE_FOLLOWUP_EXTRAPOLATION_WARNING,
    late_followup_extrapolation_warning,
    prepare_model_inputs,
)
from python.deployment.ddhigan_api.schemas import PredictionRequest


def payload(visit_count=2, query_time=1.0):
    if visit_count == 1:
        times = [0.0]
    else:
        times = np.linspace(0.0, query_time, visit_count).tolist()
    return {
        "schema_version": "1.0",
        "query_time_years": query_time,
        "kidney_failure_free_at_query": True,
        "static": {
            "age_at_biopsy_years": 40.0,
            "sex": "female",
            "creatinine_mg_dl": 1.1,
            "cystatin_c_mg_l": 1.0,
            "albumin_g_l": 38.0,
            "proteinuria_g_24h": 1.5,
        },
        "visits": [
            {
                "time_years": value,
                "creatinine_mg_dl": 1.0 + index / 100.0,
                "cystatin_c_mg_l": 1.0,
                "albumin_g_l": 38.0,
                "proteinuria_g_24h": 1.5,
            }
            for index, value in enumerate(times)
        ],
    }


def stats():
    columns = (
        "age_at_biopsy", "gender", "baseline_CREA", "baseline_CystatinC",
        "baseline_ALB", "baseline_log_PRO24H", "CREA", "CystatinC", "ALB",
        "log_PRO24H",
    )
    return {column: {"mean": 0.0, "std": 1.0} for column in columns}


class V1AllHistoryContractTest(unittest.TestCase):
    def test_existing_short_request_encoding_is_byte_identical_to_legacy_layout(self):
        request = PredictionRequest.model_validate(payload(visit_count=6, query_time=5.0))
        result = prepare_model_inputs(request, stats(), 30)
        times = np.asarray([visit.time_years for visit in request.visits], dtype=np.float32)
        legacy_deltas = np.zeros(len(times), dtype=np.float32)
        legacy_deltas[1:] = np.diff(times)
        self.assertEqual(result["x_values"][0, :6, 0].tobytes(), (legacy_deltas / 10.0).tobytes())
        self.assertEqual(result["history_processing"], {
            "received_visits": 6,
            "encoded_visits": 6,
            "omitted_visits": 0,
            "rule": "most_recent_30_visits_before_or_at_query",
        })

    def test_31_visits_select_latest_30_and_preserve_boundary_gap(self):
        raw = payload(visit_count=31, query_time=4.9)
        raw["visits"][1]["time_years"] = 0.7
        for index in range(2, 31):
            raw["visits"][index]["time_years"] = 0.7 + (index - 1) * (4.2 / 29)
        raw["visits"][-1]["time_years"] = 4.9
        request = PredictionRequest.model_validate(raw)
        result = prepare_model_inputs(request, stats(), 30)
        self.assertEqual(result["history_processing"]["omitted_visits"], 1)
        self.assertAlmostEqual(float(result["x_values"][0, 0, 0]), 0.07, places=6)
        self.assertAlmostEqual(float(result["x_values"][0, 0, 7]), 1.01, places=6)

    def test_256_visits_and_late_query_are_accepted_but_257_are_rejected(self):
        accepted = PredictionRequest.model_validate(payload(visit_count=256, query_time=12.0))
        result = prepare_model_inputs(accepted, stats(), 30)
        self.assertEqual(result["history_processing"]["received_visits"], 256)
        self.assertEqual(result["history_processing"]["encoded_visits"], 30)
        self.assertEqual(result["history_processing"]["omitted_visits"], 226)
        self.assertEqual(
            late_followup_extrapolation_warning(accepted),
            LATE_FOLLOWUP_EXTRAPOLATION_WARNING,
        )
        with self.assertRaises(ValueError):
            PredictionRequest.model_validate(payload(visit_count=257, query_time=12.0))

    def test_timeline_leakage_identifier_and_date_fields_are_rejected(self):
        for extra in ({"patient_id": "prohibited"}, {"biopsy_date": "2020-01-01"}):
            raw = payload()
            raw.update(extra)
            with self.assertRaises(ValueError):
                PredictionRequest.model_validate(raw)
        raw = payload()
        raw["visits"][-1]["time_years"] = raw["query_time_years"] + 1
        with self.assertRaises(ValueError):
            PredictionRequest.model_validate(raw)
        raw = payload()
        raw["visits"][-1]["report_date"] = "2021-01-01"
        with self.assertRaises(ValueError):
            PredictionRequest.model_validate(raw)


@unittest.skipIf(app_module is None, "FastAPI is not installed in this local interpreter")
class PublicRouteContractTest(unittest.TestCase):
    class FakeRuntime:
        provenance = {"model_version": "test-v1"}
        api_release = "test-release"

        def model_info(self):
            return {"model_version": "test-v1"}

        def predict(self, request):
            return {"schema_version": request.schema_version, "model_version": "test-v1"}

    def setUp(self):
        self.previous_runtime = app_module.runtime
        self.previous_key = os.environ.get("DDHIGAN_API_KEY")
        app_module.runtime = self.FakeRuntime()
        os.environ["DDHIGAN_API_KEY"] = "test-key"

    def tearDown(self):
        app_module.runtime = self.previous_runtime
        if self.previous_key is None:
            os.environ.pop("DDHIGAN_API_KEY", None)
        else:
            os.environ["DDHIGAN_API_KEY"] = self.previous_key

    def test_only_v1_prediction_model_info_and_readiness_are_registered(self):
        public_paths = {
            route.path for route in app_module.app.routes
            if getattr(route, "include_in_schema", True)
        }
        self.assertEqual(public_paths, {
            "/health/ready", "/v1/model-info", "/ddhigan/v1/predict",
        })
        self.assertEqual(app_module.ready()["api_release"], "test-release")
        self.assertEqual(app_module.model_info()["model_version"], "test-v1")
        request = PredictionRequest.model_validate(payload())
        self.assertEqual(app_module.predict(request)["schema_version"], "1.0")

    def test_authentication_and_validation_remain_fail_closed(self):
        app_module.require_api_key("test-key")
        with self.assertRaises(Exception) as context:
            app_module.require_api_key(None)
        self.assertEqual(context.exception.status_code, 401)
        with self.assertRaises(ValueError):
            PredictionRequest.model_validate({**copy.deepcopy(payload()), "patient_id": "prohibited"})


if __name__ == "__main__":
    unittest.main()
