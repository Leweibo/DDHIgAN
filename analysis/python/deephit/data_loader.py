import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_utils import (
    load_longitudinal_data, build_dynamic_prefix_inputs,
)
from utils.time_grid import HalfYearTimeGrid


def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == "patient_id":
            out[k] = [b[k] for b in batch]
        else:
            values = [b[k] for b in batch]
            if k in {"x_values", "x_missing", "sequence_mask", "x_time", "x_delta"}:
                max_length = max(value.shape[0] for value in values)
                padded = []
                for value in values:
                    shape = (max_length, *value.shape[1:])
                    target = value.new_zeros(shape)
                    target[:value.shape[0]] = value
                    padded.append(target)
                out[k] = torch.stack(padded)
            else:
                out[k] = torch.stack(values)
    return out


class PatientUniquePrefixBatchSampler(Sampler[list[int]]):
    """Yield one fixed-landmark prefix per patient in every training epoch.

    Dynamic-DeepHit trains one longitudinal history per subject.  For this
    project's four required landmark predictions, ``cycle`` selects exactly one
    available landmark prefix per patient and rotates it deterministically over
    epochs.  Thus a minibatch never contains two histories from the same
    patient, while every available landmark is repeatedly represented.
    """

    def __init__(
        self,
        patient_ids: list[str],
        query_times: torch.Tensor,
        batch_size: int,
        seed: int,
        selection: str = "cycle",
    ) -> None:
        if batch_size < 2:
            raise ValueError("patient-unique batch size must be at least 2")
        if selection not in {"cycle", "latest"}:
            raise ValueError(f"unsupported prefix selection: {selection}")
        groups: dict[str, list[int]] = {}
        for index, patient_id in enumerate(patient_ids):
            groups.setdefault(str(patient_id), []).append(index)
        self.groups = {
            patient_id: tuple(sorted(
                indices, key=lambda index: (float(query_times[index]), index)
            ))
            for patient_id, indices in groups.items()
        }
        self.patient_order = tuple(self.groups)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.selection = selection
        self.epoch = 0

    def __len__(self) -> int:
        return (len(self.patient_order) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        selected = []
        for patient_offset, patient_id in enumerate(self.patient_order):
            indices = self.groups[patient_id]
            if self.selection == "latest":
                selected.append(indices[-1])
            else:
                selected.append(indices[(patient_offset + epoch) % len(indices)])
        generator = torch.Generator().manual_seed(self.seed + epoch)
        order = torch.randperm(len(selected), generator=generator).tolist()
        for start in range(0, len(order), self.batch_size):
            yield [selected[index] for index in order[start:start + self.batch_size]]


class PatientMultiplicityPrefixBatchSampler(Sampler[list[int]]):
    """Expose one prefix per bootstrap patient slot in every epoch.

    A patient with multiplicity m appears exactly m times per epoch. Copies
    are placed in separate layers so a minibatch never contains the same
    biological patient twice. Prefix selection and shuffling are deterministic
    for a fixed algorithm seed.
    """

    def __init__(
        self,
        patient_ids: list[str],
        query_times: torch.Tensor,
        patient_multiplicity: dict[str, int],
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size < 2:
            raise ValueError("patient-multiplicity batch size must be at least 2")
        groups: dict[str, list[int]] = {}
        for index, patient_id in enumerate(patient_ids):
            groups.setdefault(str(patient_id), []).append(index)
        multiplicity = {str(key): int(value) for key, value in patient_multiplicity.items()}
        if set(groups) != set(multiplicity):
            raise ValueError("dataset patients must equal positive-multiplicity patients")
        if not multiplicity or any(value <= 0 for value in multiplicity.values()):
            raise ValueError("bootstrap multiplicities must be positive integers")
        self.groups = {
            patient_id: tuple(sorted(
                indices, key=lambda index: (float(query_times[index]), index)
            ))
            for patient_id, indices in groups.items()
        }
        self.patient_order = tuple(self.groups)
        self.multiplicity = multiplicity
        self.total_slots = sum(multiplicity.values())
        self.max_multiplicity = max(multiplicity.values())
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return sum(
            (sum(value > layer for value in self.multiplicity.values()) + self.batch_size - 1)
            // self.batch_size
            for layer in range(self.max_multiplicity)
        )

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        for layer in range(self.max_multiplicity):
            selected = []
            for patient_offset, patient_id in enumerate(self.patient_order):
                if self.multiplicity[patient_id] <= layer:
                    continue
                indices = self.groups[patient_id]
                selected.append(indices[(patient_offset + epoch + layer) % len(indices)])
            generator = torch.Generator().manual_seed(
                self.seed + epoch * (self.max_multiplicity + 1) + layer
            )
            order = torch.randperm(len(selected), generator=generator).tolist()
            for start in range(0, len(order), self.batch_size):
                yield [selected[index] for index in order[start:start + self.batch_size]]


class DynamicDeepHitDataset(Dataset):
    """Patient-balanced landmark/visit-prefix dataset for Dynamic-DeepHit."""

    def __init__(
        self,
        processed_dir: str,
        patient_ids: list[str],
        static_cols: list[str],
        long_cols: list[str],
        outcome: str,
        max_visits: int,
        time_grid: HalfYearTimeGrid,
        norm_stats: dict,
        evaluation_landmarks: list[float] | None = None,
        include_actual_queries: bool = True,
        max_query_time: float | None = None,
        min_history_time: float | None = None,
        anchor_nearest_t0: bool = False,
        append_missing_query_row: bool = True,
        evaluation_history_window: float = 0.0,
        multimodal_embedding_path: str | None = None,
        multimodal_cols: list[str] | None = None,
        include_time_delta: bool = False,
        lazy_prefixes: bool = False,
        reject_excess_visits: bool = False,
        evaluation_landmarks_by_patient: dict[str, list[float]] | None = None,
        input_time_scale: float | None = None,
        preserve_source_time: bool = False,
        actual_query_end_tolerance: float = 0.0,
    ) -> None:
        long_df, outcomes_df = load_longitudinal_data(processed_dir)
        time_col = f"{outcome.lower()}_time"
        status_col = f"{outcome.lower()}_status"
        outcome = outcomes_df[["patient_id", time_col, status_col]].rename(
            columns={time_col: "event_time", status_col: "event_status"}
        )
        data = build_dynamic_prefix_inputs(
            long_df,
            outcome,
            patient_ids,
            static_cols=static_cols,
            long_cols=long_cols,
            max_visits=max_visits,
            time_grid=time_grid,
            evaluation_landmarks=evaluation_landmarks,
            norm_stats=norm_stats,
            include_actual_queries=include_actual_queries,
            max_query_time=max_query_time,
            min_history_time=min_history_time,
            anchor_nearest_t0=anchor_nearest_t0,
            append_missing_query_row=append_missing_query_row,
            evaluation_history_window=evaluation_history_window,
            include_time_delta=include_time_delta,
            ragged=lazy_prefixes,
            reject_excess_visits=reject_excess_visits,
            evaluation_landmarks_by_patient=evaluation_landmarks_by_patient,
            input_time_scale=input_time_scale,
            preserve_source_time=preserve_source_time,
            actual_query_end_tolerance=actual_query_end_tolerance,
        )
        tensorize = lambda values: (
            [torch.from_numpy(value) for value in values]
            if isinstance(values, list) else torch.from_numpy(values)
        )
        self.x_values = tensorize(data["x_values"])
        self.x_missing = tensorize(data["x_missing"])
        self.sequence_mask = tensorize(data["sequence_mask"])
        self.target_time = torch.from_numpy(data["target_time"])
        self.target_event = torch.from_numpy(data["target_event"])
        self.target_bin = torch.from_numpy(data["target_bin"])
        self.evaluation_time = torch.from_numpy(data["evaluation_time"])
        self.evaluation_event = torch.from_numpy(data["evaluation_event"])
        self.patient_ids = data["patient_id"]
        self.query_time = torch.from_numpy(data["query_time"])
        self.last_observation_time = torch.from_numpy(data["last_observation_time"])
        self.sample_weight = torch.from_numpy(data["sample_weight"])
        self.x_time = tensorize(data["x_time"]) if include_time_delta else None
        self.x_delta = tensorize(data["x_delta"]) if include_time_delta else None
        self.x_multimodal = None
        if multimodal_embedding_path is not None:
            import pandas as pd

            embeddings = pd.read_csv(
                multimodal_embedding_path,
                dtype={"patient_id": str},
                low_memory=False,
            )
            if "patient_id" not in embeddings.columns:
                raise ValueError("multimodal embedding table must include patient_id")
            if embeddings["patient_id"].astype(str).duplicated().any():
                raise ValueError("multimodal embedding table contains duplicate patient_id")
            feature_cols = multimodal_cols or [
                column for column in embeddings.columns if column != "patient_id"
            ]
            missing = sorted(set(feature_cols).difference(embeddings.columns))
            if missing:
                raise ValueError(f"missing multimodal columns: {missing}")
            embedding_index = embeddings.set_index(embeddings["patient_id"].astype(str))
            values = []
            for patient_id in self.patient_ids:
                pid = str(patient_id)
                if pid in embedding_index.index:
                    row = embedding_index.loc[pid, feature_cols].to_numpy(dtype="float32")
                else:
                    row = np.zeros(len(feature_cols), dtype=np.float32)
                values.append(row)
            self.x_multimodal = torch.from_numpy(np.asarray(values, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict:
        result = {
            "x_values": self.x_values[index],
            "x_missing": self.x_missing[index],
            "sequence_mask": self.sequence_mask[index],
            "target_time": self.target_time[index],
            "target_event": self.target_event[index],
            "target_bin": self.target_bin[index],
            "evaluation_time": self.evaluation_time[index],
            "evaluation_event": self.evaluation_event[index],
            "patient_id": self.patient_ids[index],
            "query_time": self.query_time[index],
            "last_observation_time": self.last_observation_time[index],
            "sample_weight": self.sample_weight[index],
        }
        if self.x_multimodal is not None:
            result["x_multimodal"] = self.x_multimodal[index]
        if self.x_time is not None:
            result["x_time"] = self.x_time[index]
            result["x_delta"] = self.x_delta[index]
        return result
