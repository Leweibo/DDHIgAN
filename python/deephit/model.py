import torch
import torch.nn as nn
import numpy as np


def deterministic_cumprod(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Use the deterministic CPU backward path when CUDA lacks one."""
    if values.is_cuda and torch.are_deterministic_algorithms_enabled():
        return torch.cumprod(values.cpu(), dim=dim).to(values.device)
    return torch.cumprod(values, dim=dim)


def deterministic_cumsum(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Use the deterministic CPU backward path when CUDA lacks one."""
    if values.is_cuda and torch.are_deterministic_algorithms_enabled():
        return torch.cumsum(values.cpu(), dim=dim).to(values.device)
    return torch.cumsum(values, dim=dim)


def _mlp(input_dim: int, hidden_dims: list[int], output_dim: int,
         dropout: float) -> nn.Sequential:
    layers = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)])
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class FormalDynamicDeepHit(nn.Module):
    """Original-style Dynamic-DeepHit with temporal attention and PMF output."""

    def __init__(
        self,
        value_dim: int,
        missing_dim: int,
        longitudinal_dim: int,
        hidden_dim: int,
        rnn_layers: int,
        attention_hidden: int,
        event_hidden: list[int],
        num_event_bins: int,
        dropout: float = 0.2,
        multimodal_dim: int = 0,
        multimodal_hidden: int | None = None,
        multimodal_route_count: int = 0,
        multimodal_route_embedding_dim: int = 0,
        multimodal_route_hidden: int | None = None,
        multimodal_fusion_dim: int | None = None,
        multimodal_fusion_mode: str = "static_concat",
        multimodal_gate_hidden: int | None = None,
        multimodal_route_dropout: float = 0.0,
        multimodal_all_text_dropout: float = 0.0,
        multimodal_residual_gate_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.value_dim = value_dim
        self.missing_dim = missing_dim
        self.longitudinal_dim = longitudinal_dim
        self.num_event_bins = num_event_bins
        self.multimodal_dim = int(multimodal_dim)
        self.multimodal_route_count = int(multimodal_route_count)
        self.multimodal_route_embedding_dim = int(multimodal_route_embedding_dim)
        self.multimodal_fusion_mode = str(multimodal_fusion_mode)
        self.multimodal_route_dropout = float(multimodal_route_dropout)
        self.multimodal_all_text_dropout = float(multimodal_all_text_dropout)
        if self.multimodal_fusion_mode not in {
            "static_concat", "landmark_conditioned_residual",
        }:
            raise ValueError("unsupported multimodal fusion mode")
        if not 0.0 <= self.multimodal_route_dropout <= 1.0:
            raise ValueError("multimodal_route_dropout must be within [0, 1]")
        if not 0.0 <= self.multimodal_all_text_dropout <= 1.0:
            raise ValueError("multimodal_all_text_dropout must be within [0, 1]")
        self.gru = nn.GRU(
            value_dim + missing_dim,
            hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        final_feature_dim = value_dim - 1 + missing_dim
        self.longitudinal_head = nn.Linear(hidden_dim, longitudinal_dim)
        self.attention = _mlp(
            hidden_dim + final_feature_dim, [attention_hidden], 1, dropout
        )
        self.multimodal_projection = None
        self.multimodal_route_projections = None
        self.multimodal_route_gate = None
        self.multimodal_residual_gate = None
        self.multimodal_text_to_clinical = None
        clinical_feature_dim = hidden_dim + final_feature_dim
        event_input_dim = clinical_feature_dim
        if self.multimodal_route_count > 0:
            if self.multimodal_route_embedding_dim <= 0:
                raise ValueError("route-wise multimodal projection requires a positive route embedding dimension")
            expected_dim = self.multimodal_route_count * (1 + self.multimodal_route_embedding_dim)
            if self.multimodal_dim != expected_dim:
                raise ValueError(
                    f"route-wise multimodal input_dim={self.multimodal_dim} "
                    f"must equal {self.multimodal_route_count} * "
                    f"(1 + {self.multimodal_route_embedding_dim})"
                )
            route_hidden = int(multimodal_route_hidden or self.multimodal_route_embedding_dim)
            self.multimodal_route_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.multimodal_route_embedding_dim, route_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(self.multimodal_route_count)
            ])
            fusion_dim = int(multimodal_fusion_dim or route_hidden)
            if self.multimodal_fusion_mode == "landmark_conditioned_residual":
                gate_hidden = int(multimodal_gate_hidden or route_hidden)
                # Each route is scored in the context of the current clinical
                # state, landmark query time, and effective route availability.
                self.multimodal_route_gate = _mlp(
                    route_hidden + clinical_feature_dim + 2,
                    [gate_hidden],
                    1,
                    dropout,
                )
                self.multimodal_residual_gate = _mlp(
                    clinical_feature_dim + fusion_dim + 1 + self.multimodal_route_count,
                    [gate_hidden],
                    1,
                    dropout,
                )
                with torch.no_grad():
                    self.multimodal_residual_gate[-1].bias.fill_(
                        float(multimodal_residual_gate_bias)
                    )
                self.multimodal_text_to_clinical = nn.Linear(
                    fusion_dim, clinical_feature_dim
                )
                event_input_dim += self.multimodal_route_count
            else:
                self.multimodal_route_gate = nn.Sequential(
                    nn.Linear(route_hidden, 1),
                )
                event_input_dim += fusion_dim + self.multimodal_route_count
            self.multimodal_fusion_projection = nn.Linear(route_hidden, fusion_dim)
        elif self.multimodal_dim > 0:
            projected_dim = int(multimodal_hidden or self.multimodal_dim)
            self.multimodal_projection = _mlp(
                self.multimodal_dim, [projected_dim], projected_dim, dropout
            )
            event_input_dim += projected_dim
        self.event_network = _mlp(
            event_input_dim,
            event_hidden,
            num_event_bins + 1,
            dropout,
        )

    def forward(
        self,
        x_values: torch.Tensor,
        x_missing: torch.Tensor,
        sequence_mask: torch.Tensor,
        x_multimodal: torch.Tensor | None = None,
        query_time: torch.Tensor | None = None,
        x_time: torch.Tensor | None = None,
        x_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        lengths = sequence_mask.sum(1).long().clamp(min=1)
        rnn_input = torch.cat([x_values, x_missing], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            rnn_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_hidden, _ = self.gru(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_hidden, batch_first=True, total_length=x_values.size(1)
        )

        longitudinal_prediction = self.longitudinal_head(hidden[:, :-1])
        batch_index = torch.arange(x_values.size(0), device=x_values.device)
        final_index = lengths - 1
        final_values = x_values[batch_index, final_index, 1:]
        final_missing = x_missing[batch_index, final_index]
        final_features = torch.cat([final_values, final_missing], dim=-1)

        repeated_final = final_features.unsqueeze(1).expand(-1, hidden.size(1), -1)
        attention_logits = self.attention(
            torch.cat([hidden, repeated_final], dim=-1)
        ).squeeze(-1)
        positions = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0)
        history_mask = positions < final_index.unsqueeze(1)
        attention_power = torch.exp(attention_logits.clamp(max=30.0)) * history_mask
        attention_weight = attention_power / attention_power.sum(1, keepdim=True).clamp_min(1e-8)
        context = torch.sum(attention_weight.unsqueeze(-1) * hidden, dim=1)

        clinical_features = torch.cat([final_features, context], dim=-1)
        event_features = [clinical_features]
        multimodal_embedding = None
        text_residual_gate = None
        effective_text_availability = None
        if self.multimodal_route_projections is not None:
            if x_multimodal is None:
                x_multimodal = torch.zeros(
                    x_values.size(0), self.multimodal_dim, device=x_values.device
                )
            route_embeddings = []
            availability_flags = []
            stride = 1 + self.multimodal_route_embedding_dim
            for route_index, projection in enumerate(self.multimodal_route_projections):
                start = route_index * stride
                availability = x_multimodal[:, start:start + 1]
                route_vector = x_multimodal[:, start + 1:start + stride]
                # A missing route remains an exact zero representation despite
                # the projection bias; its explicit flag is fused separately.
                route_embeddings.append(projection(route_vector) * availability)
                availability_flags.append(availability)
            route_stack = torch.stack(route_embeddings, dim=1)
            availability_stack = torch.cat(availability_flags, dim=1)
            effective_availability = availability_stack
            if self.training and self.multimodal_route_dropout > 0.0:
                route_keep = (
                    torch.rand_like(availability_stack) >= self.multimodal_route_dropout
                ).to(availability_stack.dtype)
                effective_availability = effective_availability * route_keep
            if self.training and self.multimodal_all_text_dropout > 0.0:
                all_text_keep = (
                    torch.rand(
                        availability_stack.size(0), 1, device=availability_stack.device
                    ) >= self.multimodal_all_text_dropout
                ).to(availability_stack.dtype)
                effective_availability = effective_availability * all_text_keep
            route_stack = route_stack * effective_availability.unsqueeze(-1)
            if query_time is None:
                query_time = torch.zeros(
                    x_values.size(0), device=x_values.device, dtype=x_values.dtype
                )
            query_feature = query_time.reshape(-1, 1).to(x_values.dtype)
            if self.multimodal_fusion_mode == "landmark_conditioned_residual":
                repeated_clinical = clinical_features.unsqueeze(1).expand(
                    -1, self.multimodal_route_count, -1
                )
                repeated_time = query_feature.unsqueeze(1).expand(
                    -1, self.multimodal_route_count, -1
                )
                repeated_availability = effective_availability.unsqueeze(-1)
                gate_input = torch.cat(
                    [route_stack, repeated_clinical, repeated_time, repeated_availability],
                    dim=-1,
                )
                gate_logits = self.multimodal_route_gate(gate_input).squeeze(-1)
            else:
                gate_logits = self.multimodal_route_gate(route_stack).squeeze(-1)
            gate_logits = gate_logits.masked_fill(effective_availability <= 0, -1e9)
            gate_weight = torch.softmax(gate_logits, dim=1) * effective_availability
            gate_weight = gate_weight / gate_weight.sum(1, keepdim=True).clamp_min(1e-8)
            fused_routes = torch.sum(gate_weight.unsqueeze(-1) * route_stack, dim=1)
            fused_text = self.multimodal_fusion_projection(fused_routes)
            has_text = (effective_availability.sum(1, keepdim=True) > 0).to(fused_text.dtype)
            # Preserve an exact clinical-only fallback despite projection biases.
            fused_text = fused_text * has_text
            multimodal_embedding = torch.cat(
                [fused_text, effective_availability], dim=1
            )
            if self.multimodal_fusion_mode == "landmark_conditioned_residual":
                residual_gate_input = torch.cat(
                    [
                        clinical_features,
                        fused_text,
                        query_feature,
                        effective_availability,
                    ],
                    dim=1,
                )
                text_residual_gate = torch.sigmoid(
                    self.multimodal_residual_gate(residual_gate_input)
                ) * has_text
                clinical_features = clinical_features + text_residual_gate * (
                    self.multimodal_text_to_clinical(fused_text) * has_text
                )
                event_features = [clinical_features, effective_availability]
            else:
                event_features.append(multimodal_embedding)
            effective_text_availability = effective_availability
        elif self.multimodal_projection is not None:
            if x_multimodal is None:
                x_multimodal = torch.zeros(
                    x_values.size(0), self.multimodal_dim, device=x_values.device
                )
            multimodal_embedding = self.multimodal_projection(x_multimodal)
            event_features.append(multimodal_embedding)
        logits = self.event_network(torch.cat(event_features, dim=-1))
        probability = torch.softmax(logits, dim=-1)
        event_pmf = probability[:, : self.num_event_bins]
        tail_probability = probability[:, self.num_event_bins]
        cif = deterministic_cumsum(event_pmf, dim=1)
        survival = 1.0 - cif
        result = {
            "logits": logits,
            "event_pmf": event_pmf,
            "tail_probability": tail_probability,
            "cif": cif,
            "survival": survival,
            "attention": attention_weight,
            "longitudinal_prediction": longitudinal_prediction,
            "clinical_features": torch.cat([final_features, context], dim=-1),
        }
        if multimodal_embedding is not None:
            result["multimodal_embedding"] = multimodal_embedding
            if self.multimodal_route_gate is not None:
                result["text_route_attention"] = gate_weight
                result["effective_text_availability"] = effective_text_availability
            if text_residual_gate is not None:
                result["text_residual_gate"] = text_residual_gate
        return result


class TimeDeltaDynamicDeepHit(FormalDynamicDeepHit):
    """Formal Dynamic-DeepHit with variable-wise learned exponential decay."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta_log_rate = nn.Parameter(torch.zeros(self.longitudinal_dim))

    def forward(
        self,
        x_values: torch.Tensor,
        x_missing: torch.Tensor,
        sequence_mask: torch.Tensor,
        x_multimodal: torch.Tensor | None = None,
        query_time: torch.Tensor | None = None,
        x_time: torch.Tensor | None = None,
        x_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x_delta is None or x_delta.shape != x_missing.shape:
            raise ValueError("TimeDelta requires x_delta matching x_missing")
        if torch.any(x_delta < 0) or not torch.isfinite(x_delta).all():
            raise ValueError("x_delta must contain finite non-negative years")
        long_start = self.value_dim - self.longitudinal_dim
        decay = torch.exp(
            -torch.nn.functional.softplus(self.delta_log_rate) * x_delta
        )
        decayed_values = torch.cat(
            [x_values[..., :long_start], x_values[..., long_start:] * decay],
            dim=-1,
        )
        return super().forward(
            decayed_values,
            x_missing,
            sequence_mask,
            x_multimodal=x_multimodal,
            query_time=query_time,
        )


class _CDEVectorField(nn.Module):
    def __init__(self, hidden_dim: int, input_channels: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_channels = input_channels
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * input_channels),
        )

    def forward(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).view(
            *state.shape[:-1], self.hidden_dim, self.input_channels
        )


class ContinuousTimeDynamicDeepHit(nn.Module):
    """Causal rectilinear Neural CDE with the existing PMF prediction head."""

    def __init__(
        self,
        value_dim: int,
        missing_dim: int,
        longitudinal_dim: int,
        hidden_dim: int,
        event_hidden: list[int],
        num_event_bins: int,
        dropout: float = 0.2,
        solver: str = "rk4",
        step_size: float = 1.0,
    ) -> None:
        super().__init__()
        if missing_dim != longitudinal_dim:
            raise ValueError("CDE requires one missingness channel per longitudinal variable")
        if solver != "rk4" or step_size <= 0:
            raise ValueError("CDE uses fixed deterministic rk4 with positive step_size")
        self.value_dim = value_dim
        self.missing_dim = missing_dim
        self.longitudinal_dim = longitudinal_dim
        self.hidden_dim = hidden_dim
        self.num_event_bins = num_event_bins
        self.solver = solver
        self.step_size = float(step_size)
        self.static_dim = value_dim - 1 - longitudinal_dim
        self.control_dim = 1 + 3 * longitudinal_dim
        self.initial = nn.Linear(self.static_dim + self.control_dim, hidden_dim)
        self.cde_func = _CDEVectorField(hidden_dim, self.control_dim)
        self.longitudinal_head = nn.Linear(hidden_dim, longitudinal_dim)
        final_feature_dim = value_dim - 1 + missing_dim
        self.event_network = _mlp(
            hidden_dim + final_feature_dim,
            event_hidden,
            num_event_bins + 1,
            dropout,
        )

    def control_path(
        self,
        x_values: torch.Tensor,
        x_missing: torch.Tensor,
        sequence_mask: torch.Tensor,
        x_time: torch.Tensor,
        x_delta: torch.Tensor,
    ) -> torch.Tensor:
        if x_time.shape != sequence_mask.shape or x_delta.shape != x_missing.shape:
            raise ValueError("CDE time/delta tensors do not match the sequence")
        if torch.any(x_delta < 0) or not torch.isfinite(x_time).all() or not torch.isfinite(x_delta).all():
            raise ValueError("CDE time/delta tensors must be finite with non-negative deltas")
        long_values = x_values[..., -self.longitudinal_dim:]
        control = torch.cat(
            [x_time.unsqueeze(-1), long_values, x_missing, x_delta], dim=-1
        )
        lengths = sequence_mask.sum(1).long().clamp(min=1)
        batch_index = torch.arange(control.size(0), device=control.device)
        final = control[batch_index, lengths - 1].unsqueeze(1)
        return torch.where(sequence_mask.bool().unsqueeze(-1), control, final)

    def forward(
        self,
        x_values: torch.Tensor,
        x_missing: torch.Tensor,
        sequence_mask: torch.Tensor,
        x_multimodal: torch.Tensor | None = None,
        query_time: torch.Tensor | None = None,
        x_time: torch.Tensor | None = None,
        x_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x_multimodal is not None:
            raise ValueError("continuous-time candidates are clinical-only")
        if x_time is None or x_delta is None:
            raise ValueError("CDE requires x_time and x_delta")
        try:
            import torchcde
        except ImportError as exc:
            raise RuntimeError(
                "DDHIgAN-CDE requires torchcde after RIS compatibility approval"
            ) from exc

        control = self.control_path(
            x_values, x_missing, sequence_mask, x_time, x_delta
        )
        coefficients = torchcde.linear_interpolation_coeffs(control, rectilinear=0)
        interpolation = torchcde.LinearInterpolation(coefficients)
        static = x_values[:, 0, 1:1 + self.static_dim]
        initial = torch.tanh(self.initial(torch.cat([static, control[:, 0]], dim=-1)))
        rectilinear_hidden = torchcde.cdeint(
            X=interpolation,
            func=self.cde_func,
            z0=initial,
            t=interpolation.grid_points,
            method=self.solver,
            options={"step_size": self.step_size},
            adjoint=False,
        )
        hidden = rectilinear_hidden[:, ::2]
        lengths = sequence_mask.sum(1).long().clamp(min=1)
        batch_index = torch.arange(x_values.size(0), device=x_values.device)
        final_index = lengths - 1
        final_values = x_values[batch_index, final_index, 1:]
        final_missing = x_missing[batch_index, final_index]
        final_features = torch.cat([final_values, final_missing], dim=-1)
        context = hidden[batch_index, final_index]
        clinical_features = torch.cat([final_features, context], dim=-1)
        logits = self.event_network(clinical_features)
        probability = torch.softmax(logits, dim=-1)
        event_pmf = probability[:, :self.num_event_bins]
        cif = deterministic_cumsum(event_pmf, dim=1)
        return {
            "logits": logits,
            "event_pmf": event_pmf,
            "tail_probability": probability[:, self.num_event_bins],
            "cif": cif,
            "survival": 1.0 - cif,
            "attention": torch.zeros_like(sequence_mask),
            "longitudinal_prediction": self.longitudinal_head(hidden[:, :-1]),
            "clinical_features": clinical_features,
        }


class StaticDeepHit(nn.Module):
    """DeepHit with static covariates only (last observed longitudinal value
    concatenated with static features).
    """
    def __init__(self, static_dim: int, long_dim: int, hidden_dims: list[int],
                 num_time_bins: int, alpha: float = 0.5, sigma: float = 0.1):
        super().__init__()
        self.num_time_bins = num_time_bins
        self.alpha = alpha
        self.sigma = sigma

        input_dim = static_dim + long_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)])
            prev = h
        layers.append(nn.Linear(prev, num_time_bins))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_static: torch.Tensor, x_long: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # x_static: [B, static_dim]
        # x_long: [B, T, long_dim]
        # mask: [B, T]
        # Use last observed longitudinal value
        last_idx = (mask.sum(dim=1) - 1).long().clamp(min=0)
        B = x_long.size(0)
        last_long = x_long[torch.arange(B), last_idx, :]  # [B, long_dim]
        x = torch.cat([x_static, last_long], dim=1)
        logits = self.mlp(x)  # [B, num_time_bins]
        # Convert to hazard probabilities via sigmoid
        hazard = torch.sigmoid(logits)
        return hazard

    def survival_from_hazard(self, hazard: torch.Tensor) -> torch.Tensor:
        # hazard: [B, num_time_bins]
        # S(t) = prod_{k=1..t} (1 - h_k)
        S = deterministic_cumprod(1 - hazard, dim=1)
        return S

    def predict_survival(self, x_static: torch.Tensor, x_long: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            hazard = self.forward(x_static, x_long, mask)
            S = self.survival_from_hazard(hazard)
        return S


class DynamicDeepHit(nn.Module):
    """Dynamic-DeepHit: GRU processes longitudinal sequence, hidden state
    combined with static features for survival prediction.
    """
    def __init__(self, static_dim: int, long_dim: int, gru_hidden: int,
                 hidden_dims: list[int], num_time_bins: int,
                 alpha: float = 0.5, sigma: float = 0.1):
        super().__init__()
        self.num_time_bins = num_time_bins
        self.alpha = alpha
        self.sigma = sigma

        self.gru = nn.GRU(long_dim, gru_hidden, batch_first=True)
        input_dim = static_dim + gru_hidden
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)])
            prev = h
        layers.append(nn.Linear(prev, num_time_bins))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_static: torch.Tensor, x_long: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # x_long: [B, T, long_dim]
        # Pack padded sequence for efficient GRU processing
        lengths = mask.sum(dim=1).long().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_long, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, hn = self.gru(packed)  # output: packed, hn: [1, B, gru_hidden]
        # Use last hidden state (final time step) instead of mean pooling
        h_pooled = hn.squeeze(0)  # [B, gru_hidden]
        x = torch.cat([x_static, h_pooled], dim=1)
        logits = self.mlp(x)
        hazard = torch.sigmoid(logits)
        return hazard

    def survival_from_hazard(self, hazard: torch.Tensor) -> torch.Tensor:
        S = deterministic_cumprod(1 - hazard, dim=1)
        return S

    def predict_survival(self, x_static: torch.Tensor, x_long: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            hazard = self.forward(x_static, x_long, mask)
            S = self.survival_from_hazard(hazard)
        return S


def deephit_loss(hazard: torch.Tensor, time: torch.Tensor, event: torch.Tensor,
                 num_time_bins: int, alpha: float, sigma: float) -> torch.Tensor:
    """DeepHit loss = alpha * L_likelihood + (1-alpha) * L_ranking.
    hazard: [B, num_time_bins]
    time: [B]   (discrete bin indices, 0-indexed)
    event: [B]  (0 or 1)
    """
    B = hazard.size(0)
    device = hazard.device
    time_index = time.long().clamp(min=0, max=num_time_bins - 1)
    event_mask = event.to(dtype=torch.bool)

    # --- Likelihood loss ---
    S = deterministic_cumprod(1 - hazard, dim=1)  # [B, num_time_bins]
    S = torch.clamp(S, min=1e-7)
    h = torch.clamp(hazard, min=1e-7, max=1 - 1e-7)

    # P(T=t, E=1) = h_t * S_{t-1}; P(T>t, E=0) = S_t
    S_prev = torch.ones(B, 1, device=device)
    S_prev = torch.cat([S_prev, S[:, :-1]], dim=1)

    gather_index = time_index.view(-1, 1)
    h_at_t = h.gather(1, gather_index).squeeze(1)
    S_prev_at_t = S_prev.gather(1, gather_index).squeeze(1)
    S_at_t = S.gather(1, gather_index).squeeze(1)
    log_event = torch.log(h_at_t) + torch.log(S_prev_at_t)
    log_censored = torch.log(S_at_t)
    log_lik = torch.where(event_mask, log_event, log_censored)

    L_lik = -log_lik.mean()

    # --- Ranking loss ---
    # For pairs where t_i < t_j and event_i=1, encourage risk_i > risk_j
    # Use cumulative hazard as risk proxy
    risk = hazard.sum(dim=1)  # [B]
    pair_mask = event_mask.view(-1, 1) & (time_index.view(-1, 1) < time_index.view(1, -1))
    if pair_mask.any():
        risk_diff = risk.view(-1, 1) - risk.view(1, -1)
        rank_terms = torch.exp(-risk_diff / sigma).clamp(max=1e6)
        L_rank = rank_terms[pair_mask].mean()
    else:
        L_rank = hazard.new_tensor(0.0)

    loss = alpha * L_lik + (1 - alpha) * L_rank
    return loss
