"""
Asymmetric world model:

- Backward GRU reconstructs state_{t-1} from state_t
- Forward MLP predicts state_{t+1} and next-day return, modulated via FiLM
- Provides hooks to log activation variance and perturbation-based feature importance
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _log_normal_(tensor: Tensor, mean: float = 0.0, std: float = 0.35) -> None:
    """Initialise tensor with signed log-normal noise to encourage hierarchy."""
    with torch.no_grad():
        noise = torch.empty_like(tensor).normal_(mean, std)
        sign = torch.randint(0, 2, tensor.shape, device=tensor.device, dtype=torch.float32) * 2 - 1
        tensor.copy_(sign * noise.exp())


@dataclass
class AsymmetricWorldModelConfig:
    """Hyperparameters for the asymmetric world model."""

    sequence_length: int = 16
    backward_hidden_dim: int = 96
    backward_layers: int = 2
    forward_hidden_dims: Tuple[int, ...] = (128, 64)
    bottleneck_dim: int = 8
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    reconstruction_weight: float = 1.0
    prediction_weight: float = 1.0
    next_state_weight: float = 0.5
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 7
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    activation_logging_batches: int = 6
    perturbation_std: float = 0.2
    clip_grad_norm: float = 1.0
    log_dir: str = "logs"
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if (self.train_fraction + self.val_fraction) >= 0.95:
            raise ValueError("Reserve at least 5% of data for testing.")
        if self.sequence_length < 3:
            raise ValueError("sequence_length must be >= 3 for prev/next reconstruction.")
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


class WorldModelDataset(Dataset):
    """Sliding window dataset that exposes contiguous state windows."""

    def __init__(self, states: np.ndarray, targets: np.ndarray, sequence_length: int):
        if len(states) != len(targets):
            raise ValueError("states and targets must share length")
        if len(states) < 3:
            raise ValueError("Need at least 3 timesteps to build the world model dataset.")

        self.sequence_length = max(3, min(sequence_length, len(states)))
        self.states = torch.from_numpy(states.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32)).unsqueeze(-1)
        max_start = len(self.states) - self.sequence_length
        self.indices = list(range(max_start + 1)) if max_start >= 0 else [0]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        start = self.indices[idx]
        end = start + self.sequence_length
        return self.states[start:end], self.targets[start:end]


class ForwardMLPWithFiLM(nn.Module):
    """Forward predictor modulated through FiLM parameters derived from the backward encoder."""

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Sequence[int],
        bottleneck_dim: int,
        context_dim: int,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        if not hidden_dims:
            hidden_dims = (128,)
        self.layer_dims = list(hidden_dims) + [bottleneck_dim, hidden_dims[-1]]
        self.layers = nn.ModuleList()
        in_dim = state_dim
        for out_dim in self.layer_dims:
            layer = nn.Linear(in_dim, out_dim)
            _log_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            in_dim = out_dim

        self.activation = activation
        self.film_generator = nn.Linear(context_dim, 2 * sum(self.layer_dims))
        _log_normal_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        return_activations: bool = False,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Run FiLM-conditioned forward network."""
        film_params = self.film_generator(context)
        gammas: List[Tensor] = []
        betas: List[Tensor] = []
        cursor = 0
        for dim in self.layer_dims:
            gammas.append(film_params[:, cursor : cursor + dim])
            cursor += dim
            betas.append(film_params[:, cursor : cursor + dim])
            cursor += dim

        activations: List[Tensor] = []
        out = x
        for idx, layer in enumerate(self.layers):
            out = layer(out)
            out = self.activation(out)
            gamma = torch.tanh(gammas[idx])
            beta = betas[idx]
            out = out * (1 + gamma) + beta
            if return_activations:
                activations.append(out.detach())
        return out, activations


class AsymmetricWorldModel(nn.Module):
    """Combines backward encoder and FiLM-modulated forward predictor."""

    def __init__(self, state_dim: int, config: AsymmetricWorldModelConfig):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.backward_gru = nn.GRU(
            input_size=state_dim,
            hidden_size=config.backward_hidden_dim,
            num_layers=config.backward_layers,
            batch_first=True,
            dropout=config.dropout if config.backward_layers > 1 else 0.0,
        )

        self.backward_decoder = nn.Sequential(
            nn.Linear(config.backward_hidden_dim, config.backward_hidden_dim),
            nn.GELU(),
            nn.Linear(config.backward_hidden_dim, state_dim),
        )
        for layer in self.backward_decoder:
            if isinstance(layer, nn.Linear):
                _log_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.forward_mlp = ForwardMLPWithFiLM(
            state_dim=state_dim,
            hidden_dims=config.forward_hidden_dims,
            bottleneck_dim=config.bottleneck_dim,
            context_dim=config.backward_hidden_dim,
        )
        final_hidden_dim = config.forward_hidden_dims[-1]
        self.next_state_head = nn.Linear(final_hidden_dim, state_dim)
        self.target_head = nn.Linear(final_hidden_dim, 1)
        _log_normal_(self.next_state_head.weight)
        _log_normal_(self.target_head.weight)
        nn.init.zeros_(self.next_state_head.bias)
        nn.init.zeros_(self.target_head.bias)

    def encode(self, state_sequence: Tensor) -> Tuple[Tensor, Tensor]:
        """Return reconstruction of previous state and hidden activations."""
        outputs, _ = self.backward_gru(state_sequence)
        prev_state_hat = self.backward_decoder(outputs)
        return prev_state_hat, outputs

    def forward_predictor(
        self,
        current_states: Tensor,
        context: Tensor,
        return_activations: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        """Predict next states/targets conditioned on FiLM parameters."""
        latent, activations = self.forward_mlp(current_states, context, return_activations)
        next_state = self.next_state_head(latent)
        target = self.target_head(latent)
        return latent, next_state, target, activations


class AsymmetricWorldModelTrainer:
    """High-level trainer that exposes fit/predict/save hooks for integration."""

    def __init__(
        self,
        config: AsymmetricWorldModelConfig,
        feature_names: Optional[Sequence[str]] = None,
        output_dir: str = "output",
    ):
        self.config = config
        self.feature_names = list(feature_names) if feature_names else []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config.device)
        self.model: Optional[AsymmetricWorldModel] = None
        self.history: Dict[str, List[float]] = {"train": [], "val": []}
        self.activation_stats: Dict[str, List[List[float]]] = {}
        self.feature_importance: Dict[str, float] = {}
        self.test_predictions: Optional[np.ndarray] = None
        self.test_targets: Optional[np.ndarray] = None
        self.metrics: Dict[str, Dict[str, float]] = {}
        self._latest_splits: Optional[Dict[str, Dict[str, np.ndarray]]] = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train the asymmetric world model."""
        if features.ndim != 2:
            raise ValueError("features must be 2D (samples, features)")
        if len(features) != len(targets):
            raise ValueError("Features/targets lengths mismatch")

        state_dim = features.shape[1]
        self.model = AsymmetricWorldModel(state_dim, self.config).to(self.device)

        splits = self._split_sequences(features, targets)
        self._latest_splits = splits
        loaders = {
            name: DataLoader(
                WorldModelDataset(split["states"], split["targets"], self.config.sequence_length),
                batch_size=self.config.batch_size,
                shuffle=(name == "train"),
            )
            for name, split in splits.items()
            if len(split["states"]) >= 3
        }

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        best_val = float("inf")
        best_state: Optional[Dict[str, Tensor]] = None
        patience = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(loaders["train"], optimizer, train_mode=True)
            val_loss = self._run_epoch(loaders["val"], optimizer=None, train_mode=False)
            self.history["train"].append(train_loss)
            self.history["val"].append(val_loss)

            logger.info(
                "WorldModel Epoch %d/%d - train: %.5f, val: %.5f",
                epoch,
                self.config.epochs,
                train_loss,
                val_loss,
            )

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    logger.info("Early stopping triggered after %d epochs", epoch)
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        test_loss = self._run_epoch(loaders["test"], optimizer=None, train_mode=False)
        test_states = splits["test"]["states"]
        test_targets = splits["test"]["targets"]
        preds = self.predict(test_states)
        self.test_predictions = preds
        self.test_targets = test_targets
        self.metrics["asymmetric_world_model"] = {
            "test": self._compute_metric_dict(test_targets, preds),
            "baseline_mse": self._compute_baseline_mse(test_targets),
        }

        self.activation_stats = self._compute_activation_stats(loaders["val"])
        self.feature_importance = self._estimate_perturbation_importance(loaders["val"])
        self._persist_logging_artifacts(test_loss)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict next-day returns for a contiguous block of features."""
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        states = torch.from_numpy(features.astype(np.float32)).to(self.device)
        n = len(states)
        if n < 3:
            raise ValueError("Need at least 3 timesteps to produce predictions.")

        seq_len = min(self.config.sequence_length, n)
        preds = torch.zeros(n, device=self.device)
        counts = torch.zeros(n, device=self.device)
        self.model.eval()
        with torch.no_grad():
            for start in range(0, n - seq_len + 1):
                window = states[start : start + seq_len].unsqueeze(0)
                pred_vals = self._predict_window_targets(window)[0]
                idx = torch.arange(
                    start + 1,
                    start + seq_len - 1,
                    device=self.device,
                    dtype=torch.long,
                )
                preds[idx] += pred_vals
                counts[idx] += 1

        # Fill boundaries by nearest available prediction
        filled_preds = preds.clone()
        last_valid: Optional[Tensor] = None
        for i in range(n):
            if counts[i] > 0:
                filled_preds[i] = preds[i] / counts[i]
                last_valid = filled_preds[i]
            elif last_valid is not None:
                filled_preds[i] = last_valid
        last_valid = None
        for i in range(n - 1, -1, -1):
            if counts[i] > 0:
                last_valid = filled_preds[i]
            elif last_valid is not None:
                filled_preds[i] = last_valid
        filled_preds = filled_preds.cpu().numpy()
        return filled_preds

    def save(self, filepath: str) -> None:
        """Persist model state + config."""
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config.__dict__,
                "feature_names": self.feature_names,
            },
            filepath,
        )

    # --- Internal helpers ---

    def _split_sequences(self, states: np.ndarray, targets: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        n = len(states)
        train_end = int(n * self.config.train_fraction)
        val_end = train_end + int(n * self.config.val_fraction)
        splits = {
            "train": {"states": states[:train_end], "targets": targets[:train_end]},
            "val": {"states": states[train_end:val_end], "targets": targets[train_end:val_end]},
            "test": {"states": states[val_end:], "targets": targets[val_end:]},
        }
        for name, split in splits.items():
            if len(split["states"]) < 3:
                raise ValueError(f"{name} split needs at least 3 timesteps for the world model.")
            if len(split["states"]) < self.config.sequence_length:
                logger.warning(
                    "%s split shorter than sequence length (%d vs %d). Using shorter windows.",
                    name,
                    len(split["states"]),
                    self.config.sequence_length,
                )
        return splits

    def _run_epoch(self, loader: DataLoader, optimizer, train_mode: bool) -> float:
        if self.model is None:
            raise RuntimeError("Model not initialised")
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        batches = 0

        for states, targets in loader:
            states = states.to(self.device)
            targets = targets.to(self.device).squeeze(-1)

            loss = self._compute_losses(states, targets)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                if self.config.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                optimizer.step()

            total_loss += loss.item()
            batches += 1

        return total_loss / max(1, batches)

    def _compute_losses(self, states: Tensor, targets: Tensor) -> Tensor:
        prev_pred, backward_hidden = self.model.encode(states)
        recon_loss = torch.nn.functional.mse_loss(prev_pred[:, 1:, :], states[:, :-1, :])

        core_states = states[:, 1:-1, :]
        next_states = states[:, 2:, :]
        core_targets = targets[:, 1:-1]
        context = backward_hidden[:, 1:-1, :]

        bsz, seq_len, feat_dim = core_states.shape
        flat_states = core_states.reshape(bsz * seq_len, feat_dim)
        flat_context = context.reshape(bsz * seq_len, context.shape[-1])
        flat_targets = core_targets.reshape(-1, 1)
        flat_next_states = next_states.reshape(bsz * seq_len, feat_dim)

        _, next_state_pred, target_pred, _ = self.model.forward_predictor(flat_states, flat_context)

        pred_loss = torch.nn.functional.mse_loss(target_pred, flat_targets)
        next_state_loss = torch.nn.functional.mse_loss(next_state_pred, flat_next_states)

        return (
            self.config.reconstruction_weight * recon_loss
            + self.config.prediction_weight * pred_loss
            + self.config.next_state_weight * next_state_loss
        )

    def _predict_window_targets(self, window: Tensor) -> Tensor:
        """Predict returns for a batch of contiguous windows."""
        if window.ndim == 2:
            window = window.unsqueeze(0)
        prev_pred, backward_hidden = self.model.encode(window)
        core_states = window[:, 1:-1, :]
        context = backward_hidden[:, 1:-1, :]
        bsz, seq_len, feat_dim = core_states.shape
        flat_states = core_states.reshape(bsz * seq_len, feat_dim)
        flat_context = context.reshape(bsz * seq_len, context.shape[-1])
        _, _, target_pred, _ = self.model.forward_predictor(flat_states, flat_context)
        return target_pred.reshape(bsz, seq_len)

    def _compute_activation_stats(self, loader: DataLoader) -> Dict[str, List[List[float]]]:
        stats: Dict[str, List[List[float]]] = {}
        batches = 0
        for states, _ in loader:
            states = states.to(self.device)
            prev_pred, backward_hidden = self.model.encode(states)
            core_states = states[:, 1:-1, :]
            context = backward_hidden[:, 1:-1, :]

            bsz, seq_len, feat_dim = core_states.shape
            flat_states = core_states.reshape(bsz * seq_len, feat_dim)
            flat_context = context.reshape(bsz * seq_len, context.shape[-1])

            _, _, _, activations = self.model.forward_predictor(
                flat_states,
                flat_context,
                return_activations=True,
            )
            for layer_idx, act in enumerate(activations):
                variances = act.var(dim=0).detach().cpu().tolist()
                stats.setdefault(f"layer_{layer_idx}", []).append(variances)
            batches += 1
            if batches >= self.config.activation_logging_batches:
                break
        return stats

    def _estimate_perturbation_importance(self, loader: DataLoader) -> Dict[str, float]:
        try:
            states, targets = next(iter(loader))
        except StopIteration:
            return {}
        states = states.to(self.device)
        base_preds = self._predict_window_targets(states)
        importance: Dict[str, float] = {}
        state_sample = states[:, 1:-1, :].reshape(-1, states.shape[-1])
        std = state_sample.std(dim=0)
        for idx in range(states.shape[-1]):
            perturbed = states.clone()
            perturbation = torch.randn_like(perturbed[:, :, idx]) * (std[idx] + 1e-6) * self.config.perturbation_std
            perturbed[:, :, idx] += perturbation
            perturbed_preds = self._predict_window_targets(perturbed)
            delta = (perturbed_preds - base_preds).abs().mean().item()
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            importance[name] = float(delta)
        return importance

    def _persist_logging_artifacts(self, test_loss: float) -> None:
        payload = {
            "history": self.history,
            "activation_variances": self.activation_stats,
            "feature_importance": self.feature_importance,
            "test_metrics": {
                "loss": test_loss,
                **self.metrics.get("asymmetric_world_model", {}),
            },
        }
        path = self.output_dir / "asymmetric_world_model_stats.json"
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved world model diagnostics to %s", path)

    def _compute_metric_dict(self, targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
        mse = float(np.mean((preds - targets) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(preds - targets)))
        mean_true = float(np.mean(targets))
        ss_tot = float(np.sum((targets - mean_true) ** 2))
        ss_res = float(np.sum((targets - preds) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if len(targets) > 1:
            true_dir = np.diff(targets) > 0
            pred_dir = np.diff(preds) > 0
            directional_accuracy = float(np.mean(true_dir == pred_dir))
        else:
            directional_accuracy = 0.0
        return {
            "rmse": rmse,
            "mae": mae,
            "mse": mse,
            "r2": r2,
            "directional_accuracy": directional_accuracy,
        }

    def _compute_baseline_mse(self, targets: np.ndarray) -> float:
        if len(targets) < 2:
            return float(np.mean(targets**2))
        baseline = np.roll(targets, 1)
        baseline[0] = targets[0]
        return float(np.mean((baseline - targets) ** 2))
