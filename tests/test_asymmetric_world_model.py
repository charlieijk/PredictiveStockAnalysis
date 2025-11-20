import numpy as np
import pytest

from asymmetric_world_model import (
    AsymmetricWorldModelConfig,
    AsymmetricWorldModelTrainer,
)


@pytest.mark.skipif(not hasattr(np, "random"), reason="NumPy required")
def test_world_model_fits_and_predicts(tmp_path):
    n_samples = 60
    n_features = 6
    rng = np.random.default_rng(seed=42)
    features = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    targets = rng.normal(size=n_samples).astype(np.float32)

    config = AsymmetricWorldModelConfig(
        sequence_length=4,
        backward_hidden_dim=12,
        backward_layers=1,
        forward_hidden_dims=(32,),
        bottleneck_dim=4,
        dropout=0.0,
        lr=5e-3,
        weight_decay=0.0,
        reconstruction_weight=0.2,
        prediction_weight=1.0,
        next_state_weight=0.2,
        batch_size=8,
        epochs=2,
        train_fraction=0.6,
        val_fraction=0.2,
        activation_logging_batches=2,
        log_dir=str(tmp_path),
        device="cpu",
    )

    trainer = AsymmetricWorldModelTrainer(
        config=config,
        feature_names=[f"f_{i}" for i in range(n_features)],
        output_dir=str(tmp_path),
    )
    trainer.fit(features, targets)
    preds = trainer.predict(features[-10:])

    assert preds.shape == (10,)
    assert "asymmetric_world_model" in trainer.metrics


def test_world_model_config_requires_test_fraction():
    with pytest.raises(ValueError):
        AsymmetricWorldModelConfig(train_fraction=0.9, val_fraction=0.1)
