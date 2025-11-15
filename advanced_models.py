"""
Advanced Deep Learning Model Architectures

Implements state-of-the-art models for time series prediction:
- Transformer models with multi-head attention
- Temporal Convolutional Networks (TCN)
- Attention-LSTM hybrid models
- WaveNet-style architectures
- Bidirectional LSTM with attention
- GRU variants
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
import logging
from dataclasses import dataclass

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for advanced models."""

    # General
    sequence_length: int = 60
    n_features: int = 50
    n_outputs: int = 1

    # Transformer
    d_model: int = 128
    n_heads: int = 8
    n_transformer_blocks: int = 4
    ff_dim: int = 256
    dropout_rate: float = 0.1

    # TCN
    n_filters: int = 64
    kernel_size: int = 3
    n_tcn_blocks: int = 3
    dilations: List[int] = None

    # LSTM/GRU
    lstm_units: List[int] = None
    gru_units: List[int] = None

    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2

    def __post_init__(self):
        if self.dilations is None:
            self.dilations = [1, 2, 4, 8, 16]
        if self.lstm_units is None:
            self.lstm_units = [128, 64, 32]
        if self.gru_units is None:
            self.gru_units = [128, 64]


class TransformerBlock(layers.Layer):
    """
    Transformer block with multi-head attention.

    Components:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.att = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=dropout
        )

        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        # Multi-head attention with residual connection
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(layers.Layer):
    """
    Positional encoding for Transformer.

    Adds position information to input embeddings.
    """

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

        # Create positional encoding matrix
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]


class TemporalConvBlock(layers.Layer):
    """
    Temporal Convolutional Block with dilated convolutions.

    Features:
    - Causal convolution (no future leakage)
    - Dilation for large receptive field
    - Residual connections
    - Layer normalization
    """

    def __init__(self, n_filters: int, kernel_size: int, dilation_rate: int, dropout: float = 0.1):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        # Causal convolution
        self.conv1 = layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )

        self.conv2 = layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

        # Residual connection (1x1 conv for dimension matching)
        self.residual_conv = layers.Conv1D(filters=n_filters, kernel_size=1)

    def call(self, inputs, training=False):
        # First conv layer
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)

        # Second conv layer
        x = self.conv2(x)
        x = self.dropout2(x, training=training)

        # Residual connection
        residual = self.residual_conv(inputs)

        # Add and normalize
        return self.layernorm(x + residual)


class AttentionLayer(layers.Layer):
    """
    Attention mechanism for LSTM/GRU outputs.

    Learns to weight different timesteps differently.
    """

    def __init__(self, units: int):
        super().__init__()
        self.units = units

        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, inputs):
        # inputs shape: (batch_size, timesteps, features)

        # Attention scores
        score = self.V(tf.nn.tanh(self.W(inputs)))

        # Attention weights (softmax over time dimension)
        attention_weights = tf.nn.softmax(score, axis=1)

        # Context vector (weighted sum)
        context = attention_weights * inputs
        context = tf.reduce_sum(context, axis=1)

        return context, attention_weights


class TransformerModel:
    """
    Transformer model for time series prediction.

    Architecture:
    - Positional encoding
    - Multiple transformer blocks
    - Global average pooling
    - Dense layers for prediction
    """

    def __init__(self, config: ModelConfig):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for Transformer model")

        self.config = config
        self.model = None
        self.history = None

    def build(self) -> Model:
        """Build transformer model."""

        # Input
        inputs = layers.Input(shape=(self.config.sequence_length, self.config.n_features))

        # Project to d_model dimensions
        x = layers.Dense(self.config.d_model)(inputs)

        # Positional encoding
        pos_encoder = PositionalEncoding(self.config.sequence_length, self.config.d_model)
        x = pos_encoder(x)

        # Transformer blocks
        for _ in range(self.config.n_transformer_blocks):
            transformer_block = TransformerBlock(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                ff_dim=self.config.ff_dim,
                dropout=self.config.dropout_rate
            )
            x = transformer_block(x)

        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)

        # Output
        outputs = layers.Dense(self.config.n_outputs)(x)

        model = Model(inputs=inputs, outputs=outputs, name='transformer')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model = model
        logger.info(f"Built Transformer model with {model.count_params():,} parameters")

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train the model."""

        if self.model is None:
            self.build()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Train
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            validation_split=self.config.validation_split if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=1
        )

        self.history = history.history

        return self.history


class TCNModel:
    """
    Temporal Convolutional Network (TCN).

    Architecture:
    - Stack of dilated causal convolutions
    - Exponentially increasing dilation rates
    - Residual connections
    - Large receptive field
    """

    def __init__(self, config: ModelConfig):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for TCN model")

        self.config = config
        self.model = None
        self.history = None

    def build(self) -> Model:
        """Build TCN model."""

        # Input
        inputs = layers.Input(shape=(self.config.sequence_length, self.config.n_features))

        x = inputs

        # Stack of TCN blocks with increasing dilation
        for dilation in self.config.dilations:
            tcn_block = TemporalConvBlock(
                n_filters=self.config.n_filters,
                kernel_size=self.config.kernel_size,
                dilation_rate=dilation,
                dropout=self.config.dropout_rate
            )
            x = tcn_block(x)

        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)

        # Output
        outputs = layers.Dense(self.config.n_outputs)(x)

        model = Model(inputs=inputs, outputs=outputs, name='tcn')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model = model
        logger.info(f"Built TCN model with {model.count_params():,} parameters")

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train the model."""

        if self.model is None:
            self.build()

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            validation_split=self.config.validation_split if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=1
        )

        self.history = history.history

        return self.history


class AttentionLSTM:
    """
    LSTM with attention mechanism.

    Architecture:
    - Bidirectional LSTM layers
    - Attention layer to weight timesteps
    - Dense layers for prediction
    """

    def __init__(self, config: ModelConfig):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for Attention-LSTM model")

        self.config = config
        self.model = None
        self.history = None

    def build(self) -> Model:
        """Build attention-LSTM model."""

        # Input
        inputs = layers.Input(shape=(self.config.sequence_length, self.config.n_features))

        # Bidirectional LSTM layers
        x = inputs
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1 or True  # Always return sequences for attention

            x = layers.Bidirectional(
                layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate
                )
            )(x)

        # Attention layer
        attention_layer = AttentionLayer(units=128)
        context, attention_weights = attention_layer(x)

        # Dense layers
        x = layers.Dense(64, activation='relu')(context)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)

        # Output
        outputs = layers.Dense(self.config.n_outputs)(x)

        model = Model(inputs=inputs, outputs=outputs, name='attention_lstm')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model = model
        logger.info(f"Built Attention-LSTM model with {model.count_params():,} parameters")

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train the model."""

        if self.model is None:
            self.build()

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            validation_split=self.config.validation_split if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=1
        )

        self.history = history.history

        return self.history


class GRUModel:
    """
    GRU-based model (faster alternative to LSTM).

    Architecture:
    - Stacked GRU layers
    - Batch normalization
    - Dense layers
    """

    def __init__(self, config: ModelConfig):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for GRU model")

        self.config = config
        self.model = None
        self.history = None

    def build(self) -> Model:
        """Build GRU model."""

        # Input
        inputs = layers.Input(shape=(self.config.sequence_length, self.config.n_features))

        # GRU layers
        x = inputs
        for i, units in enumerate(self.config.gru_units):
            return_sequences = i < len(self.config.gru_units) - 1

            x = layers.GRU(
                units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            )(x)

            if return_sequences:
                x = layers.BatchNormalization()(x)

        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)

        # Output
        outputs = layers.Dense(self.config.n_outputs)(x)

        model = Model(inputs=inputs, outputs=outputs, name='gru')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model = model
        logger.info(f"Built GRU model with {model.count_params():,} parameters")

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train the model."""

        if self.model is None:
            self.build()

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            validation_split=self.config.validation_split if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=1
        )

        self.history = history.history

        return self.history


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    if not TF_AVAILABLE:
        print("TensorFlow not available. Install with: pip install tensorflow")
        exit(1)

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    sequence_length = 60
    n_features = 20

    X_train = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
    y_train = np.random.randn(n_samples, 1).astype(np.float32)

    X_val = np.random.randn(200, sequence_length, n_features).astype(np.float32)
    y_val = np.random.randn(200, 1).astype(np.float32)

    # Configuration
    config = ModelConfig(
        sequence_length=sequence_length,
        n_features=n_features,
        epochs=5  # Reduced for demo
    )

    print("\n=== Transformer Model ===")
    transformer = TransformerModel(config)
    transformer.build()
    print(transformer.model.summary())

    print("\n=== TCN Model ===")
    tcn = TCNModel(config)
    tcn.build()
    print(tcn.model.summary())

    print("\n=== Attention-LSTM Model ===")
    attn_lstm = AttentionLSTM(config)
    attn_lstm.build()
    print(attn_lstm.model.summary())

    print("\n=== GRU Model ===")
    gru = GRUModel(config)
    gru.build()
    print(gru.model.summary())
