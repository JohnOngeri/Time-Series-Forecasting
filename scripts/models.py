"""
Model architectures for air quality forecasting.
Implements RNN, LSTM, Stacked LSTM, and Bidirectional LSTM models.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, SimpleRNN, Dense, Dropout, 
    Bidirectional, BatchNormalization
)
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import numpy as np

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def build_model(config):
    """
    Build model based on configuration dictionary.
    
    Args:
        config (dict): Model configuration
        
    Returns:
        tf.keras.Model: Compiled model
    """
    model_type = config.get('model_type', 'lstm')
    input_shape = config.get('input_shape', (24, 10))
    units = config.get('units', 50)
    layers = config.get('layers', 1)
    dropout = config.get('dropout', 0.2)
    l2_reg = config.get('l2_reg', 0.01)
    activation = config.get('activation', 'relu')
    optimizer_name = config.get('optimizer', 'adam')
    learning_rate = config.get('learning_rate', 0.001)
    
    model = Sequential()
    
    if model_type == 'rnn':
        model = build_rnn_model(input_shape, units, layers, dropout, l2_reg, activation)
    elif model_type == 'lstm':
        model = build_lstm_model(input_shape, units, layers, dropout, l2_reg, activation)
    elif model_type == 'stacked_lstm':
        model = build_stacked_lstm_model(input_shape, units, layers, dropout, l2_reg, activation)
    elif model_type == 'bidirectional_lstm':
        model = build_bidirectional_lstm_model(input_shape, units, layers, dropout, l2_reg, activation)
    elif model_type == 'gru':
        model = build_gru_model(input_shape, units, layers, dropout, l2_reg, activation)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Get optimizer
    optimizer = get_optimizer(optimizer_name, learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    return model

def build_rnn_model(input_shape, units=50, layers=1, dropout=0.2, l2_reg=0.01, activation='relu'):
    """Build baseline RNN model."""
    model = Sequential()
    
    if layers == 1:
        model.add(SimpleRNN(
            units, 
            input_shape=input_shape,
            activation=activation,
            kernel_regularizer=l2(l2_reg)
        ))
    else:
        # First layer
        model.add(SimpleRNN(
            units, 
            input_shape=input_shape,
            return_sequences=True,
            activation=activation,
            kernel_regularizer=l2(l2_reg)
        ))
        model.add(Dropout(dropout))
        
        # Hidden layers
        for i in range(layers - 2):
            model.add(SimpleRNN(
                units,
                return_sequences=True,
                activation=activation,
                kernel_regularizer=l2(l2_reg)
            ))
            model.add(Dropout(dropout))
        
        # Last layer
        model.add(SimpleRNN(
            units,
            activation=activation,
            kernel_regularizer=l2(l2_reg)
        ))
    
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    return model

def build_lstm_model(input_shape, units=50, layers=1, dropout=0.2, l2_reg=0.01, activation='relu'):
    """Build simple LSTM model."""
    model = Sequential()
    
    if layers == 1:
        model.add(LSTM(
            units, 
            input_shape=input_shape,
            activation=activation,
            kernel_regularizer=l2(l2_reg)
        ))
    else:
        # First layer
        model.add(LSTM(
            units, 
            input_shape=input_shape,
            return_sequences=True,
            activation=activation,
            kernel_regularizer=l2(l2_reg)
        ))
        model.add(Dropout(dropout))
        
        # Hidden layers
        for i in range(layers - 2):
            model.add(LSTM(
                units,
                return_sequences=True,
                activation=activation,
                kernel_regularizer=l2(l2_reg)
            ))
            model.add(Dropout(dropout))
        
        # Last layer
        model.add(LSTM(
            units,
            activation=activation,
            kernel_regularizer=l2(l2_reg)
        ))
    
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    return model

def build_stacked_lstm_model(input_shape, units=50, layers=3, dropout=0.2, l2_reg=0.01, activation='relu'):
    """Build stacked LSTM model."""
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(
        units, 
        input_shape=input_shape,
        return_sequences=True,
        activation=activation,
        kernel_regularizer=l2(l2_reg)
    ))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    
    # Hidden LSTM layers
    for i in range(layers - 2):
        model.add(LSTM(
            units,
            return_sequences=True,
            activation=activation,
            kernel_regularizer=l2(l2_reg)
        ))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
    
    # Final LSTM layer
    model.add(LSTM(
        units,
        activation=activation,
        kernel_regularizer=l2(l2_reg)
    ))
    model.add(Dropout(dropout))
    
    # Dense layers
    model.add(Dense(units // 2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    return model

def build_bidirectional_lstm_model(input_shape, units=50, layers=2, dropout=0.2, l2_reg=0.01, activation='relu'):
    """Build bidirectional LSTM model."""
    model = Sequential()
    
    if layers == 1:
        model.add(Bidirectional(
            LSTM(units, activation=activation, kernel_regularizer=l2(l2_reg)),
            input_shape=input_shape
        ))
    else:
        # First bidirectional layer
        model.add(Bidirectional(
            LSTM(units, return_sequences=True, activation=activation, kernel_regularizer=l2(l2_reg)),
            input_shape=input_shape
        ))
        model.add(Dropout(dropout))
        
        # Hidden layers
        for i in range(layers - 2):
            model.add(Bidirectional(
                LSTM(units, return_sequences=True, activation=activation, kernel_regularizer=l2(l2_reg))
            ))
            model.add(Dropout(dropout))
        
        # Last layer
        model.add(Bidirectional(
            LSTM(units, activation=activation, kernel_regularizer=l2(l2_reg))
        ))
    
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    return model

def build_gru_model(input_shape, units=50, layers=1, dropout=0.2, l2_reg=0.01, activation='relu'):
    """Build GRU model."""
    model = Sequential()
    
    if layers == 1:
        model.add(GRU(
            units, 
            input_shape=input_shape,
            activation=activation,
            kernel_regularizer=l2(l2_reg)
        ))
    else:
        # First layer
        model.add(GRU(
            units, 
            input_shape=input_shape,
            return_sequences=True,
            activation=activation,
            kernel_regularizer=l2(l2_reg)
        ))
        model.add(Dropout(dropout))
        
        # Hidden layers
        for i in range(layers - 2):
            model.add(GRU(
                units,
                return_sequences=True,
                activation=activation,
                kernel_regularizer=l2(l2_reg)
            ))
            model.add(Dropout(dropout))
        
        # Last layer
        model.add(GRU(
            units,
            activation=activation,
            kernel_regularizer=l2(l2_reg)
        ))
    
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    return model

def get_optimizer(optimizer_name, learning_rate):
    """Get optimizer by name."""
    if optimizer_name.lower() == 'adam':
        return Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_callbacks(patience=10, min_delta=0.001, factor=0.5, min_lr=1e-7, 
                 checkpoint_path=None, monitor='val_loss'):
    """Get training callbacks."""
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Learning rate reduction
    lr_scheduler = ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience//2,
        min_lr=min_lr,
        verbose=1
    )
    callbacks.append(lr_scheduler)
    
    # Model checkpoint
    if checkpoint_path:
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
    
    return callbacks

def create_model_configs():
    """Create predefined model configurations for experiments."""
    configs = []
    
    # Baseline RNN
    configs.append({
        'name': 'baseline_rnn',
        'model_type': 'rnn',
        'units': 32,
        'layers': 1,
        'dropout': 0.2,
        'l2_reg': 0.01,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50
    })
    
    # Simple LSTM
    configs.append({
        'name': 'simple_lstm',
        'model_type': 'lstm',
        'units': 50,
        'layers': 1,
        'dropout': 0.2,
        'l2_reg': 0.01,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50
    })
    
    # Stacked LSTM
    configs.append({
        'name': 'stacked_lstm',
        'model_type': 'stacked_lstm',
        'units': 64,
        'layers': 3,
        'dropout': 0.3,
        'l2_reg': 0.01,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100
    })
    
    # Bidirectional LSTM
    configs.append({
        'name': 'bidirectional_lstm',
        'model_type': 'bidirectional_lstm',
        'units': 50,
        'layers': 2,
        'dropout': 0.25,
        'l2_reg': 0.01,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 75
    })
    
    # GRU model
    configs.append({
        'name': 'gru_model',
        'model_type': 'gru',
        'units': 64,
        'layers': 2,
        'dropout': 0.2,
        'l2_reg': 0.01,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50
    })
    
    # Hyperparameter variations
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [16, 32, 64]
    dropouts = [0.1, 0.2, 0.3]
    
    # Add variations for LSTM
    for lr in learning_rates:
        for bs in batch_sizes:
            for dp in dropouts:
                configs.append({
                    'name': f'lstm_lr{lr}_bs{bs}_dp{dp}',
                    'model_type': 'lstm',
                    'units': 50,
                    'layers': 2,
                    'dropout': dp,
                    'l2_reg': 0.01,
                    'optimizer': 'adam',
                    'learning_rate': lr,
                    'batch_size': bs,
                    'epochs': 50
                })
    
    return configs[:15]  # Return first 15 configurations

def plot_model_architecture(model, save_path='model_architecture.png'):
    """Plot and save model architecture."""
    try:
        tf.keras.utils.plot_model(
            model, 
            to_file=save_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96
        )
        print(f"Model architecture saved to {save_path}")
    except Exception as e:
        print(f"Could not save model architecture: {e}")
