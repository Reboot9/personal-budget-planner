import os
import pathlib
import warnings

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN
from tensorflow import keras

from predictions.services.main_model.utils import get_logger

from personal_budget_planner.constants import *


logger = get_logger('ModelTrainAndTest', log_file_name="model_prediction.log")

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    module="tcn.tcn"
)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DataPreprocessor:
    """
    Handles loading, scaling, splitting, and sequencing time series data for model training and forecasting.
    This version aims to replicate the exact data preparation logic of the original notebook.
    """

    def __init__(self,
                 window_size: int,
                 forecast_horizon: int,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 n_recursive_test_steps: int = 7):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.n_recursive_test_steps = n_recursive_test_steps

        # Scalers will be initialized and fitted in the data preparation logic
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        self.n_input_features = None
        self.n_target_categories = None
        self.target_cols = None  # List of target column names

        # Data attributes to be populated by prepare_all_data
        self.X_train_s, self.y_train_s = None, None
        self.X_val_s, self.y_val_s = None, None
        self.X_test_direct_eval_s, self.y_test_direct_eval_s = None, None

        # For recursive forecasting (scaled and raw outputs as in the original)
        self.X_initial_recursive_s, self.y_recursive_s = None, None  # Scaled
        self.X_initial_recursive_raw, self.y_recursive_raw_seq = None, None  # Raw

        self.y_train_raw_for_mase = None  # Unscaled y_train segment for MASE calculation

    def _create_sequences_from_set(self, X_data_scaled: np.ndarray,
                                   y_data_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates input-output sequences for time series forecasting from scaled data arrays.
        Uses instance attributes for window_size, forecast_horizon, n_input_features, n_target_categories.
        """
        X_sequences_list, y_sequences_list = [], []

        if self.n_input_features is None or self.n_target_categories is None:
            # This check ensures that prepare_all_data has first set these critical dimensions
            raise RuntimeError(
                "_create_sequences_from_set was called before n_input_features/n_target_categories were set.")

        num_possible_sequences = len(X_data_scaled) - self.window_size - self.forecast_horizon + 1

        if num_possible_sequences <= 0:
            return np.empty((0, self.window_size, self.n_input_features)), \
                np.empty((0, self.forecast_horizon, self.n_target_categories))

        for i in range(num_possible_sequences):
            X_sequences_list.append(X_data_scaled[i:(i + self.window_size)])
            y_sequences_list.append(
                y_data_scaled[(i + self.window_size):(i + self.window_size + self.forecast_horizon)])
        return np.array(X_sequences_list), np.array(y_sequences_list)

    def _execute_original_data_preparation_logic(self, x_all_raw: np.ndarray, y_all_raw: np.ndarray) -> dict:
        """
        This method contains the core data preparation logic, precisely following
        the original `prepare_time_series_data` function from the notebook.
        It uses instance attributes for configuration (e.g., self.window_size)
        and fits instance scalers (self.x_scaler, self.y_scaler).
        """
        if not (0 < self.train_ratio < 1 and 0 < self.val_ratio < 1 and (self.train_ratio + self.val_ratio) < 1):
            raise ValueError("train_ratio and val_ratio must be between 0 and 1, and their sum must be less than 1.")
        if self.n_recursive_test_steps < 1:
            raise ValueError("n_recursive_test_steps must be at least 1.")

        n_total_samples = x_all_raw.shape[0]

        # Chronological data split
        train_end_idx = int(n_total_samples * self.train_ratio)
        val_end_idx = train_end_idx + int(n_total_samples * self.val_ratio)

        X_train_raw_segment, y_train_raw_segment = x_all_raw[:train_end_idx], y_all_raw[:train_end_idx]
        X_val_raw_segment, y_val_raw_segment = x_all_raw[train_end_idx:val_end_idx], y_all_raw[
                                                                                     train_end_idx:val_end_idx]
        X_test_raw_segment, y_test_raw_segment = x_all_raw[val_end_idx:], y_all_raw[val_end_idx:]

        logger.info(' --- Data Segments (inside _execute_original_data_preparation_logic) --- ')
        logger.info(f'Training data X: {X_train_raw_segment.shape}, y: {y_train_raw_segment.shape}')
        logger.info(f'Validation data X: {X_val_raw_segment.shape}, y: {y_val_raw_segment.shape}')
        logger.info(f'Test data X: {X_test_raw_segment.shape}, y: {y_test_raw_segment.shape}')

        # Initializing and fitting scalers (using instance scalers)
        X_train_scaled = self.x_scaler.fit_transform(X_train_raw_segment)
        X_val_scaled = self.x_scaler.transform(X_val_raw_segment)
        X_test_scaled = self.x_scaler.transform(X_test_raw_segment)

        y_train_scaled = self.y_scaler.fit_transform(y_train_raw_segment)
        y_val_scaled = self.y_scaler.transform(y_val_raw_segment)
        y_test_scaled = self.y_scaler.transform(y_test_raw_segment)

        # Creating sequences using the instance method _create_sequences_from_set
        X_train_seq_scaled, y_train_seq_scaled = self._create_sequences_from_set(X_train_scaled, y_train_scaled)
        X_val_seq_scaled, y_val_seq_scaled = self._create_sequences_from_set(X_val_scaled, y_val_scaled)
        X_test_seq_all_scaled, y_test_seq_all_scaled = self._create_sequences_from_set(X_test_scaled, y_test_scaled)

        logger.info(
            ' --- Created Scaled Sequences (inside _execute_original_data_preparation_logic) --- ')
        logger.info(
            f'Training scaled sequences X: {X_train_seq_scaled.shape}, y: {y_train_seq_scaled.shape}')
        logger.info(f'Validation scaled sequences X: {X_val_seq_scaled.shape}, y: {y_val_seq_scaled.shape}')
        logger.info(
            f'Test scaled sequences X: {X_test_seq_all_scaled.shape}, y: {y_test_seq_all_scaled.shape}')

        # Preparation for recursive testing (original script logic)
        if X_test_seq_all_scaled.shape[0] < self.n_recursive_test_steps:
            warnings.warn(f"Not enough sequences in X_test_seq_all_scaled ({X_test_seq_all_scaled.shape[0]}) "
                          f"to select {self.n_recursive_test_steps} for recursive testing. "
                          f"Recursive test data will be empty.")
            X_initial_recursive_scaled = np.empty((0, self.window_size, self.n_input_features))
            y_recursive_scaled_seq = np.empty((0, self.forecast_horizon, self.n_target_categories))
            X_initial_recursive_raw = np.empty((0, self.window_size, self.n_input_features))
            y_recursive_raw_seq_list = np.empty(
                (0, self.forecast_horizon, self.n_target_categories))
        else:
            recursive_seq_start_idx_in_sequences = X_test_seq_all_scaled.shape[0] - self.n_recursive_test_steps

            # Bound check for X_initial_recursive_raw
            if recursive_seq_start_idx_in_sequences + self.window_size > X_test_raw_segment.shape[0]:
                warnings.warn(f"Index out of bounds for X_initial_recursive_raw. "
                              f"Start index {recursive_seq_start_idx_in_sequences} + window {self.window_size} > length of raw data segment {X_test_raw_segment.shape[0]}"
                              "X_initial_recursive_raw will be empty.")
                X_initial_recursive_raw = np.empty((0, self.window_size, self.n_input_features))
            else:
                X_initial_recursive_raw = X_test_raw_segment[
                                          recursive_seq_start_idx_in_sequences:
                                          recursive_seq_start_idx_in_sequences + self.window_size].copy()

            X_initial_recursive_scaled = X_test_seq_all_scaled[recursive_seq_start_idx_in_sequences].copy()
            y_recursive_scaled_seq = y_test_seq_all_scaled[recursive_seq_start_idx_in_sequences:].copy()

            # Building raw output sequences Y for each recursive step
            temp_y_recursive_raw_list = []
            for i in range(self.n_recursive_test_steps):
                y_raw_start_idx_for_step_i = recursive_seq_start_idx_in_sequences + self.window_size + i

                y_raw_end_idx_for_step_i = y_raw_start_idx_for_step_i + self.forecast_horizon

                if y_raw_end_idx_for_step_i > y_test_raw_segment.shape[0]:
                    y_seq_raw = y_test_raw_segment[y_raw_start_idx_for_step_i:]
                    warnings.warn(f"WARNING: Not enough raw Y data for step {i} of the recursive test, "
                                  f"requested up to index {y_raw_end_idx_for_step_i}, but length of y_test_raw_segment {y_test_raw_segment.shape[0]}. "
                                  f"Taken {len(y_seq_raw)} elements.")
                    if len(y_seq_raw) < self.forecast_horizon and len(
                            y_seq_raw) > 0:  # Pad if needed and possible
                        padding_needed = self.forecast_horizon - len(y_seq_raw)
                        # Simple padding with last value or NaNs. Using NaNs for clarity.
                        padding = np.full((padding_needed, y_test_raw_segment.shape[1]), np.nan)
                        y_seq_raw = np.vstack([y_seq_raw, padding])
                    elif len(y_seq_raw) == 0:
                        y_seq_raw = np.full((self.forecast_horizon, y_test_raw_segment.shape[1]),
                                            np.nan)  # Full NaN if no data
                else:
                    y_seq_raw = y_test_raw_segment[y_raw_start_idx_for_step_i:y_raw_end_idx_for_step_i]

                temp_y_recursive_raw_list.append(y_seq_raw)
            y_recursive_raw_seq_list = np.array(temp_y_recursive_raw_list)

        logger.info(' --- Recursive Testing (inside _execute_original_data_preparation_logic) ---')
        logger.info(f'Initial recursive scaled data X: {X_initial_recursive_scaled.shape}')
        logger.info(f'Initial recursive raw data X: {X_initial_recursive_raw.shape}')
        logger.info(f'Recursive scaled output Y: {y_recursive_scaled_seq.shape}')
        logger.info(f'Recursive raw output Y: {y_recursive_raw_seq_list.shape}')

        return {
            "train_set_scaled_seq": (X_train_seq_scaled, y_train_seq_scaled),
            "val_set_scaled_seq": (X_val_seq_scaled, y_val_seq_scaled),
            "test_direct_eval_scaled_seq": (X_test_seq_all_scaled, y_test_seq_all_scaled),
            "recursive_test_scaled_seq": (X_initial_recursive_scaled, y_recursive_scaled_seq),
            "recursive_test_raw_X_initial": X_initial_recursive_raw,  # Зберігання вихідного X для рекурсії
            "recursive_test_raw_y_seq_list": y_recursive_raw_seq_list,  # Зберігання списку вихідних Y для рекурсії
            "train_target_raw_segment": y_train_raw_segment,  # Для MASE
            "scalers": (self.x_scaler, self.y_scaler)
        }

    def prepare_all_data(self, x_all_raw: np.ndarray, y_all_raw: np.ndarray, target_cols: list):
        """
        Prepare all data required for training, validation, and recursive testing.
        This method encapsulates the original data preparation logic and returns the sequences as needed.
        """
        self.target_cols = target_cols
        self.n_input_features = x_all_raw.shape[1]
        self.n_target_categories = y_all_raw.shape[1]

        prepared_data_dict = self._execute_original_data_preparation_logic(x_all_raw, y_all_raw)

        self.X_train_s, self.y_train_s = prepared_data_dict["train_set_scaled_seq"]
        self.X_val_s, self.y_val_s = prepared_data_dict["val_set_scaled_seq"]
        self.X_test_direct_eval_s, self.y_test_direct_eval_s = prepared_data_dict["test_direct_eval_scaled_seq"]

        self.X_initial_recursive_s, self.y_recursive_s = prepared_data_dict["recursive_test_scaled_seq"]
        self.X_initial_recursive_raw = prepared_data_dict["recursive_test_raw_X_initial"]
        self.y_recursive_raw_seq = prepared_data_dict[
            "recursive_test_raw_y_seq_list"]
        self.y_train_raw_for_mase = prepared_data_dict["train_target_raw_segment"]

    def scale_input_for_prediction(self, x_raw_window: np.ndarray) -> np.ndarray:
        """Scales a raw input feature window for prediction and adds a batch dimension."""
        if self.n_input_features is None:
            raise RuntimeError("Preprocessor must first be fitted using prepare_all_data().")
        if not hasattr(self.x_scaler, 'data_min_') or self.x_scaler.data_min_ is None:
            raise RuntimeError("X scaler is not fitted. Call prepare_all_data() first.")
        if x_raw_window.shape != (self.window_size, self.n_input_features):
            raise ValueError(
                f"Input raw data must have shape ({self.window_size}, {self.n_input_features}), but got {x_raw_window.shape}")

        scaled_data = self.x_scaler.transform(x_raw_window)
        return np.expand_dims(scaled_data, axis=0)  # Shape (1, window_size, n_features)

    def inverse_transform_predictions(self, y_pred_scaled: np.ndarray) -> np.ndarray:
        """Inverse-transforms scaled predictions back to their original scale."""
        if self.n_target_categories is None:
            raise RuntimeError("Preprocessor must first be fitted using prepare_all_data().")
        if not hasattr(self.y_scaler, 'data_min_') or self.y_scaler.data_min_ is None:
            raise RuntimeError("Y scaler is not fitted. Call prepare_all_data() first.")

        original_shape = y_pred_scaled.shape

        # Handle different input dimensions of y_pred_scaled
        if y_pred_scaled.ndim == 3:  # (num_samples, forecast_horizon, n_outputs)
            reshaped_for_scaling = y_pred_scaled.reshape(-1, self.n_target_categories)
        elif y_pred_scaled.ndim == 2:  # (forecast_horizon_or_samples, n_outputs)
            reshaped_for_scaling = y_pred_scaled
        elif y_pred_scaled.ndim == 1:
            if self.n_target_categories == 1:
                reshaped_for_scaling = y_pred_scaled.reshape(-1, 1)
            else:
                reshaped_for_scaling = y_pred_scaled.reshape(1, self.n_target_categories)
        else:
            raise ValueError(f"Unsupported shape for y_pred_scaled: {y_pred_scaled.shape}")

        unscaled = self.y_scaler.inverse_transform(reshaped_for_scaling)

        if original_shape == unscaled.shape:
            return unscaled
        if y_pred_scaled.ndim == 3:  # (num_samples, forecast_horizon, n_outputs)
            return unscaled.reshape(original_shape)
        if y_pred_scaled.ndim == 2 and original_shape[0] == self.forecast_horizon and original_shape[
            1] == self.n_target_categories:
            return unscaled
        if y_pred_scaled.ndim == 1 and 1 < self.n_target_categories == len(y_pred_scaled):
            return unscaled.squeeze()
        if y_pred_scaled.ndim == 1 and self.n_target_categories == 1:
            return unscaled.squeeze()

        try:
            return unscaled.reshape(original_shape)
        except ValueError:
            warnings.warn(
                f"Could not reshape unscaled predictions from {unscaled.shape} back to original shape {original_shape}. Returning as {unscaled.shape}.")
            return unscaled

def build_model_architecture(hp: kt.HyperParameters,
                             window_size: int,
                             n_features: int,
                             forecast_horizon: int,
                             n_outputs: int) -> keras.Model:
    """Determines and compiles model architecture depending on hyperparams."""
    model = keras.Sequential()
    model.add(keras.Input(shape=(window_size, n_features)))

    num_rnn_layers = hp.Int('num_rnn_layers', 1, 3)
    for i in range(num_rnn_layers):
        rnn_units = hp.Int(f'rnn_units_layer_{i + 1}', min_value=32, max_value=256, step=32)
        is_last = (i == num_rnn_layers - 1)
        model.add(layers.LSTM(units=rnn_units, return_sequences=not is_last))

    tcn_filters = hp.Int('tcn_filters', min_value=32, max_value=128, step=32)
    tcn_kernel_size = hp.Int('tcn_kernel_size', min_value=2, max_value=7, step=1)
    tcn_nb_stacks = hp.Int('tcn_nb_stacks', min_value=1, max_value=2)
    dilation_choice = hp.Choice('tcn_dilation_set', values=['short', 'medium', 'long'])
    tcn_dilations = [1, 2, 4] if dilation_choice == 'short' else [1, 2, 4, 8] if dilation_choice == 'medium' else [1, 2, 4, 8, 16]
    
    model.add(TCN(nb_filters=tcn_filters, kernel_size=tcn_kernel_size, nb_stacks=tcn_nb_stacks,
                  dilations=tcn_dilations, padding='causal',
                  use_skip_connections=hp.Boolean('tcn_skip_connections', default=True),
                  dropout_rate=hp.Float('tcn_internal_dropout', min_value=0.0, max_value=0.3, step=0.05),
                  return_sequences=False))
    
    model.add(layers.Dense(units=forecast_horizon * n_outputs, activation='softplus'))
    model.add(layers.Reshape((forecast_horizon, n_outputs)))

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'adamw'])
    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'adamw':
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    else:  # 'rmsprop'
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mae',
                  metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model


class CustomHyperModel(kt.HyperModel):
    """Keras Tuner HyperModel wrapper for model-building function."""

    def __init__(self, window_size: int, n_features: int, forecast_horizon: int, n_outputs: int,
                 batch_size_min: int, batch_size_max: int, batch_size_step: int):
        self.window_size = window_size
        self.n_features = n_features
        self.forecast_horizon = forecast_horizon
        self.n_outputs = n_outputs
        self.batch_size_min = batch_size_min
        self.batch_size_max = batch_size_max
        self.batch_size_step = batch_size_step
        super().__init__()

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        return build_model_architecture(hp, self.window_size, self.n_features,
                                        self.forecast_horizon, self.n_outputs)

    def fit(self, hp, model, *args, **kwargs):
        """Fit method for Keras Tuner, includes batch_size as a hyperparameter for tuning."""
        batch_size = hp.Int('batch_size',
                            min_value=self.batch_size_min,
                            max_value=self.batch_size_max,
                            step=self.batch_size_step)
        return model.fit(*args, batch_size=batch_size, **kwargs)


class ModelTrainer:
    """Handles the model hyperparameter tuning and final training."""

    def __init__(self,
                 data_preprocessor: DataPreprocessor,
                 checkpoint_filepath: str = './final_trained_model.keras',
                 tuner_directory_path: str = '/.tuner_model',
                 batch_size_min: int = 16,
                 batch_size_max: int = 128,
                 batch_size_step: int = 16):
        self.dp = data_preprocessor
        if self.dp.n_input_features is None:
            raise ValueError(
                "DataPreprocessor must run prepare_all_data() before initializing ModelTrainer to set dimensions.")

        self.checkpoint_filepath = checkpoint_filepath
        self.tuner_directory_path = tuner_directory_path
        self.hypermodel = CustomHyperModel(
            window_size=self.dp.window_size,
            n_features=self.dp.n_input_features,
            forecast_horizon=self.dp.forecast_horizon,
            n_outputs=self.dp.n_target_categories,
            batch_size_min=batch_size_min,
            batch_size_max=batch_size_max,
            batch_size_step=batch_size_step
        )
        self.best_hps = None
        self.trained_model = None
        self.training_history = None

    def tune_hyperparameters(self, max_trials: int = 10, executions_per_trial: int = 1,
                             epochs: int = 20,
                             project_name: str = 'time_series_tuning'):
        """Performs hyperparameter search using Keras Tuner."""
        if self.dp.X_train_s is None or self.dp.X_val_s is None:
            raise RuntimeError("Data preprocessor has not prepared training/validation data.")

        tuner = kt.RandomSearch(
            self.hypermodel,
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=self.tuner_directory_path,
            project_name=project_name,
            overwrite=True
        )
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, verbose=1, restore_best_weights=True
        )
        logger.info("Starting hyperparameter tuning...")
        tuner.search(
            self.dp.X_train_s, self.dp.y_train_s,
            epochs=epochs,
            validation_data=(self.dp.X_val_s, self.dp.y_val_s),
            callbacks=[early_stopping_callback],
            verbose=1
        )
        self.best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info("\n--- Best hyperparameters found ---")
        for hp_name in self.best_hps.values:  # Iterate through actual hyperparameter names
            logger.info(f"- {hp_name}: {self.best_hps.get(hp_name)}")

        temp_best_model = self.hypermodel.build(self.best_hps)  # Build with the best hyperparameters
        logger.info("\nSummary of the model built with the best hyperparameters:")
        temp_best_model.summary(print_fn=logger.info)  # Direct summary to logger
        return self.best_hps

    def train_final_model(self, final_train_epochs: int = 150, early_stopping_patience: int = 20):
        """Trains the final model using the best-found hyperparameters."""
        if self.best_hps is None:
            raise RuntimeError("Hyperparameters have not been tuned. Call tune_hyperparameters first.")

        logger.info("\nBuilding and training the final model with the best hyperparameters...")
        self.trained_model = build_model_architecture(  # Using a separate function
            self.best_hps,
            window_size=self.dp.window_size,
            n_features=self.dp.n_input_features,
            forecast_horizon=self.dp.forecast_horizon,
            n_outputs=self.dp.n_target_categories
        )
        best_batch_size = self.best_hps.get('batch_size')
        if best_batch_size is None:
            warnings.warn("Batch size not found in best_hps from tuner, defaulting to 32.")
            best_batch_size = 32

        callbacks_list = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1,
                                          restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_filepath, monitor='val_loss', save_best_only=True,
                                            save_weights_only=False, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7, verbose=1)
        ]
        logger.info(f"\nStarting final training for {final_train_epochs} epochs with batch_size={best_batch_size}...")
        self.training_history = self.trained_model.fit(
            self.dp.X_train_s, self.dp.y_train_s,
            epochs=final_train_epochs,
            batch_size=best_batch_size,
            validation_data=(self.dp.X_val_s, self.dp.y_val_s),
            callbacks=callbacks_list,
            verbose=1
        )
        logger.info(f"\nFinal training completed.")
        if os.path.exists(self.checkpoint_filepath):
            logger.info(f"The best model during training was saved at {self.checkpoint_filepath}.")
            self.load_model(
                self.checkpoint_filepath)
        else:
            warnings.warn(f"Checkpoint file {self.checkpoint_filepath} was not created. "
                          "The in-memory model may not be the best if early stopping occurred before the first save.")
        return self.trained_model

    def load_model(self, filepath: str):
        """Loads a previously trained model."""
        logger.info(f"Loading model from {filepath}...")

        custom_objects = {
            'TCN': TCN
        }
        self.trained_model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
        logger.info("Model successfully loaded.")
        logger.info("Summary of the loaded model:")
        self.trained_model.summary(print_fn=logger.info)
        return self.trained_model


class Predictor:
    """Handles prediction creation and evaluation."""

    def __init__(self, model: keras.Model, data_preprocessor: DataPreprocessor):
        if not isinstance(model, keras.Model):
            raise ValueError("The model must be an instance of keras.Model.")
        if not isinstance(data_preprocessor, DataPreprocessor):
            raise ValueError("The data preprocessor must be an instance of DataPreprocessor.")

        self.model = model
        self.dp = data_preprocessor

    def predict_one_day(self, raw_data_for_window: np.ndarray) -> np.ndarray:
        """
        Makes a single prediction for the next forecast period (forecast_horizon).
        """
        if raw_data_for_window.shape != (self.dp.window_size, self.dp.n_input_features):
            raise ValueError(
                f"Input raw_data_for_window must have shape ({self.dp.window_size}, {self.dp.n_input_features}), "
                f"but got {raw_data_for_window.shape}")

        scaled_input_sequence = self.dp.scale_input_for_prediction(raw_data_for_window)
        scaled_prediction = self.model.predict(scaled_input_sequence, verbose=0)[0]  # Get (forecast_horizon, n_outputs)
        unscaled_prediction = self.dp.inverse_transform_predictions(scaled_prediction)
        return unscaled_prediction

    def predict_recursively(self, initial_raw_window_data: np.ndarray,
                            num_recursive_steps: int,
                            true_future_exogenous_raw: np.ndarray = None) -> np.ndarray:
        """
        Makes recursive predictions for the specified number of steps.
        """
        if initial_raw_window_data.shape != (self.dp.window_size, self.dp.n_input_features):
            raise ValueError(
                f"initial_raw_window_data must have shape ({self.dp.window_size}, {self.dp.n_input_features})")

        n_exo_features = self.dp.n_input_features - self.dp.n_target_categories
        if n_exo_features > 0:
            if true_future_exogenous_raw is None:
                raise ValueError(
                    f"The model has {n_exo_features} exogenous features. true_future_exogenous_raw must be provided.")
            if true_future_exogenous_raw.shape[
                0] < num_recursive_steps - 1:  # Needs enough rows for all except the last step
                raise ValueError(
                    f"true_future_exogenous_raw requires {num_recursive_steps - 1} rows for exogenous features, "
                    f"but got {true_future_exogenous_raw.shape[0]}")
            if true_future_exogenous_raw.shape[1] != n_exo_features:
                raise ValueError(
                    f"true_future_exogenous_raw must have {n_exo_features} columns, but got {true_future_exogenous_raw.shape[1]}.")

        current_scaled_input_sequence = self.dp.x_scaler.transform(initial_raw_window_data)
        all_recursive_predictions_scaled_list = []

        for i in range(num_recursive_steps):
            model_input = np.expand_dims(current_scaled_input_sequence, axis=0)
            predicted_step_scaled_fh = self.model.predict(model_input, verbose=0)[
                0]  # Shape (forecast_horizon, n_outputs)
            all_recursive_predictions_scaled_list.append(predicted_step_scaled_fh)

            if i < num_recursive_steps - 1:
                if self.dp.forecast_horizon != 1:
                    warnings.warn(
                        "The recursive prediction logic currently assumes forecast_horizon=1 "
                        "for extracting the targets of the next step. Adaptation is needed for >1.")

                new_target_values_scaled = predicted_step_scaled_fh[0, :]

                new_last_row_features_scaled = np.zeros(self.dp.n_input_features)
                new_last_row_features_scaled[:self.dp.n_target_categories] = new_target_values_scaled

                if n_exo_features > 0:
                    raw_exo_for_next_step_input = true_future_exogenous_raw[i]  # (n_exo_features,)
                    temp_full_row_for_scaling_exo = np.zeros((1, self.dp.n_input_features))

                    placeholder_targets = np.mean(current_scaled_input_sequence[:, :self.dp.n_target_categories],
                                                  axis=0)
                    temp_full_row_for_scaling_exo[0,
                    :self.dp.n_target_categories] = placeholder_targets  # Use scaled means
                    temp_full_row_for_scaling_exo[0, self.dp.n_target_categories:] = raw_exo_for_next_step_input

                    scaled_full_row = self.dp.x_scaler.transform(temp_full_row_for_scaling_exo)
                    new_last_row_features_scaled[self.dp.n_target_categories:] = scaled_full_row[0,
                                                                                 self.dp.n_target_categories:]

                current_scaled_input_sequence = np.roll(current_scaled_input_sequence, -1, axis=0)
                current_scaled_input_sequence[-1, :] = new_last_row_features_scaled

        all_recursive_predictions_scaled_arr = np.array(all_recursive_predictions_scaled_list)
        # Shape: (num_recursive_steps, forecast_horizon, n_outputs)

        # Inverse transformation of all predictions at once
        num_steps, fh, n_out = all_recursive_predictions_scaled_arr.shape
        reshaped_for_inverse = all_recursive_predictions_scaled_arr.reshape(num_steps * fh, n_out)
        unscaled_predictions_flat = self.dp.y_scaler.inverse_transform(reshaped_for_inverse)

        return unscaled_predictions_flat.reshape(num_steps, fh, n_out)

    @staticmethod
    def _calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(y_true - y_pred), axis=0)

    @staticmethod
    def _calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

    @staticmethod
    def _calculate_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train_unscaled: np.ndarray,
                        seasonality_period: int = 1) -> np.ndarray:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        y_train_unscaled_arr = np.asarray(y_train_unscaled)

        # Check dimensions of the data, 2D: (samples, features_or_targets)
        if y_train_unscaled_arr.ndim == 1: y_train_unscaled_arr = y_train_unscaled_arr.reshape(-1, 1)
        if y_true_arr.ndim == 1: y_true_arr = y_true_arr.reshape(-1, 1)
        if y_pred_arr.ndim == 1: y_pred_arr = y_pred_arr.reshape(-1, 1)

        if y_true_arr.shape[1] != y_train_unscaled_arr.shape[1] or y_pred_arr.shape[1] != y_train_unscaled_arr.shape[1]:
            warnings.warn(
                f"MASE: Mismatch in columns: y_true({y_true_arr.shape}), y_pred({y_pred_arr.shape}), y_train({y_train_unscaled_arr.shape})")
            return np.full(y_true_arr.shape[1] if y_true_arr.ndim > 1 else 1, np.nan)

        forecast_errors = np.abs(y_true_arr - y_pred_arr)
        mean_absolute_forecast_error = np.mean(forecast_errors, axis=0)  # Mean over time steps for each target

        if len(y_train_unscaled_arr) <= seasonality_period:
            warnings.warn(
                f"MASE: Length of y_train_unscaled ({len(y_train_unscaled_arr)}) <= seasonality period ({seasonality_period}).")
            return np.full(y_true_arr.shape[1] if y_true_arr.ndim > 1 else 1, np.nan)

        naive_forecast_errors_train = np.abs(
            y_train_unscaled_arr[seasonality_period:] - y_train_unscaled_arr[:-seasonality_period])
        mean_absolute_naive_error_train = np.mean(naive_forecast_errors_train, axis=0)

        mase_scores = np.full_like(mean_absolute_forecast_error, np.nan)
        non_zero_denom_mask = mean_absolute_naive_error_train > 1e-9  # Avoid division by zero or very small number
        mase_scores[non_zero_denom_mask] = mean_absolute_forecast_error[non_zero_denom_mask] / \
                                           mean_absolute_naive_error_train[non_zero_denom_mask]
        return mase_scores

    def evaluate_one_day_prediction(self,
                                    y_true_unscaled: np.ndarray,
                                    y_pred_unscaled: np.ndarray) -> dict:
        """
        Evaluates one or more unscaled predictions against the true values.
        """
        if y_true_unscaled.shape != y_pred_unscaled.shape:
            raise ValueError(
                f"y_true_unscaled ({y_true_unscaled.shape}) and y_pred_unscaled ({y_pred_unscaled.shape}) must have the same shape.")
        if y_true_unscaled.ndim == 1 and self.dp.n_target_categories > 1:  # (n_outputs,)
            y_true_unscaled = y_true_unscaled.reshape(1, -1)
            y_pred_unscaled = y_pred_unscaled.reshape(1, -1)
        elif y_true_unscaled.ndim == 1 and self.dp.n_target_categories == 1:  # (horizon_steps,)
            y_true_unscaled = y_true_unscaled.reshape(-1, 1)
            y_pred_unscaled = y_pred_unscaled.reshape(-1, 1)

        if y_true_unscaled.shape[1] != self.dp.n_target_categories:
            raise ValueError(
                f"The number of columns ({y_true_unscaled.shape[1]}) must match n_target_categories ({self.dp.n_target_categories})")
        if self.dp.y_train_raw_for_mase is None:
            raise RuntimeError(
                "y_train_raw_for_mase is not available in DataPreprocessor. Please run prepare_all_data.")

        metrics_summary = {}
        mae_scores = self._calculate_mae(y_true_unscaled, y_pred_unscaled)
        rmse_scores = self._calculate_rmse(y_true_unscaled, y_pred_unscaled)
        mase_scores = self._calculate_mase(y_true_unscaled, y_pred_unscaled, self.dp.y_train_raw_for_mase)

        for i, col_name in enumerate(self.dp.target_cols):
            metrics_summary[col_name] = {
                'MAE': mae_scores[i] if mae_scores.ndim > 0 else mae_scores,
                'RMSE': rmse_scores[i] if rmse_scores.ndim > 0 else rmse_scores,
                'MASE': mase_scores[i] if mase_scores.ndim > 0 else mase_scores
            }

        metrics_summary['Average'] = {
            'MAE': np.nanmean(mae_scores),
            'RMSE': np.nanmean(rmse_scores),
            'MASE': np.nanmean(mase_scores)
        }
        return metrics_summary

    def evaluate_on_test_set(self):
        """Evaluates the model on the predefined test set for direct evaluation."""
        if self.dp.X_test_direct_eval_s is None or self.dp.X_test_direct_eval_s.shape[0] == 0:
            logger.info("Test set for direct evaluation is empty or not prepared. Skipping evaluation.")
            return None

        direct_predictions_norm = self.model.predict(self.dp.X_test_direct_eval_s, verbose=0)
        num_samples, fh, n_out = direct_predictions_norm.shape

        direct_predictions_unscaled = self.dp.inverse_transform_predictions(direct_predictions_norm)
        y_direct_eval_unscaled = self.dp.inverse_transform_predictions(self.dp.y_test_direct_eval_s)

        y_pred_eval = direct_predictions_unscaled.reshape(num_samples * fh, n_out)
        y_true_eval = y_direct_eval_unscaled.reshape(num_samples * fh, n_out)

        return self.evaluate_one_day_prediction(y_true_eval, y_pred_eval)


def main():
    file_path_data_folder = pathlib.Path(__file__).parents[3].resolve() / 'data/intermediate'

    print("Loading data...")
    X_train_df = pd.read_pickle(file_path_data_folder / 'X_train.pkl')
    X_test_df = pd.read_pickle(file_path_data_folder / 'X_test.pkl')
    y_train_df = pd.read_pickle(file_path_data_folder / 'y_train.pkl')
    y_test_df = pd.read_pickle(file_path_data_folder / 'y_test.pkl')

    X_raw_main = pd.concat([X_train_df, X_test_df], axis=0).to_numpy()
    y_raw_main = pd.concat([y_train_df, y_test_df], axis=0).to_numpy()
    target_cols_main = y_train_df.columns.tolist()
    print(f'Initial shape of X: {X_raw_main.shape}, Initial shape of y: {y_raw_main.shape}, Targets: {target_cols_main}')

    # 1. Data Preprocessing
    print("\n--- Initializing DataPreprocessor ---")
    data_preprocessor = DataPreprocessor(
        window_size=WINDOW_SIZE_MAIN, forecast_horizon=FORECAST_HORIZON_MAIN,
        train_ratio=TRAIN_RATIO_MAIN, val_ratio=VAL_RATIO_MAIN,
        n_recursive_test_steps=N_RECURSIVE_TEST_STEPS_MAIN
    )
    data_preprocessor.prepare_all_data(X_raw_main, y_raw_main, target_cols_main)

    # 2. Model Training
    print("\n--- Initializing ModelTrainer ---")

    model_checkpoint = file_path_data_folder / f'model/{CHECKPOINT_FILEPATH_MAIN}'
    tuner_directory = file_path_data_folder / f'keras_tuner'

    model_trainer = ModelTrainer(data_preprocessor=data_preprocessor,
                                 checkpoint_filepath=str(model_checkpoint.resolve()),
                                 tuner_directory_path=str(tuner_directory.resolve()))

    train_new_model = True
    if os.path.exists(model_checkpoint):
        user_choice = input(
            f"Existing model found at {model_checkpoint}. Load it? (y/n, default y): ").lower()
        if user_choice == '' or user_choice == 'y':
            try:
                print("Attempting to load the existing model...")
                model_trainer.load_model(str(model_checkpoint.resolve()))
                if model_trainer.trained_model:
                    print("Model successfully loaded. Skipping tuning and training.")
                    train_new_model = False
            except Exception as e:
                print(f"Error loading existing model: {e}. Retraining will proceed.")
                if os.path.exists(model_checkpoint): os.remove(
                    str(model_checkpoint.resolve()))
        else:
            print("Proceeding with retraining the model.")

    if train_new_model:
        print("\n--- Hyperparameter Tuning ---")
        model_trainer.tune_hyperparameters(max_trials=TUNER_MAX_TRIALS_MAIN, epochs=TUNER_EPOCHS_MAIN)
        print("\n--- Training the Final Model ---")
        model_trainer.train_final_model(final_train_epochs=FINAL_TRAIN_EPOCHS_MAIN)

    if model_trainer.trained_model is None:
        print("ERROR: Failed to train or load the model. Exiting.")
        exit()

    # 3. Prediction and Evaluation
    print("\n--- Initializing Predictor ---")
    predictor = Predictor(model=model_trainer.trained_model, data_preprocessor=data_preprocessor)

    # Example: Recursive Prediction
    num_steps_for_recursion = N_RECURSIVE_TEST_STEPS_MAIN
    required_raw_len_for_recursion = WINDOW_SIZE_MAIN + (
            num_steps_for_recursion - 1) + FORECAST_HORIZON_MAIN  # Maximum required index

    if len(X_raw_main) >= required_raw_len_for_recursion:
        print(f"\n--- Recursive prediction for {num_steps_for_recursion} steps ---")
        # Initial point for the initial window
        rec_input_start_idx = len(X_raw_main) - required_raw_len_for_recursion
        if rec_input_start_idx < 0: rec_input_start_idx = 0

        initial_raw_window_for_rec = X_raw_main[rec_input_start_idx: rec_input_start_idx + WINDOW_SIZE_MAIN]

        true_future_exo_raw_main = None
        n_exo_main = data_preprocessor.n_input_features - data_preprocessor.n_target_categories
        if n_exo_main > 0:
            exo_start_idx = rec_input_start_idx + WINDOW_SIZE_MAIN
            exo_end_idx = exo_start_idx + (num_steps_for_recursion - 1)
            if exo_end_idx <= len(X_raw_main):  # Check if we are not going beyond X_raw_main
                true_future_exo_raw_main = X_raw_main[exo_start_idx: exo_end_idx,
                                           data_preprocessor.n_target_categories:]
            else:
                print(
                    f"Warning: Insufficient future data for all {n_exo_main} exogenous features. Skipping recursive prediction.")
                num_steps_for_recursion = 0

        if num_steps_for_recursion > 0:
            recursive_predictions_unscaled = predictor.predict_recursively(
                initial_raw_window_for_rec,
                num_recursive_steps=num_steps_for_recursion,
                true_future_exogenous_raw=true_future_exo_raw_main
            )
            print(
                f"Unscaled recursive predictions (shape {recursive_predictions_unscaled.shape}):\n{recursive_predictions_unscaled}")

            reshaped_recursive_predictions_unscaled = recursive_predictions_unscaled.reshape(N_RECURSIVE_TEST_STEPS_MAIN, 5)
            predict_json = {f'Day {n + 1}': dict(zip(target_cols_main, reshaped_recursive_predictions_unscaled[n])) for
                            n in range(N_RECURSIVE_TEST_STEPS_MAIN)}
            print(f'Predictions in nested JSON format: {predict_json}')
            # True values for these recursive steps
            y_true_rec_start = rec_input_start_idx + WINDOW_SIZE_MAIN
            # Ensure y_true_rec_end does not exceed the bounds of y_raw_main
            max_possible_y_elements = num_steps_for_recursion * FORECAST_HORIZON_MAIN
            available_y_elements = len(y_raw_main) - y_true_rec_start

            elements_to_take = min(max_possible_y_elements, available_y_elements)
            actual_num_recursive_steps_for_eval = elements_to_take // FORECAST_HORIZON_MAIN

            if actual_num_recursive_steps_for_eval < num_steps_for_recursion:
                print(
                    f"Warning: Fewer true Y values available ({actual_num_recursive_steps_for_eval} steps) for evaluating recursive predictions than predicted ({num_steps_for_recursion} steps). Evaluation will be for available data.")
                # Trim predictions to match available true values
                recursive_predictions_unscaled = recursive_predictions_unscaled[:actual_num_recursive_steps_for_eval]

            if actual_num_recursive_steps_for_eval > 0:
                y_true_rec_end = y_true_rec_start + actual_num_recursive_steps_for_eval * FORECAST_HORIZON_MAIN
                y_true_recursive_raw = y_raw_main[y_true_rec_start: y_true_rec_end].reshape(
                    actual_num_recursive_steps_for_eval, FORECAST_HORIZON_MAIN, data_preprocessor.n_target_categories
                )

                # Evaluate all recursive steps together
                metrics_recursive_all = predictor.evaluate_one_day_prediction(
                    y_true_recursive_raw.reshape(-1, data_preprocessor.n_target_categories),
                    # Flatten time steps
                    recursive_predictions_unscaled.reshape(-1, data_preprocessor.n_target_categories)
                )
                print(f"Metrics for all {actual_num_recursive_steps_for_eval} recursive steps (averaged):")
                for cat, mets in metrics_recursive_all.items(): print(f"  {cat}: {mets}")
            else:
                print("Not enough data for evaluating recursive predictions.")

        else:
            print("Not enough source data for recursive prediction example.")

        print("\n--- Evaluating model on internal test set (Direct Evaluation) ---")
        test_set_metrics = predictor.evaluate_on_test_set()
        if test_set_metrics:
            print(
                "Metrics for internal test set (direct one-step predictions, averaged by samples/horizon):")
            for cat, mets in test_set_metrics.items(): print(f"  {cat}: {mets}")

if __name__ == '__main__':
    main()
