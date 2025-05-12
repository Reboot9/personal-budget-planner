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
    Обробляє завантаження, масштабування, розділення та секвенування даних часових рядів
    для навчання моделі та прогнозування. Ця версія має на меті точно відтворити
    логіку підготовки даних оригінального ноутбука.
    """

    def __init__(self,
                 window_size: int,
                 forecast_horizon: int,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 n_recursive_test_steps: int = 5):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.n_recursive_test_steps = n_recursive_test_steps

        # Масштабувальники будуть ініціалізовані та навчені в логіці підготовки
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        self.n_input_features = None
        self.n_target_categories = None
        self.target_cols = None  # Список назв цільових колонок

        # Атрибути даних, які будуть заповнені prepare_all_data
        self.X_train_s, self.y_train_s = None, None
        self.X_val_s, self.y_val_s = None, None
        self.X_test_direct_eval_s, self.y_test_direct_eval_s = None, None

        # Для рекурсивного прогнозування (масштабовані та вихідні, як в оригіналі)
        self.X_initial_recursive_s, self.y_recursive_s = None, None  # Масштабовані
        self.X_initial_recursive_raw, self.y_recursive_raw_seq = None, None  # Вихідні

        self.y_train_raw_for_mase = None  # Немасштабований сегмент y_train для розрахунку MASE

    def _create_sequences_from_set(self, X_data_scaled: np.ndarray,
                                   y_data_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Створює вхідні-вихідні послідовності для прогнозування часових рядів з масштабованих масивів даних.
        Використовує атрибути екземпляра для window_size, forecast_horizon, n_input_features, n_target_categories.
        """
        X_sequences_list, y_sequences_list = [], []

        if self.n_input_features is None or self.n_target_categories is None:
            # Ця перевірка гарантує, що prepare_all_data спочатку встановила ці критичні розмірності
            raise RuntimeError(
                "_create_sequences_from_set викликано до того, як були встановлені n_input_features/n_target_categories.")

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
        Цей метод містить основну логіку підготовки даних, точно наслідуючи
        оригінальну функцію `prepare_time_series_data` з ноутбука.
        Він використовує атрибути екземпляра для конфігурації (наприклад, self.window_size)
        та навчає масштабувальники екземпляра (self.x_scaler, self.y_scaler).
        """
        if not (0 < self.train_ratio < 1 and 0 < self.val_ratio < 1 and (self.train_ratio + self.val_ratio) < 1):
            raise ValueError("train_ratio та val_ratio повинні бути між 0 та 1, а їх сума меншою за 1.")
        if self.n_recursive_test_steps < 1:
            raise ValueError("n_recursive_test_steps повинен бути щонайменше 1.")

        n_total_samples = x_all_raw.shape[0]

        # Хронологічне розділення даних
        train_end_idx = int(n_total_samples * self.train_ratio)
        val_end_idx = train_end_idx + int(n_total_samples * self.val_ratio)

        X_train_raw_segment, y_train_raw_segment = x_all_raw[:train_end_idx], y_all_raw[:train_end_idx]
        X_val_raw_segment, y_val_raw_segment = x_all_raw[train_end_idx:val_end_idx], y_all_raw[
                                                                                     train_end_idx:val_end_idx]
        X_test_raw_segment, y_test_raw_segment = x_all_raw[val_end_idx:], y_all_raw[val_end_idx:]

        logger.info(' --- Сегменти вихідних даних (всередині _execute_original_data_preparation_logic) --- ')
        logger.info(f'Тренувальні вихідні X: {X_train_raw_segment.shape}, y: {y_train_raw_segment.shape}')
        logger.info(f'Валідаційні вихідні X: {X_val_raw_segment.shape}, y: {y_val_raw_segment.shape}')
        logger.info(f'Тестові вихідні X: {X_test_raw_segment.shape}, y: {y_test_raw_segment.shape}')

        # Ініціалізація та навчання масштабувальників (використовуючи масштабувальники екземпляра)
        X_train_scaled = self.x_scaler.fit_transform(X_train_raw_segment)
        X_val_scaled = self.x_scaler.transform(X_val_raw_segment)
        X_test_scaled = self.x_scaler.transform(X_test_raw_segment)

        y_train_scaled = self.y_scaler.fit_transform(y_train_raw_segment)
        y_val_scaled = self.y_scaler.transform(y_val_raw_segment)
        y_test_scaled = self.y_scaler.transform(y_test_raw_segment)

        # Створення послідовностей за допомогою методу екземпляра _create_sequences_from_set
        X_train_seq_scaled, y_train_seq_scaled = self._create_sequences_from_set(X_train_scaled, y_train_scaled)
        X_val_seq_scaled, y_val_seq_scaled = self._create_sequences_from_set(X_val_scaled, y_val_scaled)
        X_test_seq_all_scaled, y_test_seq_all_scaled = self._create_sequences_from_set(X_test_scaled, y_test_scaled)

        logger.info(
            ' --- Створені масштабовані послідовності (всередині _execute_original_data_preparation_logic) --- ')
        logger.info(
            f'Тренувальні масштабовані послідовності X: {X_train_seq_scaled.shape}, y: {y_train_seq_scaled.shape}')
        logger.info(f'Валідаційні масштабовані послідовності X: {X_val_seq_scaled.shape}, y: {y_val_seq_scaled.shape}')
        logger.info(
            f'Тестові масштабовані послідовності X: {X_test_seq_all_scaled.shape}, y: {y_test_seq_all_scaled.shape}')

        # Підготовка до рекурсивного тестування (логіка з оригінального скрипта)
        if X_test_seq_all_scaled.shape[0] < self.n_recursive_test_steps:
            warnings.warn(f"Недостатньо послідовностей у X_test_seq_all_scaled ({X_test_seq_all_scaled.shape[0]}) "
                          f"щоб вибрати {self.n_recursive_test_steps} для рекурсивного тестування. "
                          f"Дані рекурсивного тесту будуть порожніми.")
            X_initial_recursive_scaled = np.empty((0, self.window_size, self.n_input_features))
            y_recursive_scaled_seq = np.empty((0, self.forecast_horizon, self.n_target_categories))
            X_initial_recursive_raw = np.empty((0, self.window_size, self.n_input_features))
            y_recursive_raw_seq_list = np.empty(
                (0, self.forecast_horizon, self.n_target_categories))
        else:
            recursive_seq_start_idx_in_sequences = X_test_seq_all_scaled.shape[0] - self.n_recursive_test_steps

            # Перевірка меж для X_initial_recursive_raw
            if recursive_seq_start_idx_in_sequences + self.window_size > X_test_raw_segment.shape[0]:
                warnings.warn(f"Індекс поза межами для X_initial_recursive_raw. "
                              f"Початковий індекс вихідних даних {recursive_seq_start_idx_in_sequences} + вікно {self.window_size} > довжина сегменту вихідних даних {X_test_raw_segment.shape[0]}"
                              "X_initial_recursive_raw буде порожнім.")
                X_initial_recursive_raw = np.empty((0, self.window_size, self.n_input_features))
            else:
                X_initial_recursive_raw = X_test_raw_segment[
                                          recursive_seq_start_idx_in_sequences:
                                          recursive_seq_start_idx_in_sequences + self.window_size].copy()

            X_initial_recursive_scaled = X_test_seq_all_scaled[recursive_seq_start_idx_in_sequences].copy()
            y_recursive_scaled_seq = y_test_seq_all_scaled[recursive_seq_start_idx_in_sequences:].copy()

            # Побудова вихідних послідовностей Y для кожного рекурсивного кроку
            temp_y_recursive_raw_list = []
            for i in range(self.n_recursive_test_steps):
                y_raw_start_idx_for_step_i = recursive_seq_start_idx_in_sequences + self.window_size + i

                y_raw_end_idx_for_step_i = y_raw_start_idx_for_step_i + self.forecast_horizon

                if y_raw_end_idx_for_step_i > y_test_raw_segment.shape[0]:
                    y_seq_raw = y_test_raw_segment[y_raw_start_idx_for_step_i:]
                    warnings.warn(f"ПОПЕРЕДЖЕННЯ: Недостатньо вихідних Y для кроку {i} рекурсивного тесту, "
                                  f"запитано до індексу {y_raw_end_idx_for_step_i}, але довжина y_test_raw_segment {y_test_raw_segment.shape[0]}. "
                                  f"Взято {len(y_seq_raw)} елементів.")
                    if len(y_seq_raw) < self.forecast_horizon and len(
                            y_seq_raw) > 0:  # Доповнити, якщо необхідно і можливо
                        padding_needed = self.forecast_horizon - len(y_seq_raw)
                        # Просте доповнення останнім значенням або NaNs. Використання NaNs для ясності.
                        padding = np.full((padding_needed, y_test_raw_segment.shape[1]), np.nan)
                        y_seq_raw = np.vstack([y_seq_raw, padding])
                    elif len(y_seq_raw) == 0:
                        y_seq_raw = np.full((self.forecast_horizon, y_test_raw_segment.shape[1]),
                                            np.nan)  # Повністю NaN, якщо немає даних
                else:
                    y_seq_raw = y_test_raw_segment[y_raw_start_idx_for_step_i:y_raw_end_idx_for_step_i]

                temp_y_recursive_raw_list.append(y_seq_raw)
            y_recursive_raw_seq_list = np.array(temp_y_recursive_raw_list)

        logger.info(' --- Для рекурсивного тестування (всередині _execute_original_data_preparation_logic) ---')
        logger.info(f'Вихідний X початковий для рекурсії: {X_initial_recursive_raw.shape}')
        logger.info(f'Масштабований X початковий для рекурсії: {X_initial_recursive_scaled.shape}')
        logger.info(f'Вихідні Y послідовності для рекурсії: {y_recursive_raw_seq_list.shape}')
        logger.info(f'Масштабовані Y послідовності для рекурсії: {y_recursive_scaled_seq.shape}')

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
        Основний публічний метод для підготовки всіх даних. Він викликає внутрішню логіку
        та заповнює атрибути екземпляра.
        """
        self.target_cols = target_cols
        # Встановлюємо розмірності спочатку, оскільки вони використовуються _create_sequences_from_set
        self.n_input_features = x_all_raw.shape[1]
        self.n_target_categories = y_all_raw.shape[1]

        logger.info(f"--- DataPreprocessor: викликано prepare_all_data ---")
        logger.info(f"Форма вхідних вихідних X: {x_all_raw.shape}, Форма вхідних вихідних Y: {y_all_raw.shape}")
        logger.info(
            f"Вікно: {self.window_size}, Горизонт: {self.forecast_horizon}, Тренування: {self.train_ratio}, Валідація: {self.val_ratio}")

        # Виклик методу, що містить оригінальну логіку
        prepared_data_dict = self._execute_original_data_preparation_logic(x_all_raw, y_all_raw)

        # Розпакування словника в атрибути екземпляра
        self.X_train_s, self.y_train_s = prepared_data_dict["train_set_scaled_seq"]
        self.X_val_s, self.y_val_s = prepared_data_dict["val_set_scaled_seq"]
        self.X_test_direct_eval_s, self.y_test_direct_eval_s = prepared_data_dict["test_direct_eval_scaled_seq"]

        self.X_initial_recursive_s, self.y_recursive_s = prepared_data_dict["recursive_test_scaled_seq"]
        self.X_initial_recursive_raw = prepared_data_dict["recursive_test_raw_X_initial"]
        self.y_recursive_raw_seq = prepared_data_dict[
            "recursive_test_raw_y_seq_list"]

        self.y_train_raw_for_mase = prepared_data_dict["train_target_raw_segment"]

        logger.info("\n--- DataPreprocessor: Зведення заповнених атрибутів після prepare_all_data ---")
        logger.info(
            f"К-сть вхідних ознак: {self.n_input_features}, К-сть цільових категорій: {self.n_target_categories}")
        logger.info(f"Цільові колонки: {self.target_cols}")
        logger.info(
            f"Тренувальні послідовності: X_train_s {self.X_train_s.shape if self.X_train_s is not None else 'None'}, y_train_s {self.y_train_s.shape if self.y_train_s is not None else 'None'}")
        logger.info(
            f"Валідаційні послідовності: X_val_s {self.X_val_s.shape if self.X_val_s is not None else 'None'}, y_val_s {self.y_val_s.shape if self.y_val_s is not None else 'None'}")
        logger.info(
            f"Тестові (пряма оцінка) послідовності: X_test_direct_eval_s {self.X_test_direct_eval_s.shape if self.X_test_direct_eval_s is not None else 'None'}, y_test_direct_eval_s {self.y_test_direct_eval_s.shape if self.y_test_direct_eval_s is not None else 'None'}")

        if self.X_initial_recursive_s is not None and self.X_initial_recursive_s.size > 0:
            logger.info(
                f"Початковий рекурсивний вхід (масштабований): X_initial_recursive_s {self.X_initial_recursive_s.shape}")
            logger.info(f"Рекурсивні цільові значення (масштабовані): y_recursive_s {self.y_recursive_s.shape}")
            logger.info(
                f"Початковий рекурсивний вхід (вихідний): X_initial_recursive_raw {self.X_initial_recursive_raw.shape}")
            logger.info(
                f"Рекурсивні цільові значення (вихідні послідовності): y_recursive_raw_seq {self.y_recursive_raw_seq.shape}")
        else:
            logger.info("Дані рекурсивного тесту не були згенеровані або порожні.")

        logger.info(
            f"Форма y_train_raw_for_mase: {self.y_train_raw_for_mase.shape if self.y_train_raw_for_mase is not None else 'None'}")
        logger.info(
            f"Масштабувальник X навчений: {hasattr(self.x_scaler, 'data_min_') and self.x_scaler.data_min_ is not None}")
        logger.info(
            f"Масштабувальник Y навчений: {hasattr(self.y_scaler, 'data_min_') and self.y_scaler.data_min_ is not None}")
        logger.info("--- DataPreprocessor: prepare_all_data завершено ---\n")

    def scale_input_for_prediction(self, x_raw_window: np.ndarray) -> np.ndarray:
        """Масштабує вікно вихідних вхідних ознак для прогнозування та додає розмірність пакету."""
        if self.n_input_features is None:
            raise RuntimeError("Препроцесор повинен бути спочатку навчений за допомогою prepare_all_data().")
        if not hasattr(self.x_scaler, 'data_min_') or self.x_scaler.data_min_ is None:
            raise RuntimeError("Масштабувальник X не навчений. Спочатку викличте prepare_all_data().")
        if x_raw_window.shape != (self.window_size, self.n_input_features):
            raise ValueError(
                f"Вхідні вихідні дані повинні мати форму ({self.window_size}, {self.n_input_features}), отримано {x_raw_window.shape}")

        scaled_data = self.x_scaler.transform(x_raw_window)
        return np.expand_dims(scaled_data, axis=0)  # Форма (1, window_size, n_features)

    def inverse_transform_predictions(self, y_pred_scaled: np.ndarray) -> np.ndarray:
        """Обернено трансформує масштабовані прогнози до їх вихідного масштабу."""
        if self.n_target_categories is None:
            raise RuntimeError("Препроцесор повинен бути спочатку навчений за допомогою prepare_all_data().")
        if not hasattr(self.y_scaler, 'data_min_') or self.y_scaler.data_min_ is None:
            raise RuntimeError("Масштабувальник Y не навчений. Спочатку викличте prepare_all_data().")

        original_shape = y_pred_scaled.shape

        # Обробка різних вхідних розмірностей для y_pred_scaled
        if y_pred_scaled.ndim == 3:  # (num_samples, forecast_horizon, n_outputs)
            reshaped_for_scaling = y_pred_scaled.reshape(-1, self.n_target_categories)
        elif y_pred_scaled.ndim == 2:  # (forecast_horizon_or_samples, n_outputs)
            # Це може бути (forecast_horizon, n_outputs) або (samples, n_outputs), якщо fh=1 і стиснуто
            reshaped_for_scaling = y_pred_scaled
        elif y_pred_scaled.ndim == 1:
            if self.n_target_categories == 1:
                reshaped_for_scaling = y_pred_scaled.reshape(-1, 1)
            else:
                reshaped_for_scaling = y_pred_scaled.reshape(1, self.n_target_categories)
        else:
            raise ValueError(f"Непідтримувана форма y_pred_scaled: {y_pred_scaled.shape}")

        unscaled = self.y_scaler.inverse_transform(reshaped_for_scaling)

        if original_shape == unscaled.shape:  # Якщо reshaped_for_scaling вже мав правильну 2D форму
            return unscaled
        if y_pred_scaled.ndim == 3:  # (num_samples, forecast_horizon, n_outputs)
            return unscaled.reshape(original_shape)
        if y_pred_scaled.ndim == 2 and original_shape[0] == self.forecast_horizon and original_shape[
            1] == self.n_target_categories:
            return unscaled  # Вже (forecast_horizon, n_outputs)
        if y_pred_scaled.ndim == 1 and 1 < self.n_target_categories == len(
                y_pred_scaled):  # (n_outputs,) для fh=1
            return unscaled.squeeze()  # назад до 1D (n_outputs)
        if y_pred_scaled.ndim == 1 and self.n_target_categories == 1:  # (forecast_horizon,)
            return unscaled.squeeze()  # назад до 1D (forecast_horizon)

        try:
            return unscaled.reshape(original_shape)
        except ValueError:
            warnings.warn(
                f"Не вдалося змінити форму немасштабованих прогнозів з {unscaled.shape} назад до вихідної {original_shape}. Повернення як {unscaled.shape}.")
            return unscaled


def build_model_architecture(hp: kt.HyperParameters,
                             window_size: int,
                             n_features: int,
                             forecast_horizon: int,
                             n_outputs: int) -> keras.Model:
    """Визначає та компілює архітектуру нейронної мережі на основі гіперпараметрів."""
    model = keras.Sequential()
    model.add(keras.Input(shape=(window_size, n_features)))
    model_type = hp.Choice('model_type', values=['lstm', 'gru', 'tcn'])

    if model_type in ['lstm', 'gru']:
        num_rnn_layers = hp.Int('num_rnn_layers', 1, 3)
        for i in range(num_rnn_layers):
            rnn_units = hp.Int(f'rnn_units_layer_{i + 1}', min_value=32, max_value=256, step=32)
            is_last_rnn_layer_in_stack = (i == num_rnn_layers - 1)
            common_rnn_params = {'units': rnn_units, 'return_sequences': not is_last_rnn_layer_in_stack}
            if model_type == 'lstm':
                model.add(layers.LSTM(**common_rnn_params))
            else:
                model.add(layers.GRU(**common_rnn_params))
            model.add(
                layers.Dropout(rate=hp.Float(f'dropout_rnn_layer_{i + 1}', min_value=0.0, max_value=0.5, step=0.1)))
    elif model_type == 'tcn':
        tcn_filters = hp.Int('tcn_filters', min_value=32, max_value=128, step=32)
        tcn_kernel_size = hp.Int('tcn_kernel_size', min_value=2, max_value=7, step=1)
        tcn_nb_stacks = hp.Int('tcn_nb_stacks', min_value=1, max_value=2)
        dilation_choice = hp.Choice('tcn_dilation_set', values=['short', 'medium', 'long'])  # Додано 'long'
        if dilation_choice == 'short':
            tcn_dilations = [1, 2, 4]
        elif dilation_choice == 'medium':
            tcn_dilations = [1, 2, 4, 8]
        else:  # 'long'
            tcn_dilations = [1, 2, 4, 8, 16]
        model.add(TCN(nb_filters=tcn_filters, kernel_size=tcn_kernel_size, nb_stacks=tcn_nb_stacks,
                      dilations=tcn_dilations, padding='causal',
                      use_skip_connections=hp.Boolean('tcn_skip_connections', default=True),
                      dropout_rate=hp.Float('tcn_internal_dropout', min_value=0.0, max_value=0.3, step=0.05),
                      return_sequences=False, input_shape=(window_size, n_features)))
        model.add(layers.Dropout(rate=hp.Float('dropout_after_tcn', min_value=0.0, max_value=0.5, step=0.1)))

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
    """Обгортка Keras Tuner HyperModel для функції побудови моделі."""

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
        """Метод fit для Keras Tuner, включає batch_size як гіперпараметр для налаштування."""
        batch_size = hp.Int('batch_size',
                            min_value=self.batch_size_min,
                            max_value=self.batch_size_max,
                            step=self.batch_size_step)
        return model.fit(*args, batch_size=batch_size, **kwargs)


class ModelTrainer:
    """Обробляє налаштування гіперпараметрів моделі та фінальне навчання."""

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
                "DataPreprocessor повинен запустити prepare_all_data() перед ініціалізацією ModelTrainer для встановлення розмірностей.")

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
        """Виконує пошук гіперпараметрів за допомогою Keras Tuner."""
        if self.dp.X_train_s is None or self.dp.X_val_s is None:
            raise RuntimeError("Препроцесор даних не підготував тренувальні/валідаційні дані.")

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
        logger.info("Початок налаштування гіперпараметрів...")
        tuner.search(
            self.dp.X_train_s, self.dp.y_train_s,
            epochs=epochs,
            validation_data=(self.dp.X_val_s, self.dp.y_val_s),
            callbacks=[early_stopping_callback],
            verbose=1
        )
        self.best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info("\n--- Знайдені найкращі гіперпараметри ---")
        for hp_name in self.best_hps.values:  # Ітерація по фактичних назвах гіперпараметрів
            logger.info(f"- {hp_name}: {self.best_hps.get(hp_name)}")

        temp_best_model = self.hypermodel.build(self.best_hps)  # Побудова з найкращими гіперпараметрами
        logger.info("\nЗведення моделі, побудованої з найкращими гіперпараметрами:")
        temp_best_model.summary(print_fn=logger.info)  # Направлення summary в логер
        return self.best_hps

    def train_final_model(self, final_train_epochs: int = 150, early_stopping_patience: int = 20):
        """Навчає фінальну модель з використанням найкращих знайдених гіперпараметрів."""
        if self.best_hps is None:
            raise RuntimeError("Гіперпараметри не були налаштовані. Спочатку викличте tune_hyperparameters.")

        logger.info("\nПобудова та навчання фінальної моделі з найкращими гіперпараметрами...")
        self.trained_model = build_model_architecture(  # Використання окремої функції
            self.best_hps,
            window_size=self.dp.window_size,
            n_features=self.dp.n_input_features,
            forecast_horizon=self.dp.forecast_horizon,
            n_outputs=self.dp.n_target_categories
        )
        best_batch_size = self.best_hps.get('batch_size')
        if best_batch_size is None:
            warnings.warn("Розмір пакету не знайдено в best_hps з тюнера, використовується стандартний 32.")
            best_batch_size = 32

        callbacks_list = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1,
                                          restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_filepath, monitor='val_loss', save_best_only=True,
                                            save_weights_only=False, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7, verbose=1)
        ]
        logger.info(f"\nПочаток фінального навчання до {final_train_epochs} епох з batch_size={best_batch_size}...")
        self.training_history = self.trained_model.fit(
            self.dp.X_train_s, self.dp.y_train_s,
            epochs=final_train_epochs,
            batch_size=best_batch_size,
            validation_data=(self.dp.X_val_s, self.dp.y_val_s),
            callbacks=callbacks_list,
            verbose=1
        )
        logger.info(f"\nФінальне навчання завершено.")
        if os.path.exists(self.checkpoint_filepath):
            logger.info(f"Найкраща модель під час навчання збережена в {self.checkpoint_filepath}.")
            self.load_model(
                self.checkpoint_filepath)
        else:
            warnings.warn(f"Файл контрольної точки {self.checkpoint_filepath} не було створено. "
                          "Модель у пам'яті може бути не найкращою, якщо рання зупинка відбулася до першого збереження.")
        return self.trained_model

    def load_model(self, filepath: str):
        """Завантажує попередньо навчену модель."""
        logger.info(f"Завантаження моделі з {filepath}...")

        custom_objects = {
            'TCN': TCN
        }
        self.trained_model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
        logger.info("Модель успішно завантажена.")
        logger.info("Зведення завантаженої моделі:")
        self.trained_model.summary(print_fn=logger.info)
        return self.trained_model


class Predictor:
    """Обробляє створення прогнозів та їх оцінку."""

    def __init__(self, model: keras.Model, data_preprocessor: DataPreprocessor):
        if not isinstance(model, keras.Model):
            raise ValueError("Модель повинна бути екземпляром keras.Model.")
        if not isinstance(data_preprocessor, DataPreprocessor):
            raise ValueError("Препроцесор даних повинен бути екземпляром DataPreprocessor.")

        self.model = model
        self.dp = data_preprocessor

    def predict_one_day(self, raw_data_for_window: np.ndarray) -> np.ndarray:
        """
        Робить один прогноз на наступний період forecast_horizon.
        Args:
            raw_data_for_window: Масив Numpy форми (window_size, n_features) з вихідними даними ознак.
        Returns:
            Масив Numpy форми (forecast_horizon, n_outputs) з немасштабованими прогнозами.
        """
        if raw_data_for_window.shape != (self.dp.window_size, self.dp.n_input_features):
            raise ValueError(
                f"Вхідний raw_data_for_window повинен мати форму ({self.dp.window_size}, {self.dp.n_input_features}), "
                f"отримано {raw_data_for_window.shape}")

        scaled_input_sequence = self.dp.scale_input_for_prediction(raw_data_for_window)
        scaled_prediction = self.model.predict(scaled_input_sequence, verbose=0)[
            0]  # Отримати (forecast_horizon, n_outputs)
        unscaled_prediction = self.dp.inverse_transform_predictions(scaled_prediction)
        return unscaled_prediction

    def predict_recursively(self, initial_raw_window_data: np.ndarray,
                            num_recursive_steps: int,
                            true_future_exogenous_raw: np.ndarray = None) -> np.ndarray:
        """
        Робить рекурсивні прогнози на вказану кількість кроків.
        Args:
            initial_raw_window_data: Вихідні дані ознак для початкового вікна. Форма: (window_size, n_features).
            num_recursive_steps: Кількість майбутніх кроків для прогнозування.
            true_future_exogenous_raw: Необов'язково. Вихідні значення для майбутніх екзогенних ознак.
                                       Потрібно, якщо n_features > n_outputs.
                                       Форма: (num_recursive_steps - 1, n_exogenous_features).
                                       n_exogenous_features = n_features - n_outputs.
                                       i-й рядок відповідає екзогенним ознакам для входу (i+1)-го кроку прогнозування.
        Returns:
            Масив Numpy немасштабованих рекурсивних прогнозів. Форма: (num_recursive_steps, forecast_horizon, n_outputs).
        """
        if initial_raw_window_data.shape != (self.dp.window_size, self.dp.n_input_features):
            raise ValueError(
                f"initial_raw_window_data повинен мати форму ({self.dp.window_size}, {self.dp.n_input_features})")

        n_exo_features = self.dp.n_input_features - self.dp.n_target_categories
        if n_exo_features > 0:
            if true_future_exogenous_raw is None:
                raise ValueError(
                    f"Модель має {n_exo_features} екзогенних ознак. true_future_exogenous_raw повинен бути наданий.")
            if true_future_exogenous_raw.shape[
                0] < num_recursive_steps - 1:  # Потрібно достатньо для всіх, крім останнього кроку
                raise ValueError(
                    f"true_future_exogenous_raw потребує {num_recursive_steps - 1} рядків для екзогенних ознак, отримано {true_future_exogenous_raw.shape[0]}")
            if true_future_exogenous_raw.shape[1] != n_exo_features:
                raise ValueError(
                    f"true_future_exogenous_raw повинен мати {n_exo_features} колонок, отримано {true_future_exogenous_raw.shape[1]}.")

        current_scaled_input_sequence = self.dp.x_scaler.transform(initial_raw_window_data)
        all_recursive_predictions_scaled_list = []

        for i in range(num_recursive_steps):
            model_input = np.expand_dims(current_scaled_input_sequence, axis=0)
            predicted_step_scaled_fh = self.model.predict(model_input, verbose=0)[
                0]  # Форма (forecast_horizon, n_outputs)
            all_recursive_predictions_scaled_list.append(predicted_step_scaled_fh)

            if i < num_recursive_steps - 1:
                if self.dp.forecast_horizon != 1:
                    warnings.warn(
                        "Логіка оновлення ознак рекурсивного прогнозування наразі припускає forecast_horizon=1 "
                        "для вилучення цілей одного наступного кроку. Потрібна адаптація для >1.")

                new_target_values_scaled = predicted_step_scaled_fh[0,
                                           :]

                new_last_row_features_scaled = np.zeros(self.dp.n_input_features)
                new_last_row_features_scaled[:self.dp.n_target_categories] = new_target_values_scaled

                if n_exo_features > 0:
                    raw_exo_for_next_step_input = true_future_exogenous_raw[i]  # (n_exo_features,)
                    temp_full_row_for_scaling_exo = np.zeros((1, self.dp.n_input_features))

                    placeholder_targets = np.mean(current_scaled_input_sequence[:, :self.dp.n_target_categories],
                                                  axis=0)
                    temp_full_row_for_scaling_exo[0,
                    :self.dp.n_target_categories] = placeholder_targets  # Використання масштабованих середніх
                    temp_full_row_for_scaling_exo[0, self.dp.n_target_categories:] = raw_exo_for_next_step_input

                    scaled_full_row = self.dp.x_scaler.transform(temp_full_row_for_scaling_exo)
                    new_last_row_features_scaled[self.dp.n_target_categories:] = scaled_full_row[0,
                                                                                 self.dp.n_target_categories:]

                current_scaled_input_sequence = np.roll(current_scaled_input_sequence, -1, axis=0)
                current_scaled_input_sequence[-1, :] = new_last_row_features_scaled

        all_recursive_predictions_scaled_arr = np.array(all_recursive_predictions_scaled_list)
        # Форма: (num_recursive_steps, forecast_horizon, n_outputs)

        # Обернене перетворення всіх одразу
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

        # Перевірка розмірності даних, 2D: (samples, features_or_targets)
        if y_train_unscaled_arr.ndim == 1: y_train_unscaled_arr = y_train_unscaled_arr.reshape(-1, 1)
        if y_true_arr.ndim == 1: y_true_arr = y_true_arr.reshape(-1, 1)
        if y_pred_arr.ndim == 1: y_pred_arr = y_pred_arr.reshape(-1, 1)

        if y_true_arr.shape[1] != y_train_unscaled_arr.shape[1] or y_pred_arr.shape[1] != y_train_unscaled_arr.shape[1]:
            warnings.warn(
                f"MASE: Невідповідність колонок: y_true({y_true_arr.shape}), y_pred({y_pred_arr.shape}), y_train({y_train_unscaled_arr.shape})")
            return np.full(y_true_arr.shape[1] if y_true_arr.ndim > 1 else 1, np.nan)

        forecast_errors = np.abs(y_true_arr - y_pred_arr)
        mean_absolute_forecast_error = np.mean(forecast_errors, axis=0)  # Середнє за часовими кроками для кожної цілі

        if len(y_train_unscaled_arr) <= seasonality_period:
            warnings.warn(
                f"MASE: Довжина y_train_unscaled ({len(y_train_unscaled_arr)}) <= сезонності ({seasonality_period}).")
            return np.full(y_true_arr.shape[1] if y_true_arr.ndim > 1 else 1, np.nan)

        naive_forecast_errors_train = np.abs(
            y_train_unscaled_arr[seasonality_period:] - y_train_unscaled_arr[:-seasonality_period])
        mean_absolute_naive_error_train = np.mean(naive_forecast_errors_train, axis=0)

        mase_scores = np.full_like(mean_absolute_forecast_error, np.nan)
        non_zero_denom_mask = mean_absolute_naive_error_train > 1e-9  # Уникнення ділення на нуль або дуже мале число
        mase_scores[non_zero_denom_mask] = mean_absolute_forecast_error[non_zero_denom_mask] / \
                                           mean_absolute_naive_error_train[non_zero_denom_mask]
        return mase_scores

    def evaluate_one_day_prediction(self,
                                    y_true_unscaled: np.ndarray,
                                    y_pred_unscaled: np.ndarray) -> dict:
        """
        Оцінює один або кілька немасштабованих прогнозів відносно істинних значень.
        Args:
            y_true_unscaled: Істинні значення, форма (num_predictions_or_horizon_steps, n_outputs).
            y_pred_unscaled: Прогнозовані значення, форма (num_predictions_or_horizon_steps, n_outputs).
        Returns:
            Словник метрик (MAE, RMSE, MASE для кожної цільової категорії та Середнє).
        """
        if y_true_unscaled.shape != y_pred_unscaled.shape:
            raise ValueError(
                f"y_true_unscaled ({y_true_unscaled.shape}) та y_pred_unscaled ({y_pred_unscaled.shape}) повинні мати однакову форму.")
        if y_true_unscaled.ndim == 1 and self.dp.n_target_categories > 1:  # (n_outputs,)
            y_true_unscaled = y_true_unscaled.reshape(1, -1)
            y_pred_unscaled = y_pred_unscaled.reshape(1, -1)
        elif y_true_unscaled.ndim == 1 and self.dp.n_target_categories == 1:  # (horizon_steps,)
            y_true_unscaled = y_true_unscaled.reshape(-1, 1)
            y_pred_unscaled = y_pred_unscaled.reshape(-1, 1)

        if y_true_unscaled.shape[1] != self.dp.n_target_categories:
            raise ValueError(
                f"Кількість колонок ({y_true_unscaled.shape[1]}) повинна відповідати n_target_categories ({self.dp.n_target_categories})")
        if self.dp.y_train_raw_for_mase is None:
            raise RuntimeError("y_train_raw_for_mase недоступний в DataPreprocessor. Запустіть prepare_all_data.")

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

        metrics_summary['Середнє'] = {
            'MAE': np.nanmean(mae_scores),
            'RMSE': np.nanmean(rmse_scores),
            'MASE': np.nanmean(mase_scores)
        }
        return metrics_summary

    def evaluate_on_test_set(self):
        """Оцінює модель на попередньо визначеному тестовому наборі для прямої оцінки."""
        if self.dp.X_test_direct_eval_s is None or self.dp.X_test_direct_eval_s.shape[0] == 0:
            logger.info("Тестовий набір для прямої оцінки порожній або не підготовлений. Пропуск оцінки.")
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

    print("Завантаження даних...")
    X_train_df = pd.read_pickle(file_path_data_folder / 'X_train.pkl')
    X_test_df = pd.read_pickle(file_path_data_folder / 'X_test.pkl')
    y_train_df = pd.read_pickle(file_path_data_folder / 'y_train.pkl')
    y_test_df = pd.read_pickle(file_path_data_folder / 'y_test.pkl')

    X_raw_main = pd.concat([X_train_df, X_test_df], axis=0).to_numpy()
    y_raw_main = pd.concat([y_train_df, y_test_df], axis=0).to_numpy()
    target_cols_main = y_train_df.columns.tolist()
    print(f'Вихідна форма X: {X_raw_main.shape}, Вихідна форма y: {y_raw_main.shape}, Цільові: {target_cols_main}')

    # 1. Попередня обробка даних
    print("\n--- Ініціалізація DataPreprocessor ---")
    data_preprocessor = DataPreprocessor(
        window_size=WINDOW_SIZE_MAIN, forecast_horizon=FORECAST_HORIZON_MAIN,
        train_ratio=TRAIN_RATIO_MAIN, val_ratio=VAL_RATIO_MAIN,
        n_recursive_test_steps=N_RECURSIVE_TEST_STEPS_MAIN
    )
    data_preprocessor.prepare_all_data(X_raw_main, y_raw_main, target_cols_main)

    # 2. Навчання моделі
    print("\n--- Ініціалізація ModelTrainer ---")

    model_checkpoint = file_path_data_folder / f'model/{CHECKPOINT_FILEPATH_MAIN}'
    tuner_directory = file_path_data_folder / f'keras_tuner'

    model_trainer = ModelTrainer(data_preprocessor=data_preprocessor,
                                 checkpoint_filepath=str(model_checkpoint.resolve()),
                                 tuner_directory_path=str(tuner_directory.resolve()))

    train_new_model = True
    if os.path.exists(model_checkpoint):
        user_choice = input(
            f"Знайдено існуючу модель за шляхом {model_checkpoint}. Завантажити її? (y/n, за замовчуванням y): ").lower()
        if user_choice == '' or user_choice == 'y':
            try:
                print("Спроба завантажити існуючу модель...")
                model_trainer.load_model(str(model_checkpoint.resolve()))
                if model_trainer.trained_model:
                    print("Модель успішно завантажена. Пропуск налаштування та навчання.")
                    train_new_model = False
            except Exception as e:
                print(f"Помилка завантаження існуючої моделі: {e}. Буде виконано перенавчання.")
                if os.path.exists(model_checkpoint): os.remove(
                    str(model_checkpoint.resolve()))
        else:
            print("Перехід до перенавчання моделі.")

    if train_new_model:
        print("\n--- Налаштування гіперпараметрів ---")
        model_trainer.tune_hyperparameters(max_trials=TUNER_MAX_TRIALS_MAIN, epochs=TUNER_EPOCHS_MAIN)
        print("\n--- Навчання фінальної моделі ---")
        model_trainer.train_final_model(final_train_epochs=FINAL_TRAIN_EPOCHS_MAIN)

    if model_trainer.trained_model is None:
        print("ПОМИЛКА: Модель не вдалося навчити або завантажити. Вихід.")
        exit()

    # 3. Прогнозування та оцінка
    print("\n--- Ініціалізація Predictor ---")
    predictor = Predictor(model=model_trainer.trained_model, data_preprocessor=data_preprocessor)

    # Приклад: Рекурсивне прогнозування
    num_steps_for_recursion = N_RECURSIVE_TEST_STEPS_MAIN
    required_raw_len_for_recursion = WINDOW_SIZE_MAIN + (
            num_steps_for_recursion - 1) + FORECAST_HORIZON_MAIN  # Максимальний потрібний індекс

    if len(X_raw_main) >= required_raw_len_for_recursion:
        print(f"\n--- Рекурсивне прогнозування на {num_steps_for_recursion} кроків ---")
        # Початкова точка для початкового вікна
        rec_input_start_idx = len(X_raw_main) - required_raw_len_for_recursion
        if rec_input_start_idx < 0: rec_input_start_idx = 0

        initial_raw_window_for_rec = X_raw_main[rec_input_start_idx: rec_input_start_idx + WINDOW_SIZE_MAIN]

        true_future_exo_raw_main = None
        n_exo_main = data_preprocessor.n_input_features - data_preprocessor.n_target_categories
        if n_exo_main > 0:
            exo_start_idx = rec_input_start_idx + WINDOW_SIZE_MAIN
            exo_end_idx = exo_start_idx + (num_steps_for_recursion - 1)
            if exo_end_idx <= len(X_raw_main):  # Перевірка, чи не виходимо за межі X_raw_main
                true_future_exo_raw_main = X_raw_main[exo_start_idx: exo_end_idx,
                                           data_preprocessor.n_target_categories:]
            else:
                print(
                    f"Попередження: Недостатньо майбутніх вихідних даних для всіх {n_exo_main} екзогенних ознак. Пропуск рекурсивного прогнозування.")
                num_steps_for_recursion = 0

        if num_steps_for_recursion > 0:
            recursive_predictions_unscaled = predictor.predict_recursively(
                initial_raw_window_for_rec,
                num_recursive_steps=num_steps_for_recursion,
                true_future_exogenous_raw=true_future_exo_raw_main
            )
            print(
                f"Немасштабовані рекурсивні прогнози (форма {recursive_predictions_unscaled.shape}):\n{recursive_predictions_unscaled}")

            reshaped_recursive_predictions_unscaled = recursive_predictions_unscaled.reshape(5, 5)
            predict_json = {f'Day {n + 1}': dict(zip(target_cols_main, reshaped_recursive_predictions_unscaled[n])) for
                            n in range(5)}
            print(f'Прогноз у формі вкладеного JSON: {predict_json}')
            # Істинні значення для цих рекурсивних кроків
            y_true_rec_start = rec_input_start_idx + WINDOW_SIZE_MAIN
            # Переконуємось, що y_true_rec_end не виходить за межі y_raw_main
            max_possible_y_elements = num_steps_for_recursion * FORECAST_HORIZON_MAIN
            available_y_elements = len(y_raw_main) - y_true_rec_start

            elements_to_take = min(max_possible_y_elements, available_y_elements)
            actual_num_recursive_steps_for_eval = elements_to_take // FORECAST_HORIZON_MAIN

            if actual_num_recursive_steps_for_eval < num_steps_for_recursion:
                print(
                    f"Попередження: Доступно менше істинних значень Y ({actual_num_recursive_steps_for_eval} кроків) для оцінки рекурсивних прогнозів, ніж було спрогнозовано ({num_steps_for_recursion} кроків). Оцінка буде для доступних даних.")
                # Обрізаємо прогнози, щоб відповідати доступним істинним значенням
                recursive_predictions_unscaled = recursive_predictions_unscaled[:actual_num_recursive_steps_for_eval]

            if actual_num_recursive_steps_for_eval > 0:
                y_true_rec_end = y_true_rec_start + actual_num_recursive_steps_for_eval * FORECAST_HORIZON_MAIN
                y_true_recursive_raw = y_raw_main[y_true_rec_start: y_true_rec_end].reshape(
                    actual_num_recursive_steps_for_eval, FORECAST_HORIZON_MAIN, data_preprocessor.n_target_categories
                )

                # Оцінка всіх рекурсивних кроків разом
                metrics_recursive_all = predictor.evaluate_one_day_prediction(
                    y_true_recursive_raw.reshape(-1, data_preprocessor.n_target_categories),
                    # Згладжування часових кроків
                    recursive_predictions_unscaled.reshape(-1, data_preprocessor.n_target_categories)
                )
                print(f"Метрики для всіх {actual_num_recursive_steps_for_eval} рекурсивних кроків (усереднені):")
                for cat, mets in metrics_recursive_all.items(): print(f"  {cat}: {mets}")
            else:
                print("Недостатньо даних для оцінки рекурсивних прогнозів.")

    else:
        print("Недостатньо вихідних даних для прикладу рекурсивного прогнозування.")

    print("\n--- Оцінка моделі на внутрішньому тестовому наборі (Пряма оцінка) ---")
    test_set_metrics = predictor.evaluate_on_test_set()
    if test_set_metrics:
        print(
            "Метрики для внутрішнього тестового набору (прямі прогнози на один крок, усереднені за зразками/горизонтом):")
        for cat, mets in test_set_metrics.items(): print(f"  {cat}: {mets}")

if __name__ == '__main__':
    main()