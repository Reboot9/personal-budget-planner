# data_splitter_module.py
import json
import pathlib

import pandas as pd
import logging

from predictions.services.main_model.utils import get_logger

from personal_budget_planner.constants import *

class DataSplitter:
    """
    Клас для розділення DataFrame на навчальну та тестову вибірки (X, y).
    """

    def __init__(self, logger: logging.Logger | None = None):
        """
        Ініціалізація DataSplitter.

        Args:
            logger (logging.Logger | None): Екземпляр логера. Якщо None, створюється новий.
        """
        self.logger = logger if logger else get_logger(self.__class__.__name__,
                                                       log_file_name="data_splitter.log")

    def split_data(self,
                   full_df: pd.DataFrame,
                   selected_feature_names: list[str],
                   target_names: list[str],
                   test_set_size_days: int = TEST_SIZE_DAYS,
                   use_log_transformed_targets: bool = False,
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Розділяє дані на навчальні та тестові набори X та y.

        Args:
            full_df (pd.DataFrame): Повний DataFrame, що містить ознаки та цілі.
            selected_feature_names (list[str]): Список назв відібраних ознак.
            target_names (list[str]): Список назв вихідних цільових колонок.
            test_set_size_days (int): Кількість днів для тестового набору.
            use_log_transformed_targets (bool): Чи використовувати логарифмовані цільові змінні.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                X_train, X_test, y_train, y_test

        Приклад виклику:
        ```python
        # splitter = DataSplitter(logger=my_logger)
        # # df_after_fe - DataFrame після FeatureEngineer
        # # final_selected_features - список відібраних ознак від FeatureSelector
        # # original_targets = ['Market', 'Coffee']
        # # test_days = 90
        # # use_log = False
        # X_train, X_test, y_train, y_test = splitter.split_data(
        #     full_df=df_after_fe,
        #     selected_feature_names=final_selected_features,
        #     target_names=original_targets,
        #     test_set_size_days=test_days,
        #     use_log_transformed_targets=use_log
        # )
        # print(f"X_train shape: {X_train.shape}, y_test shape: {y_test.shape}")
        ```
        """
        self.logger.info("Початок розділення даних на навчальну та тестову вибірки...")
        df = full_df.copy()

        # Визначення цільових колонок для використання
        actual_targets_to_use = [f'{col}_log' for col in target_names] if use_log_transformed_targets else target_names
        actual_targets_to_use = [t for t in actual_targets_to_use if t in df.columns]  # Тільки існуючі

        if not actual_targets_to_use:
            self.logger.error("Цільові колонки для розділення не знайдено в DataFrame.")
            # Повертаємо порожні DataFrames у випадку помилки
            empty_df = pd.DataFrame()
            return empty_df, empty_df, empty_df, empty_df

        # Перевірка наявності відібраних ознак
        valid_selected_features = [f for f in selected_feature_names if f in df.columns]
        if not valid_selected_features:
            self.logger.error("Список відібраних ознак порожній або ознаки відсутні в DataFrame.")
            empty_df = pd.DataFrame()
            return empty_df, empty_df, df[actual_targets_to_use] if actual_targets_to_use else empty_df, df[
                actual_targets_to_use] if actual_targets_to_use else empty_df  # Повернемо хоча б y, якщо є

        # Розділення даних
        if df.index.max() is None or len(df) == 0:
            self.logger.error("DataFrame порожній або не має часового індексу для розділення.")
            empty_df = pd.DataFrame()
            return empty_df, empty_df, empty_df, empty_df

        split_date = df.index.max() - pd.Timedelta(days=test_set_size_days)
        df_train = df[df.index <= split_date].copy()
        df_test = df[df.index > split_date].copy()

        self.logger.info(
            f"Дані розділено за датою: {split_date}. Навчальних записів: {len(df_train)}, тестових: {len(df_test)}.")

        X_train = df_train[valid_selected_features]
        y_train = df_train[actual_targets_to_use]

        X_test = df_test[valid_selected_features]
        y_test = df_test[actual_targets_to_use]

        # Заповнення NaN в y_train/y_test нулями (як у ноутбуці для навчання моделей)
        y_train = y_train.fillna(0)
        y_test = y_test.fillna(0)

        self.logger.info(
            f"Розмірності: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")
        self.logger.info("Розділення даних завершено.")
        return X_train, X_test, y_train, y_test

def main():
    file_path_data_folder = pathlib.Path(__file__).parents[3].resolve() / 'data'

    df = pd.read_pickle(file_path_data_folder / 'intermediate/engineered_df.pkl')

    with open(file_path_data_folder / 'intermediate/selected_features.json', 'r') as f:
        selected_features = json.load(f)['selected_features']

    with open(file_path_data_folder / 'intermediate/features_dict.json', 'r') as f:
        target_cols = json.load(f)['target']

    X_train, X_test, y_train, y_test = DataSplitter().split_data(df,
                                                                 selected_features,
                                                                 target_names=target_cols)

    X_train.to_pickle(file_path_data_folder / 'intermediate/X_train.pkl')
    y_train.to_pickle(file_path_data_folder / 'intermediate/y_train.pkl')
    X_test.to_pickle(file_path_data_folder / 'intermediate/X_test.pkl')
    y_test.to_pickle(file_path_data_folder / 'intermediate/y_test.pkl')


if __name__ == '__main__':
    main()