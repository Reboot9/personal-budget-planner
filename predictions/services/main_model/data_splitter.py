import json
import pathlib

import pandas as pd
import logging

from predictions.services.main_model.utils import get_logger

from personal_budget_planner.constants import *

class DataSplitter:
    """
    Class for splitting a DataFrame into training and test sets (X, y).
    """

    def __init__(self, logger: logging.Logger | None = None):
        """
        Initialize DataSplitter.
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
        Splits the data into training and test sets X and y.
        """
        self.logger.info("Starting data split into training and test sets...")
        df = full_df.copy()

        # Determine actual target columns to use
        actual_targets_to_use = [f'{col}_log' for col in target_names] if use_log_transformed_targets else target_names
        actual_targets_to_use = [t for t in actual_targets_to_use if t in df.columns]

        if not actual_targets_to_use:
            self.logger.error("No target columns found in DataFrame.")
            empty_df = pd.DataFrame()
            return empty_df, empty_df, empty_df, empty_df

        # Validate selected features
        valid_selected_features = [f for f in selected_feature_names if f in df.columns]
        if not valid_selected_features:
            self.logger.error("Selected features list is empty or not found in DataFrame.")
            empty_df = pd.DataFrame()
            return (
                empty_df, empty_df,
                df[actual_targets_to_use] if actual_targets_to_use else empty_df,
                df[actual_targets_to_use] if actual_targets_to_use else empty_df
            )

        # Check if index and data are valid
        if df.index.max() is None or len(df) == 0:
            self.logger.error("DataFrame is empty or has no valid datetime index.")
            empty_df = pd.DataFrame()
            return empty_df, empty_df, empty_df, empty_df

        # Perform the split
        split_date = df.index.max() - pd.Timedelta(days=test_set_size_days)
        df_train = df[df.index <= split_date].copy()
        df_test = df[df.index > split_date].copy()

        self.logger.info(
            f"Data split at: {split_date}. Training samples: {len(df_train)}, test samples: {len(df_test)}."
        )

        X_train = df_train[valid_selected_features]
        y_train = df_train[actual_targets_to_use]
        X_test = df_test[valid_selected_features]
        y_test = df_test[actual_targets_to_use]

        # Fill NaNs in targets with 0
        y_train = y_train.fillna(0)
        y_test = y_test.fillna(0)

        self.logger.info(
            f"Shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}"
        )
        self.logger.info("Data split completed.")
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