import json
import logging
import pathlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from predictions.services.main_model.utils import get_logger

from personal_budget_planner.constants import *


class DataPreprocessor:
    """
    Class for processing spending data, including data preparation,
    feature engineering, and category analysis.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger if logger else get_logger(self.__class__.__name__,
                                                       log_file_name='data_formatting.log')
        self.df: pd.DataFrame = pd.DataFrame()

    def _load_and_prepare_initial_df(self, init_data_file_path: pathlib.Path) -> tuple[pd.DataFrame, list[str]]:
        self.logger.info(f"Loading initial dataset from: {init_data_file_path}")
        try:
            initial_df = pd.read_csv(init_data_file_path)
        except FileNotFoundError:
            self.logger.error(f"File not found: {init_data_file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading CSV file {init_data_file_path}: {e}")
            raise

        self.logger.info("Creating temporal features from date...")
        initial_df['date'] = pd.to_datetime(initial_df['date'])
        initial_df['month'] = initial_df['date'].dt.month
        initial_df['day_of_week'] = initial_df['date'].dt.dayofweek  # Monday=0, Sunday=6
        initial_df['week_of_month'] = (initial_df['date'].dt.day - 1) // 7 + 1
        initial_df['year'] = initial_df['date'].dt.year
        initial_df['day'] = initial_df['date'].dt.day

        initial_df['month_period'] = initial_df[['month', 'year']].apply(lambda x: f'{x.iloc[0]}-{x.iloc[1]}', axis=1)

        self.logger.info(f"Years: {initial_df['year'].nunique()} ({','.join(map(str, sorted(initial_df['year'].unique())))})")
        self.logger.info(f"Sequential months: {initial_df['month_period'].nunique()}")

        categories = list(initial_df["category"].unique())
        self.logger.info(f"Found categories: {len(categories)}")
        self.logger.info(f"Initial dataframe length: {len(initial_df)}")
        return initial_df, categories

    def _create_pivot_table(self, initial_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a pivot table with expense sums by category.
        """
        self.logger.info("Creating pivot table...")
        initial_df_copy = initial_df.copy()  # Work with a copy to avoid SettingWithCopyWarning
        initial_df_copy['date_agg'] = initial_df_copy['date'].dt.date

        # Use all temporal features in the index to retain them
        df_pivoted = initial_df_copy.pivot_table(
            index=['date_agg', 'year', 'month', 'week_of_month', 'day_of_week'],
            columns='category',
            values='amount',
            aggfunc='sum'
        ).reset_index()

        df_pivoted.rename(columns={'date_agg': 'date'}, inplace=True)

        # Add 'month_period' again if lost during pivot_table
        df_pivoted['month_period'] = df_pivoted[['month', 'year']].apply(lambda x: f'{x.iloc[0]}-{x.iloc[1]}', axis=1)

        df_pivoted.fillna(0, inplace=True)
        self.logger.info(f"Number of unique dates in pivot table: {len(df_pivoted['date'].unique())}")
        self.logger.info(f"Pivot table length: {len(df_pivoted)}")
        return df_pivoted

    def _classify_spending_categories(self, df_pivoted: pd.DataFrame, categories: list[str],
                                      initial_month_period_nunique: int) -> tuple[
        dict[str, list[str]], list[str], list[str], list[str]]:
        """
        Classifies spending categories into regular, periodic, and rare
        based on the number of zero values.
        """
        self.logger.info("Classifying spending categories by frequency...")
        threshold_80 = 0.80 * len(df_pivoted)
        threshold_95 = len(df_pivoted) - initial_month_period_nunique + SKIPPED_PERIODS

        zero_values = {}
        for category in categories:
            if category in df_pivoted.columns:
                zero_values[category] = len(df_pivoted.loc[df_pivoted[category] == 0])
            else:
                zero_values[category] = len(df_pivoted)

        zero_values_series = pd.Series(zero_values).sort_values(ascending=True)

        above_95_nan_cols = zero_values_series.loc[zero_values_series > threshold_95].index.to_list()
        above_80_nan_cols = zero_values_series.loc[
            zero_values_series.between(threshold_80, threshold_95, inclusive='right')].index.to_list()
        regular_cols = zero_values_series.loc[zero_values_series < threshold_80].index.to_list()

        self.logger.info(f"Regular categories (<80% zeros): {len(regular_cols)}")
        self.logger.info(f"Periodic categories (80–95% zeros): {len(above_80_nan_cols)}")
        self.logger.info(f"Rare categories (>95% zeros): {len(above_95_nan_cols)}")

        classified_cols = {
            'regular': regular_cols,
            'above_80_nan': above_80_nan_cols,
            'above_95_nan': above_95_nan_cols
        }
        return classified_cols, regular_cols, above_80_nan_cols, above_95_nan_cols

    def _engineer_temporal_features(self, df_target: pd.DataFrame) -> list[str]:
        """
        Creates cyclical temporal features (sin, cos) and encodes the year.
        Modifies the DataFrame `df_target` in place.
        """
        self.logger.info("Creating cyclical temporal features...")
        new_temporal_cols: list[str] = []
        for col, period in TEMPORAL_COLUMNS_CONFIG.items():
            if col not in df_target.columns:
                self.logger.warning(f"Column '{col}' not found in DataFrame. Skipping cyclical features for it.")
                continue

            values = df_target[col]
            sin_col_name = f'{col}_sin'
            cos_col_name = f'{col}_cos'
            df_target[sin_col_name] = np.sin(2 * np.pi * values / period)
            df_target[cos_col_name] = np.cos(2 * np.pi * values / period)
            new_temporal_cols.extend([sin_col_name, cos_col_name])

        if 'year' in df_target.columns:
            self.logger.info("Encoding year...")
            df_target['encoded_year'] = LabelEncoder().fit_transform(df_target['year'])
            new_temporal_cols.append('encoded_year')
        else:
            self.logger.warning("Column 'year' not found for encoding.")

        self.logger.info(f"New temporal columns created: {new_temporal_cols}")
        return new_temporal_cols

    def _engineer_aggregated_spend_features(self, df_target: pd.DataFrame, periodical_cols: list[str],
                                            occasional_cols: list[str]) -> list[str]:
        """
        Creates aggregated features for periodic and rare expenses.
        Modifies the DataFrame `df_target` in place.
        """
        self.logger.info("Creating aggregated spending features...")
        subtotals_columns: list[str] = []

        if periodical_cols:
            df_target['periodical_sum'] = df_target[periodical_cols].sum(axis=1)
            subtotals_columns.append('periodical_sum')
            self.logger.info("'periodical_sum' created.")
        else:
            df_target['periodical_sum'] = 0
            subtotals_columns.append('periodical_sum')
            self.logger.info("No columns for 'periodical_sum', zero-filled column created.")

        if occasional_cols:
            df_target['occasional_sum'] = df_target[occasional_cols].sum(axis=1)
            subtotals_columns.append('occasional_sum')
            self.logger.info("'occasional_sum' created.")
        else:
            df_target['occasional_sum'] = 0
            subtotals_columns.append('occasional_sum')
            self.logger.info("No columns for 'occasional_sum', zero-filled column created.")

        return subtotals_columns

    def _find_correlated_aux_features(
            self,
            df_corr_analysis: pd.DataFrame,
            target_columns: list[str],
            correlation_threshold: float = 0.5,
            method: str = 'pearson'
    ) -> list[str]:
        """
        Finds auxiliary features that have correlation (by absolute value)
        above the given threshold with any of the target columns.
        """
        self.logger.info(f"Searching for correlated auxiliary features with threshold {correlation_threshold}...")
        if not isinstance(df_corr_analysis, pd.DataFrame):
            self.logger.error("'df_corr_analysis' must be a pandas DataFrame.")
            raise ValueError("'df_corr_analysis' must be a pandas DataFrame.")
        if not isinstance(target_columns, list) or not all(isinstance(col, str) for col in target_columns):
            self.logger.error("'target_columns' must be a list of strings.")
            raise ValueError("'target_columns' must be a list of strings.")
        if not 0 <= correlation_threshold <= 1:
            self.logger.error("'correlation_threshold' must be in [0, 1].")
            raise ValueError("'correlation_threshold' must be in [0, 1].")

        valid_target_columns = [col for col in target_columns if col in df_corr_analysis.columns]
        if not valid_target_columns:
            self.logger.warning("None of the specified target columns found in correlation DataFrame.")
            return []

        aux_candidate_columns = [col for col in df_corr_analysis.columns if col not in valid_target_columns]

        if not aux_candidate_columns:
            self.logger.warning("No auxiliary numeric columns for correlation analysis.")
            return []

        try:
            corr_matrix = df_corr_analysis.corr(method=method)
        except Exception as e:
            self.logger.error(f"Correlation matrix computation failed: {e}")
            return []

        correlations_with_targets = corr_matrix.loc[aux_candidate_columns, valid_target_columns].abs()
        highly_correlated_mask = (correlations_with_targets > correlation_threshold).any(axis=1)
        result_features = correlations_with_targets[highly_correlated_mask].index.tolist()

        self.logger.info(f"Found {len(result_features)} highly correlated auxiliary features.")
        return sorted(result_features)

    def process_spending_data(self, file_path: pathlib.Path,
                              correlation_threshold_for_aux_features: float = 0.1) -> tuple[
        pd.DataFrame, dict[str, list[str]], list[str]]:
        """
        Main method for processing spending data.
        Loads data, cleans, transforms, engineers features,
        and returns processed DataFrame, column classification and correlated features.
        """
        self.logger.info(f"Starting data processing for file: {file_path}")

        initial_df, all_categories = self._load_and_prepare_initial_df(file_path)
        initial_month_period_nunique = initial_df["month_period"].nunique()

        self.df = self._create_pivot_table(initial_df)

        classified_categories, regular_cols, periodical_cols, occasional_cols = \
            self._classify_spending_categories(self.df, all_categories, initial_month_period_nunique)

        self._engineer_temporal_features(self.df)
        self._engineer_aggregated_spend_features(self.df, periodical_cols, occasional_cols)

        all_expense_columns_for_corr = [col for col in (regular_cols + periodical_cols + occasional_cols) if
                                        col in self.df.columns]

        if not all_expense_columns_for_corr:
            self.logger.warning("No expense columns found for correlation analysis. Returning empty list.")
            correlated_aux_features = []
        else:
            df_for_correlation_analysis = self.df[all_expense_columns_for_corr].copy()
            valid_regular_cols_for_corr = [col for col in regular_cols if col in df_for_correlation_analysis.columns]

            if not valid_regular_cols_for_corr:
                self.logger.warning("No valid regular target columns for correlation found. Returning empty list.")
                correlated_aux_features = []
            else:
                correlated_aux_features = self._find_correlated_aux_features(
                    df_corr_analysis=df_for_correlation_analysis,
                    target_columns=valid_regular_cols_for_corr,
                    correlation_threshold=correlation_threshold_for_aux_features,
                    method='pearson'
                )

        cols_to_drop = [col for col in RAW_TEMPORAL_COLUMNS_TO_DROP if col in self.df.columns]
        processed_df = self.df.drop(columns=cols_to_drop)

        self.logger.info("Data processing completed successfully.")
        return processed_df, classified_categories, correlated_aux_features

def main():
    print("Running test scenario for SpendingDataProcessor...")

    file_path_data_folder = pathlib.Path(__file__).parents[3].resolve() / 'data'
    file_path = file_path_data_folder / 'dataset.csv'

    processor = DataPreprocessor()

    try:
        final_df, column_categories, correlated_features = processor.process_spending_data(
            file_path)

        if not (file_path_data_folder / 'intermediate').exists():
            pathlib.Path(file_path_data_folder / 'intermediate').mkdir(parents=True)

        pd.to_pickle(final_df, file_path_data_folder / 'intermediate/data_prepare.pkl')

        with open(file_path_data_folder / 'intermediate/columns_cat.json', 'w') as f:
            json.dump(column_categories, f)

        with open(file_path_data_folder / 'intermediate/correlated_features.txt', 'w') as f:
            f.writelines(correlated_features)

        print(f"DataFrame shape: {final_df.shape}")

        print(f"\n--- Classified categories ---")
        for key, val in column_categories.items():
            print(f"{key}: {val}")

        print(f"\n--- Correlated auxiliary features (with regular expenses, threshold 0.1) ---")
        print(correlated_features)

        # Check for expected columns in the final DataFrame
        print(f"\n--- Columns in final DataFrame ---")
        print(final_df.columns.tolist())

    except Exception as e:
        print(f"An error occurred during the test run: {e}")

if __name__ == '__main__':
    main()