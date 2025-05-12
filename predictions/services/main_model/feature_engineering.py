import json
import logging
import pathlib
import warnings
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from predictions.services.main_model.utils import get_logger
from personal_budget_planner.constants import *
import warnings

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class FeatureEngineer:
    """
    Class for performing advanced data analysis (EDA) and feature engineering
    based on time series cost data.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger: logging.Logger = logger if logger else get_logger(self.__class__.__name__,
                                                                       log_file_name='feature_engineer.log')
        self.df: pd.DataFrame = pd.DataFrame()
        self.features: dict[str, list[str]] = {}
        self.target_cols: list[str] = []
        self.including_secondary_features: list[str] = []

        warnings.filterwarnings('ignore', category=UserWarning)  # General UserWarnings
        warnings.filterwarnings('ignore', category=FutureWarning)  # FutureWarnings from Pandas/Numpy
        warnings.filterwarnings('ignore',
                                message="A value is trying to be set on a copy of a slice from a DataFrame")  # Specific Pandas warnin

    def _check_datetime_index_continuity(self) -> None:
        """
        Checks the continuity of the datetime index in the DataFrame (self.df).
        """
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.logger.warning("The DataFrame index is not a DatetimeIndex. Continuity check is not possible.")
            return

        self.logger.debug("Checking datetime index continuity...")
        expected_freq: str | None = pd.infer_freq(self.df.index)

        if self.df.empty:
            self.logger.info("The DataFrame is empty, continuity check is not performed.")
            return

        if not expected_freq:
            # Try to determine if it's daily with gaps
            is_daily_normalized: bool = all(self.df.index == self.df.index.normalize())
            diffs: pd.Series = self.df.index.to_series().diff().dropna()
            # If all intervals are 1 day or the index is normalized but infer_freq failed due to gaps
            if not diffs.empty and (diffs == pd.Timedelta(days=1)).all():
                expected_freq = 'D'
            elif is_daily_normalized and diffs.empty and len(self.df.index) == 1:  # Single record
                expected_freq = 'D'  # Assume daily frequency for one record if normalized
            elif is_daily_normalized and not diffs.empty:  # Multiple records, normalized, but infer_freq didn't work
                self.logger.warning(
                    "Failed to automatically determine the index frequency (possibly 'D' with gaps). Assuming 'D' for checking.")
                expected_freq = 'D'
            else:
                self.logger.warning(
                    f"Failed to determine the expected index frequency. Current inferred frequency: {expected_freq}. Skipping detailed gap check.")
                self.logger.info(f"Current index range: from {self.df.index.min()} to {self.df.index.max()}")
                return

        try:
            actual_range: pd.DatetimeIndex = pd.date_range(start=self.df.index.min(), end=self.df.index.max(),
                                                           freq=expected_freq)
            missing: pd.DatetimeIndex = actual_range.difference(self.df.index)
            if not missing.empty:
                self.logger.warning(
                    f"Found {len(missing)} missing periods ({expected_freq}) in the index! First 5: {missing[:5].to_list()}")
            else:
                self.logger.info(f"The datetime index is continuous with the expected frequency {expected_freq}.")
        except Exception as e:
            self.logger.error(f"Error during index continuity check with frequency {expected_freq}: {e}")

    def _check_features(self, verbose: bool = False) -> list[str]:
        """
        Checks for the presence of features described in the `self.features` dictionary in `self.df`.
        """
        self.logger.debug("Running _check_features.")
        all_features_in_dict_values: list[str] = []
        for key, feature_list_val in self.features.items():
            if isinstance(feature_list_val, list):
                all_features_in_dict_values.extend(feature_list_val)
            elif isinstance(feature_list_val, str):  # In case the value is not a list
                all_features_in_dict_values.append(feature_list_val)

        unique_features_in_dict: set[str] = set(all_features_in_dict_values)
        df_columns: set[str] = set(self.df.columns)

        missing_in_df: list[str] = list(unique_features_in_dict - df_columns)

        if missing_in_df:
            self.logger.warning(
                f"Features listed in the `features` dictionary but MISSING in the DataFrame: {missing_in_df}")
        elif verbose:
            self.logger.info("All features from the `features` dictionary are present in the DataFrame.")

        return missing_in_df

    def _aggregate_spending_categories(self, irregular_cols: list[str], non_periodic_cols: list[str]) -> None:
        """
        Aggregates the specified spending columns into thematic groups and calculates the total sum for these groups.
        """
        self.logger.info("Aggregating spending categories...")

        if irregular_cols:
            self.df['IrregularSpendings'] = self.df[irregular_cols].sum(axis=1, min_count=0)
        else:
            self.df['IrregularSpendings'] = 0.0

        if non_periodic_cols:
            self.df['NonPeriodicSpendings'] = self.df[non_periodic_cols].sum(axis=1, min_count=0)
        else:
            self.df['NonPeriodicSpendings'] = 0.0

        # target_cols are set in a public method
        all_listed_cols_for_sum: list[str] = list(set(self.target_cols + irregular_cols + non_periodic_cols))
        # Ensure these columns exist in self.df
        all_listed_cols_for_sum = [col for col in all_listed_cols_for_sum if col in self.df.columns]

        if all_listed_cols_for_sum:
            self.df['AllListedSpendings'] = self.df[all_listed_cols_for_sum].sum(axis=1, min_count=0)
        else:
            self.df['AllListedSpendings'] = 0.0
        self.logger.info(
            "Aggregated columns 'IrregularSpendings', 'NonPeriodicSpendings', 'AllListedSpendings' created/updated.")

    def _fill_missing_dates(self) -> None:
        """
        Fills missing dates in the DataFrame index `self.df` with zeros.

        Sorts the DataFrame by the index, determines the full date range
        with daily frequency ('D') from the minimum to the maximum date in the index,
        finds missing dates, and adds rows filled with zeros for those dates.
        If the DataFrame is empty, the index is not pd.DatetimeIndex, or contains
        fewer than two records, the filling is not performed.
        """
        self.logger.info("Filling missing dates in the index...")
        self.df = self.df.sort_index()
        if self.df.empty or not isinstance(self.df.index, pd.DatetimeIndex) or len(
                self.df.index) < 1:  # len(self.df.index) < 2 was added for date_range, but <1 is sufficient for empty check
            self.logger.info("The DataFrame is empty or the index is not suitable for filling dates.")
            return

        if len(self.df.index) < 2:  # If only one element, we can't determine freq or fill gaps.
            # However, with date_range using explicit 'D' frequency, this is not an issue.
            self.logger.info(
                "Insufficient data in the index to automatically determine range and frequency for filling gaps, but will try with freq='D'.")
            # Even with one element, if we want to fill before or after it, this is possible, but the logic fills *between* min and max.
            # If only one record, min() == max(), so full_range will contain just that one date, and missing_dates will be empty.
            # This is correct.

        try:
            full_range: pd.DatetimeIndex = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq='D')
            missing_dates: pd.DatetimeIndex = full_range.difference(self.df.index)
            if not missing_dates.empty:
                self.logger.info(f"Found {len(missing_dates)} missing dates. Filling with zeros...")
                zero_rows: pd.DataFrame = pd.DataFrame(0, index=missing_dates, columns=self.df.columns)
                self.df = pd.concat([self.df, zero_rows]).sort_index()
                self.logger.info(f"DataFrame size after filling missing dates: {self.df.shape}")
            else:
                self.logger.info("No missing dates found.")
        except Exception as e:
            self.logger.error(f"Error while filling missing dates: {e}")

    def _apply_log_transformation(self) -> None:
        """
        Applies a logarithmic transformation log(1+x) to all numerical spending columns in `self.df`.
        """
        self.logger.info("Applying logarithmic transformation (log1p)...")
        log_cols: list[str] = []
        # Use current columns in self.df that are numerical and not already log-transformed
        cols_for_log: list[str] = [col for col in self.df.columns
                                   if pd.api.types.is_numeric_dtype(self.df[col]) and not col.endswith('_log')]

        for col in cols_for_log:
            log_col_name = f'{col}_log'
            self.df[log_col_name] = np.log1p(self.df[col])
            log_cols.append(log_col_name)

        if 'log_columns' not in self.features:
            self.features['log_columns'] = []
        self.features['log_columns'].extend(log_cols)
        self.features['log_columns'] = sorted(list(set(self.features['log_columns'])))  # Unique and sorted
        self.logger.info(f"Created {len(log_cols)} log-transformed features.")

    def _check_stationarity(self) -> None:
        """
        Checks the stationarity of time series using the Dickey-Fuller (ADF) test.

        The test is performed for target columns (`self.target_cols`) and their
        log-transformed versions (if they exist in `self.df`).
        Results (p-value and stationarity conclusion) are logged.
        Null hypothesis (H0): the series is non-stationary. Alternative (H1): the series is stationary.
        If p-value < 0.05, H0 is rejected.
        """
        self.logger.info("Checking stationarity for target and log-transformed target categories (ADF test)...")
        self.logger.info("H0: Series is non-stationary. H1: Series is stationary. If p-value < 0.05, H0 is rejected.")

        # Columns to test - target and their log-transformed versions
        cols_to_test_adf: list[str] = self.target_cols + [f"{col}_log" for col in self.target_cols if
                                                          f"{col}_log" in self.df.columns]

        for col_name in cols_to_test_adf:
            if col_name not in self.df.columns:
                self.logger.warning(f"Column {col_name} for ADF test is missing.")
                continue
            series_to_test: pd.Series = self.df[col_name].dropna()
            if series_to_test.empty or len(series_to_test) < 10:
                self.logger.info(
                    f"Column '{col_name}': not enough data for ADF test ({len(series_to_test)} points).")
                continue

            try:
                result: tuple[float, float, int, int, dict[str, float], float] = adfuller(series_to_test)
                p_value: float = result[1]
                if p_value < 0.05:
                    self.logger.info(f"  The series '{col_name}' is likely stationary (p-value: {p_value:.4f}).")
                else:
                    self.logger.info(
                        f"  The series '{col_name}' is likely NON-stationary (p-value: {p_value:.4f}). ADF Stat: {result[0]:.4f}.")
            except Exception as e:
                self.logger.error(f"Error while performing ADF test for column '{col_name}': {e}")

    @staticmethod
    def _week_of_month(dt: date) -> int | None:
        """
        Returns the week number of the month (starting from 0) for the given date.

        The calculation accounts for cases where the week at the beginning or end of the year
        may belong to another year according to the ISO calendar.
        Note: Logic may return None in some cases when week_in_month >= 0
        without satisfying week_in_month < 0.
        """
        first_day: date = dt.replace(day=1)
        first_calendar_week_of_month: int = first_day.isocalendar().week
        current_calendar_week: int = dt.isocalendar().week

        if dt.month == 1 and first_calendar_week_of_month > 50:  # January, first day on week of the previous year
            first_calendar_week_of_month = 0  # Effectively makes it week 0 for calculation
        elif dt.month == 12 and current_calendar_week == 1:  # December, date on week 1 of the next year
            current_calendar_week = first_day.isocalendar().week + (dt - first_day).days // 7 + 1

        week_in_month: int = current_calendar_week - first_calendar_week_of_month
        if week_in_month < 0:
            offset: int = 0
            if dt.isocalendar()[0] > first_day.isocalendar()[0]:  # The Current date belongs to the next ISO year
                offset = first_day.replace(month=12, day=31).isocalendar().week  # weeks in the first_day year
                if first_day.isocalendar().week > current_calendar_week:  # e.g., first day is week 53, current is week 1 of the next year
                    offset = first_day.replace(month=12, day=31).isocalendar().week - first_day.isocalendar().week + 1
                else:  # the current date is a later week of the next year
                    offset = first_day.replace(month=12, day=31).isocalendar().week

            week_in_month = (current_calendar_week + offset) - first_calendar_week_of_month

        return week_in_month if week_in_month >= 0 else None

    def _create_calendar_features(self) -> None:
        """
        Creates calendar-based features using the index of `self.df`.

        Adds the following columns: 'day_of_year', 'week_of_year', 'quarter', 'year',
        'month', 'day_of_week', 'day_of_month', 'week_of_month', 'is_weekend',
        'is_month_start', 'is_month_end'.
        The names of the created features are added to `self.features['calendar_features']`.
        Requires `self.df` to have a pd.DatetimeIndex.
        """
        self.logger.info("Creating calendar features...")
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.logger.error("Cannot create calendar features: index is not a DatetimeIndex.")
            return

        self.df['day_of_year'] = self.df.index.dayofyear
        self.df['week_of_year'] = self.df.index.isocalendar().week.astype(int)
        self.df['quarter'] = self.df.index.quarter
        self.df['year'] = self.df.index.year
        self.df['month'] = self.df.index.month
        self.df['day_of_week'] = self.df.index.dayofweek  # Monday=0, Sunday=6
        self.df['day_of_month'] = self.df.index.day

        if not self.df.index.empty:
            self.df['week_of_month'] = [self._week_of_month(idx.date()) for idx in self.df.index]
        else:
            self.df['week_of_month'] = pd.Series(dtype='object')

        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        self.df['is_month_start'] = self.df['day_of_month'].isin([1, 2, 3]).astype(int)
        self.df['is_month_end'] = self.df.index.is_month_end.astype(int)

        calendar_features: list[str] = ['day_of_week', 'day_of_month', 'day_of_year',
                                        'week_of_year', 'week_of_month', 'month', 'quarter', 'year',
                                        'is_weekend', 'is_month_start', 'is_month_end']
        self.features['calendar_features'] = sorted(list(set(calendar_features)))
        self.logger.info(f"Calendar features created: {calendar_features}")

    def _create_cyclical_features(self) -> None:
        """
        Creates cyclical (sin/cos) features for temporal components.

        Transforms 'day_of_week', 'month', and 'day_of_year' (if present in `self.df`)
        into their respective sine and cosine components for better model representation.
        The names of the created features are added to `self.features['cyclical_features']`.
        """
        self.logger.info("Creating cyclical time features...")
        if 'day_of_week' in self.df.columns:
            self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
            self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        if 'month' in self.df.columns:
            self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        if 'day_of_year' in self.df.columns:
            days_in_year: np.ndarray = np.where(self.df.index.is_leap_year, 366, 365)
            self.df['day_of_year_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / days_in_year)
            self.df['day_of_year_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / days_in_year)

        cyclical_features: list[str] = [col for col in self.df.columns if '_sin' in col or '_cos' in col]
        self.features['cyclical_features'] = sorted(list(set(cyclical_features)))
        self.logger.info(f"Cyclical features created: {self.features['cyclical_features']}")

    def _create_lag_features(self) -> None:
        """
        Creates lag features for target columns (`self.target_cols`).

        For each target column, new columns are created with values shifted by periods
        specified in the `LAGS` constant. Rows with NaNs from shifting are dropped.
        Created lag feature names are added to `self.features['lag_features']`.
        """
        self.logger.info(f"Creating lag features for {self.target_cols} with lags: {LAGS}...")
        lag_cols: list[str] = []
        for col in self.target_cols:
            if col not in self.df.columns:
                self.logger.warning(f"Target column '{col}' not found in DataFrame.")
                continue
            for lag in LAGS:
                lag_col_name: str = f'{col}_lag_{lag}'
                self.df[lag_col_name] = self.df[col].shift(lag)
                lag_cols.append(lag_col_name)

        if lag_cols:
            initial_rows: int = self.df.shape[0]
            self.df.dropna(subset=lag_cols, inplace=True)
            rows_dropped: int = initial_rows - self.df.shape[0]
            self.logger.info(
                f"Dropped {rows_dropped} rows due to NaNs in lag features (max lag: {max(LAGS)}).")

        self.features['lag_features'] = sorted(list(set(lag_cols)))
        self.logger.info(f"Created {len(lag_cols)} lag features.")

    @staticmethod
    def _rolling_mad(x: np.ndarray) -> float:
        """
        Calculates Median Absolute Deviation (MAD) for a rolling window.
        """
        if len(x) == 0 or np.all(np.isnan(x)):
            return np.nan
        median: float = np.nanmedian(x)
        return np.nanmedian(np.abs(x - median))

    @staticmethod
    def _rolling_trimmed_mean(x: np.ndarray, proportiontocut: float = 0.1) -> float:
        """
        Computes trimmed mean for a rolling window.

        Removes the lowest and highest values by the given `proportiontocut`.
        NaNs are removed before trimming.
        """
        x_clean: np.ndarray = x[~np.isnan(x)]
        if len(x_clean) == 0:
            return np.nan
        n: int = len(x_clean)
        lowercut: int = int(n * proportiontocut)
        uppercut: int = n - lowercut
        if lowercut >= uppercut:
            return np.nanmedian(x_clean)
        x_sorted: np.ndarray = np.sort(x_clean)
        trimmed_x: np.ndarray = x_sorted[lowercut:uppercut]
        return np.mean(trimmed_x) if len(trimmed_x) > 0 else np.nan

    def _create_rolling_window_features(self) -> None:
        """
        Creates rolling window features (standard and robust) for target columns.

        For each target column (`self.target_cols`) and each window size from the `WINDOWS` constant,
        it calculates various statistical metrics (mean, std, sum, median, skewness, kurtosis,
        quantiles, IQR, MAD, trimmed mean). A shift of 1 (`shift(1)`) is used to prevent data leakage.
        NaNs that appear at the start are filled with zeros.
        Created feature names are added to `self.features['rolling_cols']`.
        """
        self.logger.info(f"Creating rolling window features for {self.target_cols} with windows: {WINDOWS}...")
        rolling_cols: list[str] = []

        for col in self.target_cols:
            if col not in self.df.columns:
                self.logger.warning(f"Target column '{col}' is missing for rolling window features.")
                continue
            for window in WINDOWS:
                shifted_series: pd.Series = self.df[col].shift(1)  # shift to prevent data leakage

                # Standard statistics
                ops: dict[str, Any] = {'mean': np.mean, 'std': np.std, 'sum': np.sum,
                                       'median': np.median, 'skew': pd.Series.skew, 'kurt': pd.Series.kurt,
                                       'q25': lambda x: np.percentile(x, 25) if not np.all(np.isnan(x)) and len(
                                           x[~np.isnan(x)]) > 0 else np.nan,
                                       'q75': lambda x: np.percentile(x, 75) if not np.all(np.isnan(x)) and len(
                                           x[~np.isnan(x)]) > 0 else np.nan,
                                       }
                for op_name, op_func in ops.items():
                    r_col_name: str = f'{col}_roll_{op_name}_{window}'
                    if op_name in ['skew', 'kurt']:
                        self.df[r_col_name] = shifted_series.rolling(window=window, closed='left').agg(op_name)
                    else:
                        self.df[r_col_name] = shifted_series.rolling(window=window, closed='left').agg(op_func)
                    rolling_cols.append(r_col_name)

                # IQR
                q75_col: str = f'{col}_roll_q75_{window}'
                q25_col: str = f'{col}_roll_q25_{window}'
                if q75_col in self.df.columns and q25_col in self.df.columns:
                    self.df[f'{col}_roll_iqr_{window}'] = self.df[q75_col] - self.df[q25_col]
                    rolling_cols.append(f'{col}_roll_iqr_{window}')

                # Robust stats
                self.df[f'{col}_roll_mad_{window}'] = shifted_series.rolling(window=window, closed='left').apply(
                    self._rolling_mad, raw=True)
                rolling_cols.append(f'{col}_roll_mad_{window}')
                self.df[f'{col}_roll_trimmean_{window}'] = shifted_series.rolling(window=window, closed='left').apply(
                    lambda x: self._rolling_trimmed_mean(x, 0.1), raw=True)
                rolling_cols.append(f'{col}_roll_trimmean_{window}')

        self.df.fillna(0, inplace=True)
        self.logger.info(f"Total rolling features created: {len(rolling_cols)}")
        self.features['rolling_cols'] = sorted(list(set(rolling_cols)))

    def _create_time_since_and_frequency_features(self) -> None:
        """
        Creates 'time since last spending' and 'spending frequency' features for target columns.

        For each target column:
        1. 'time_since_last': days since last non-zero spending.
        2. 'freq_{window}': number of non-zero spendings in rolling windows of 30, 60, and 90 days (with shift).
        NaNs are filled with 0.
        Created feature names are added to `self.features['times_since_cols']` and `self.features['freq_cols']`.
        """
        self.logger.info(f"Creating 'time since last' and frequency features for {self.target_cols}...")
        time_since_cols: list[str] = []
        freq_cols: list[str] = []

        for col in self.target_cols:
            if col not in self.df.columns:
                self.logger.warning(f"Target column '{col}' is missing for time/frequency features.")
                continue

            time_since_col: str = f'{col}_time_since_last'
            spending_days: pd.Series = self.df[col] > 0
            cum_days: pd.Series = (~spending_days).cumsum()
            self.df[time_since_col] = cum_days - cum_days.where(spending_days).ffill().fillna(0)
            time_since_cols.append(time_since_col)

            for window in [30, 60, 90]:
                freq_col_name: str = f'{col}_freq_{window}'
                self.df[freq_col_name] = (self.df[col] > 0).shift(1).rolling(window=window, closed='left').sum().fillna(
                    0)
                freq_cols.append(freq_col_name)

        self.df.fillna(0, inplace=True)
        self.features['times_since_cols'] = sorted(list(set(time_since_cols)))
        self.features['freq_cols'] = sorted(list(set(freq_cols)))
        self.logger.info(f"Created {len(time_since_cols)} 'time since last' and {len(freq_cols)} frequency features.")

    def _create_domain_specific_features(self, irregular_spend_col: str = 'IrregularSpendings',
                                         non_periodic_spend_col: str = 'NonPeriodicSpendings') -> None:
        """
        Creates domain-specific features.

        1. Spending ratio: share of each target spending category to 'AllListedSpendings'
           over a rolling window `WINDOW_RATIO`.
        2. Large spend impact: creates 'large_spend_yesterday' binary flag for large non-target spendings (95th percentile),
           and rolling sum of non-target spendings over `WINDOW_NON_TARGET`.

        NaNs are filled with 0. Feature names are added to `self.features['domain_features']`.
        """
        self.logger.info("Creating domain-specific features...")
        domain_features: list[str] = []

        total_spend_col_rolling: str = 'AllListedSpendings_roll_sum_30'
        if 'AllListedSpendings' in self.df.columns:
            self.df[total_spend_col_rolling] = self.df['AllListedSpendings'].shift(1).rolling(window=WINDOW_RATIO,
                                                                                              closed='left').sum().fillna(
                0)
        else:
            self.df[total_spend_col_rolling] = 0
            self.logger.warning("Missing 'AllListedSpendings' for ratio calculations.")

        ratio_cols_created: list[str] = []
        for col in self.target_cols:
            if col not in self.df.columns:
                self.logger.warning(f"Target column '{col}' is missing for ratio features.")
                continue
            ratio_col_name: str = f'{col}_ratio_{WINDOW_RATIO}d'
            category_rolling_sum_col: str = f'{col}_roll_sum_{WINDOW_RATIO}'
            if category_rolling_sum_col in self.df.columns and total_spend_col_rolling in self.df.columns:
                self.df[ratio_col_name] = self.df[category_rolling_sum_col].divide(
                    self.df[total_spend_col_rolling] + 1e-9)
                self.df[ratio_col_name].fillna(0, inplace=True)
                self.df[ratio_col_name].replace([np.inf, -np.inf], 0, inplace=True)
                ratio_cols_created.append(ratio_col_name)
            else:
                self.logger.warning(
                    f"Missing columns for ratio feature {ratio_col_name} ({category_rolling_sum_col} or {total_spend_col_rolling}).")

        domain_features.extend(ratio_cols_created)
        if total_spend_col_rolling in self.df.columns:
            self.df.drop(columns=[total_spend_col_rolling], inplace=True)

        non_target_category_cols: list[str] = []
        if irregular_spend_col in self.df.columns: non_target_category_cols.append(irregular_spend_col)
        if non_periodic_spend_col in self.df.columns: non_target_category_cols.append(non_periodic_spend_col)

        if non_target_category_cols:
            all_non_target_spends_flat: pd.Series = self.df[non_target_category_cols].unstack().replace(0,
                                                                                                        np.nan).dropna()
            large_spend_threshold: float
            if not all_non_target_spends_flat.empty:
                large_spend_threshold = all_non_target_spends_flat.quantile(0.95)
            else:
                large_spend_threshold = np.inf
            self.logger.info(f"95th percentile threshold for large non-target spending: {large_spend_threshold:.2f}")

            self.df['large_spend_yesterday'] = (self.df[non_target_category_cols].shift(1) > large_spend_threshold).any(
                axis=1).astype(int)
            domain_features.append('large_spend_yesterday')

            non_target_sum_col: str = f'non_target_spend_sum_{WINDOW_NON_TARGET}d'
            self.df[non_target_sum_col] = self.df[non_target_category_cols].shift(1).rolling(
                window=WINDOW_NON_TARGET, closed='left').sum().sum(axis=1).fillna(0)
            domain_features.append(non_target_sum_col)
        else:
            self.logger.warning("No non-target columns found for large spending impact features.")
            self.df['large_spend_yesterday'] = 0
            domain_features.append('large_spend_yesterday')
            self.df[f'non_target_spend_sum_{WINDOW_NON_TARGET}d'] = 0
            domain_features.append(f'non_target_spend_sum_{WINDOW_NON_TARGET}d')

        self.df.fillna(0, inplace=True)
        self.features['domain_features'] = sorted(list(set(domain_features)))
        self.logger.info(f"Domain-specific feature creation completed. Features: {self.features['domain_features']}")

    def engineer_features(self,
                          input_df: pd.DataFrame,
                          column_categories: dict[str, list[str]],
                          target_cols: list[str],
                          including_secondary_features: list[str] | None = None
                          ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        Main method for performing feature engineering.

        Processes the input DataFrame to create various features: aggregated,
        log-transformed, calendar, cyclical, lagged, rolling window,
        time-since-last-event, frequency, and domain-specific.
        Fills in missing dates and handles NaNs.
        """
        self.logger.info("Starting feature engineering process...")
        self.df = input_df.copy()
        self.target_cols = target_cols
        if including_secondary_features is not None:
            self.including_secondary_features = including_secondary_features

        if not isinstance(self.df.index, pd.DatetimeIndex):
            try:
                self.df['date'] = pd.to_datetime(self.df['date'])
                self.df.set_index('date', inplace=True)
                self.logger.info("Set 'date' as DatetimeIndex.")
            except KeyError:
                self.logger.error("Column 'date' not found in input DataFrame.")
                raise
            except Exception as e:
                self.logger.error(f"Failed to set DatetimeIndex: {e}")
                raise

        # Get column lists from categories
        regular_cols: list[str] = column_categories.get("regular", [])
        irregular_cols: list[str] = column_categories.get("above_80_nan", [])
        non_periodic_cols: list[str] = column_categories.get("above_95_nan", [])

        self.logger.info(f"Regular spendings: {regular_cols}")
        self.logger.info(f"Irregular spendings (80%+ NaN): {irregular_cols}")
        self.logger.info(f"Non-periodic spendings (95%+ NaN): {non_periodic_cols}")
        self.logger.info(f"Target columns: {self.target_cols}")

        # 1. Aggregate spending categories
        self._aggregate_spending_categories(irregular_cols, non_periodic_cols)

        # Filter columns if DROP_SECONDARY_FEATURES is enabled
        if DROP_SECONDARY_FEATURES:
            subtotal_cols: list[str] = ['IrregularSpendings', 'NonPeriodicSpendings', 'AllListedSpendings']
            cols_to_keep: list[str] = self.target_cols + \
                                      [col for col in (non_periodic_cols + irregular_cols) if
                                       col in self.including_secondary_features] + \
                                      subtotal_cols
            cols_to_keep = sorted(list(set(c for c in cols_to_keep if c in self.df.columns)))
            self.df = self.df[cols_to_keep].copy()
            self.logger.info(f"Columns kept after DROP_SECONDARY_FEATURES: {cols_to_keep}")

        # 2. Initialize feature dictionary
        aggregations_in_df: list[str] = [col for col in
                                         ['IrregularSpendings', 'NonPeriodicSpendings', 'AllListedSpendings'] if
                                         col in self.df.columns]
        other_non_aggregated_features_in_df: list[str] = [col for col in self.df.columns if
                                                          col not in self.target_cols and col not in aggregations_in_df]

        self.features = {
            'target': [col for col in self.target_cols if col in self.df.columns],
            'other_non_aggregated_features': other_non_aggregated_features_in_df,
            'aggregated_features': aggregations_in_df
        }
        self.logger.info(
            f"Initial feature groups: target({len(self.features['target'])}), other_non_aggregated({len(self.features['other_non_aggregated_features'])}), aggregated({len(self.features['aggregated_features'])}).")

        # 3. Check index and fill missing dates
        self._check_datetime_index_continuity()
        self._fill_missing_dates()
        self._check_datetime_index_continuity()

        # 4. Log transformation
        self._apply_log_transformation()
        self._check_features(verbose=True)

        # 5. Stationarity check (analytical step)
        self._check_stationarity()

        # 6. Calendar features
        self._create_calendar_features()
        self._check_features()

        # 7. Cyclical features
        self._create_cyclical_features()
        self._check_features()

        # 8. Lag features
        self._create_lag_features()
        self._check_features()
        self._check_datetime_index_continuity()

        # 9. Rolling window features
        self._create_rolling_window_features()
        self._check_features(verbose=True)
        self._check_datetime_index_continuity()

        # 10. Time-since and frequency features
        self._create_time_since_and_frequency_features()
        self._check_features()

        irregular_col_for_domain: str = 'IrregularSpendings'
        if self.features['aggregated_features'] and 'IrregularSpendings' in self.features['aggregated_features']:
            irregular_col_for_domain = 'IrregularSpendings'

        non_periodic_col_for_domain: str = 'NonPeriodicSpendings'
        if len(self.features['aggregated_features']) > 1 and 'NonPeriodicSpendings' in self.features[
            'aggregated_features']:
            non_periodic_col_for_domain = 'NonPeriodicSpendings'

        self._create_domain_specific_features(
            irregular_spend_col=self.features['aggregated_features'][0] if len(
                self.features['aggregated_features']) > 0 and 'IrregularSpendings' in self.features[
                                                                               'aggregated_features'] else 'IrregularSpendings',
            non_periodic_spend_col=self.features['aggregated_features'][1] if len(
                self.features['aggregated_features']) > 1 and 'NonPeriodicSpendings' in self.features[
                                                                                  'aggregated_features'] else 'NonPeriodicSpendings'
        )
        self._check_features()

        if self.df.isna().sum().sum() > 0:
            self.logger.warning(
                f"Found {self.df.isna().sum().sum()} NaN values before final return. Filling with zeros...")
            self.df.fillna(0, inplace=True)

        self.logger.info("Feature engineering process completed.")
        return self.df.copy(), self.features.copy()

def main():
    print("Starting the test scenario for FeatureEngineer...")
    try:
        file_path_data_folder = pathlib.Path(__file__).parents[3].resolve() / 'data'
        if not file_path_data_folder.exists():
            current_path = pathlib.Path('.').resolve()
            if (current_path / 'data').exists():
                file_path_data_folder = current_path / 'data'
            elif (current_path.parent / 'data').exists():
                file_path_data_folder = current_path.parent / 'data'
            else:
                raise FileNotFoundError("The 'data' folder was not found.")

        df_pickle_path: pathlib.Path = file_path_data_folder / 'intermediate/data_prepare.pkl'
        categories_json_path: pathlib.Path = file_path_data_folder / 'intermediate/columns_cat.json'
        correlated_features_path: pathlib.Path = file_path_data_folder / 'intermediate/correlated_features.txt'  # Not used

        engineered_df_output_path: pathlib.Path = file_path_data_folder / 'intermediate/engineered_df.pkl'
        features_dict_output_path: pathlib.Path = file_path_data_folder / 'intermediate/features_dict.json'

        df = pd.read_pickle(df_pickle_path)

        with open(categories_json_path, 'r') as f:
            column_categories_main: dict[str, list[str]] = json.load(f)

        with open(correlated_features_path, 'r') as f:  # Not used
            correlated_features_main: list[str] = [line.strip() for line in f.readlines() if line.strip()]

        target_cols_main: list[str] = column_categories_main.get('regular', [])

        engineer = FeatureEngineer()

        engineered_df, features_dict = engineer.engineer_features(
            input_df=df,
            column_categories=column_categories_main,
            target_cols=target_cols_main,
            including_secondary_features=correlated_features_main
        )

        engineered_df.to_pickle(engineered_df_output_path)
        with open(features_dict_output_path, 'w') as f:
            json.dump(features_dict, f, indent=4)  # Added indent for better readability of json

        print(f"DataFrame size after feature engineering: {engineered_df.shape}")
        print(f"Total NaN values in final DataFrame: {engineered_df.isna().sum().sum()}")

        print(f"\n--- Dictionary of created features (keys and counts) ---")
        for key, val_list in features_dict.items():
            print(f"- {key}: ({len(val_list)} features)")

        print(f"\n--- Checking some expected columns ---")
        if target_cols_main:  # Check if there are target columns
            expected_example_cols: list[str] = [
                f"{target_cols_main[0]}_log",
                f"{target_cols_main[0]}_lag_{LAGS[0] if LAGS else 'N/A'}",
                f"{target_cols_main[0]}_roll_mean_{WINDOWS[0] if WINDOWS else 'N/A'}",
                'day_of_week_sin',
                'IrregularSpendings'  # This column is created even if irregular_cols is empty (with a value of 0.0)
            ]
            for ec in expected_example_cols:
                if 'N/A' in ec:  # Skip if LAGS or WINDOWS are empty
                    print(f"Column '{ec}' cannot be checked (LAGS/WINDOWS are empty).")
                    continue
                status: str = "Present" if ec in engineered_df.columns else "MISSING"
                print(f"Column '{ec}': {status}")
        else:
            print("No target columns to check for examples.")

    except FileNotFoundError as e_fnf:
        print(f"File path error: {e_fnf}. Please ensure the data files are in the correct directory.")
        print(f"Expected structure: <project_root>/data/intermediate/...")
        print(f"Current working directory: {pathlib.Path.cwd()}")
    except Exception as e:
        print(f"An error occurred during the test run of feature engineering: {e}")
        import traceback
        print(traceback.format_exc())

    print("FeatureEngineer test scenario completed.")


if __name__ == '__main__':
    main()