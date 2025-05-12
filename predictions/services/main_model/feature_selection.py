import json
import logging
import pathlib
import warnings
from typing import Any, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import Bunch

from predictions.services.main_model.utils import get_logger, select_features_multicollinearity_graph_target_corr

from personal_budget_planner.constants import *


class FeatureSelector:
    """
    Class for performing feature selection based on various methods:
    correlation analysis, statistical tests, mutual information,
    RandomForest permutation importance.
    """

    def __init__(self, logger: logging.Logger | None = None):
        """
        Initialize FeatureSelector.
        """
        self.logger: logging.Logger = logger if logger else get_logger(self.__class__.__name__,
                                                                       log_file_name="feature_selector.log")
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message="A value is trying to be set on a copy of a slice from a DataFrame")

    def _perform_calendar_feature_analysis(self, df: pd.DataFrame, feature_lists: dict[str, list[str]],
                                           target_cols_to_use: list[str]) -> list[str]:
        """
        Performs statistical analysis on the impact of calendar features (Kruskal-Wallis test).
        """
        self.logger.info("Performing statistical analysis on the impact of calendar features (Kruskal-Wallis test)...")

        calendar_features_to_test: list[str] = feature_lists.get('calendar_features', [])
        calendar_features_to_test = [feat for feat in calendar_features_to_test if feat in df.columns]

        if not calendar_features_to_test:
            self.logger.warning("No calendar features available for testing in the DataFrame.")
            return CALENDAR_COLUMNS_TO_DROP_POST_ANALYSIS

        significant_results_summary: dict[str, list[str]] = {}
        for cal_feature in calendar_features_to_test:
            significant_results_summary[cal_feature] = []
            for target_col in target_cols_to_use:
                if target_col not in df.columns: continue

                unique_values: np.ndarray = df[cal_feature].unique()
                data_groups: list[pd.Series] = [df[df[cal_feature] == val][target_col].dropna() for val in
                                                unique_values]
                data_groups = [group for group in data_groups if not group.empty and len(group) > 1]

                if len(data_groups) < 2:
                    self.logger.debug(
                        f"  For '{target_col}': Insufficient groups ({len(data_groups)}) for testing the impact of '{cal_feature}'.")
                    continue
                try:
                    stat: float
                    p_value: float
                    stat, p_value = stats.kruskal(*data_groups)
                    if p_value < ALPHA_SIGNIFICANCE:
                        significant_results_summary[cal_feature].append(target_col)
                except ValueError as e:
                    self.logger.warning(f"  '{target_col}' vs '{cal_feature}': Kruskal-Wallis test error - {e}")

        calendar_features_to_drop = list(filter(lambda x: x not in significant_results_summary[cal_feature],
                                                calendar_features_to_test))
        self.logger.info(
            f"Scheduled removal of calendar columns: {calendar_features_to_drop}")
        return calendar_features_to_drop

    def _filter_by_sparsity_and_variance(self, X_df: pd.DataFrame) -> list[str]:
        """
        Filters features by sparsity and variance of non-zero values.
        """
        self.logger.info("Filtering features by sparsity and variance...")
        # Filter by sparsity
        non_zero_fraction: pd.Series = (X_df != 0).sum() / len(X_df)
        features_after_sparsity: list[str] = non_zero_fraction[
            non_zero_fraction >= MIN_NON_ZERO_FRACTION].index.tolist()
        dropped_sparsity_count: int = len(X_df.columns) - len(features_after_sparsity)
        self.logger.info(
            f"Removed features due to high sparsity (< {MIN_NON_ZERO_FRACTION * 100}% non-zero): {dropped_sparsity_count}")

        # Filter by variance of non-zero values
        features_to_check_variance: list[str] = [f for f in features_after_sparsity if
                                                 X_df[f].dtype in ['float64', 'int64', 'float32', 'int32']]
        features_after_variance: list[str] = []
        for feature in features_to_check_variance:
            non_zero_values: pd.Series = X_df.loc[X_df[feature] != 0, feature]
            if len(non_zero_values) > 1 and non_zero_values.var() >= MIN_VARIANCE_NON_ZERO:
                features_after_variance.append(feature)

        # Add non-numeric features that passed the sparsity filter
        non_numeric_passed_sparsity: list[str] = [f for f in features_after_sparsity if
                                                  f not in features_to_check_variance]
        features_after_variance.extend(non_numeric_passed_sparsity)

        dropped_variance_count: int = len(features_after_sparsity) - len(features_after_variance)
        self.logger.info(f"Removed features due to low variance of non-zero values: {dropped_variance_count}")

        return sorted(list(set(features_after_variance)))

    def _calculate_mutual_information(self, X_df: pd.DataFrame, y_df: pd.DataFrame, target_cols_for_mi: list[str]) -> \
            list[str]:
        """
        Calculates mutual information and filters features.
        """
        self.logger.info("Calculating mutual information and filtering features...")
        mi_scores_dict: dict[str, pd.Series] = {}
        for target in target_cols_for_mi:
            if target in y_df.columns and not X_df.empty:
                mi_np_array: np.ndarray = mutual_info_regression(X_df, y_df[target].fillna(0), random_state=42,
                                                                 n_neighbors=min(5, len(X_df) - 1) if len(
                                                                     X_df) > 1 else 1)
                mi_scores_dict[target] = pd.Series(mi_np_array, index=X_df.columns)

        if not mi_scores_dict:
            self.logger.warning("Failed to calculate MI Scores. Maybe there are no target columns or X_df is empty.")
            return X_df.columns.tolist()

        mi_scores_df: pd.DataFrame = pd.DataFrame(mi_scores_dict).fillna(0)
        max_mi_score: pd.Series = mi_scores_df.max(axis=1)
        mi_threshold: float = max_mi_score.quantile(0.25)  # 25th percentile as threshold

        features_after_mi: list[str] = max_mi_score[max_mi_score > mi_threshold].index.tolist()
        dropped_mi_count: int = len(X_df.columns) - len(features_after_mi)
        self.logger.info(f"Removed features due to low mutual information (< {mi_threshold:.2e}): {dropped_mi_count}")
        return features_after_mi

    def _get_permutation_importance(self, X_df: pd.DataFrame, y_df: pd.DataFrame) -> list[str]:
        """Calculates RandomForest permutation importance and filters features.
        """
        self.logger.info("Calculating RandomForest permutation importance...")
        if X_df.empty or y_df.empty:
            self.logger.warning("X_df or y_df is empty. Skipping permutation importance calculation.")
            return X_df.columns.tolist()

        base_estimator_rf = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF, random_state=42, n_jobs=-1
        )
        multi_output_rf = MultiOutputRegressor(base_estimator_rf, n_jobs=-1)

        self.logger.info("Training RandomForestRegressor for Permutation Importance...")
        multi_output_rf.fit(X_df, y_df.fillna(0))

        self.logger.info("Calculating permutation importance...")
        perm_importance_result: Bunch = permutation_importance(
            multi_output_rf, X_df, y_df.fillna(0),
            n_repeats=RF_N_REPEATS_PERMUTATION, random_state=42, n_jobs=-1
        )
        perm_importances_df: pd.Series = pd.Series(perm_importance_result.importances_mean,
                                                   index=X_df.columns).sort_values(
            ascending=False)

        features_after_rf: list[str] = perm_importances_df[
            perm_importances_df > IMPORTANCE_SCORE_THRESHOLD].index.tolist()
        dropped_rf_count: int = len(X_df.columns) - len(features_after_rf)
        self.logger.info(
            f"Removed features based on permutation importance threshold (< {IMPORTANCE_SCORE_THRESHOLD:.1e}): {dropped_rf_count}")
        return features_after_rf

    def select_features(self,
                        df_processed: pd.DataFrame,
                        target_names: list[str],
                        test_set_size_days: int = TEST_SIZE_DAYS,
                        use_log_transformed_targets: bool = False,
                        feature_lists_dict: dict[str, list[str]] | None = None,
                        feature_list_input: list[str] | None = None
                        ) -> list[str]:
        self.logger.info("Starting the feature selection process...")
        df: pd.DataFrame = df_processed.copy()

        _feature_lists_internal: Union[dict[str, list[str]], list[str]]
        if not feature_list_input:
            if feature_lists_dict:
                # Ensure the values are lists of strings for consistency
                _feature_lists_internal = {
                    k: [str(item) for item in v] if isinstance(v, list) else v
                    for k, v in feature_lists_dict.items()
                }
            else:
                raise RuntimeError('No feature list is given (neither feature_lists_dict nor feature_list_input)')
        else:
            _feature_lists_internal = feature_list_input

        if not isinstance(_feature_lists_internal, dict):
            self.logger.warning(
                "feature_list_input (list) was provided. Subsequent operations might expect a dictionary of features.")
            if feature_list_input:
                pass

        feature_lists: dict[str, list[str]]
        if isinstance(_feature_lists_internal, dict):
            feature_lists = _feature_lists_internal
        else:
            self.logger.error(
                "Feature processing expects a dictionary structure for feature_lists, but received a list.")
            feature_lists = {}

        # 1. "Patch" for logarithmic targets in feature_lists
        current_target_cols_log: list[str] = [f'{col}_log' for col in target_names]
        if 'log_columns' in feature_lists:
            feature_lists['log_columns'] = [f for f in feature_lists.get('log_columns', []) if
                                            f not in current_target_cols_log]
        feature_lists['log_target'] = current_target_cols_log

        # Determining which target columns to use
        targets_to_use: list[str] = current_target_cols_log if use_log_transformed_targets else target_names
        targets_to_use = [t for t in targets_to_use if t in df.columns]  # Only existing columns
        if not targets_to_use:
            self.logger.error("Target columns not found in DataFrame. Feature selection is not possible.")
            return []

        numerical_features_for_corr: list[str] = []
        for key in ['lag_features', 'rolling_cols', 'times_since_cols', 'freq_cols', 'domain_features',
                    'calendar_features', 'cyclical_features']:
            numerical_features_for_corr.extend(feature_lists.get(key, []))

        # 2. Analyzing and removing calendar features
        calendar_targets_for_analysis: list[str] = current_target_cols_log
        calendar_targets_for_analysis = [t for t in calendar_targets_for_analysis if t in df.columns]

        cols_to_drop_calendar: list[str] = self._perform_calendar_feature_analysis(df, feature_lists,
                                                                                   calendar_targets_for_analysis)

        df.drop(columns=[col for col in cols_to_drop_calendar if col in df.columns], inplace=True, errors='ignore')
        if 'calendar_features' in feature_lists:  # Updating the list of calendar features
            feature_lists['calendar_features'] = [f for f in feature_lists.get('calendar_features', []) if
                                                  f not in cols_to_drop_calendar]
        self.logger.info(
            f"Calendar features removed: {cols_to_drop_calendar}. Remaining calendar features: {feature_lists.get('calendar_features', [])}")

        # 3. Creating a complete list of current features
        all_current_features: list[str] = []
        if isinstance(feature_lists, dict):
            for key, f_list in feature_lists.items():
                if key not in ['target', 'log_target']:  # Exclude target columns from the X feature list
                    if isinstance(f_list, list): all_current_features.extend(f_list)
        elif isinstance(feature_lists, list):
            all_current_features = [f for f in feature_lists if
                                    f not in targets_to_use and f not in current_target_cols_log and f in df.columns]

        all_current_features = sorted(list(set(f for f in all_current_features if f in df.columns)))
        self.logger.info(f"Total number of potential features for selection: {len(all_current_features)}")

        # 4. Splitting into train and test sets (for selection based on training data)
        if df.index.max() is None or len(df) == 0:  # Check for empty df
            self.logger.error("DataFrame is empty or does not have a time index for splitting.")
            return []

        split_date: pd.Timestamp = df.index.max() - pd.Timedelta(days=test_set_size_days)
        df_train: pd.DataFrame = df[df.index <= split_date].copy()

        X_train_selection: pd.DataFrame = df_train[all_current_features]
        y_train_selection: pd.DataFrame = df_train[targets_to_use]

        if X_train_selection.empty:
            self.logger.error("X_train_selection is empty after splitting. Check test_set_size_days and data.")
            return []

        # 5. Filtering by sparsity and variance (on X_train_selection)
        features_after_sparsity_variance: list[str] = self._filter_by_sparsity_and_variance(
            X_train_selection.copy())
        X_train_selection = X_train_selection[features_after_sparsity_variance]
        self.logger.info(
            f"Number of features after sparsity/variance filtering: {len(X_train_selection.columns)}")

        # 6. Handling multicollinearity (on X_train_selection)
        features_after_multicollinearity: list[str] = select_features_multicollinearity_graph_target_corr(
            X_train_selection.copy(), y_train_selection.copy(),
            PREDICTOR_CORRELATION_THRESHOLD,
            MIN_TARGET_CORRELATION_FOR_GROUP_REPRESENTATIVE
        )
        X_train_selection = X_train_selection[features_after_multicollinearity]
        self.logger.info(f"Number of features after multicollinearity processing: {len(X_train_selection.columns)}")

        # 7. Feature selection based on mutual information (on current X_train_selection)
        features_after_mi: list[str] = self._calculate_mutual_information(X_train_selection.copy(),
                                                                          y_train_selection.copy(),
                                                                          targets_to_use)
        X_train_selection = X_train_selection[features_after_mi]
        self.logger.info(f"Number of features after mutual information filtering: {len(X_train_selection.columns)}")

        # 8. Feature selection based on RandomForest permutation importance (on current X_train_selection)
        final_selected_features: list[str] = self._get_permutation_importance(X_train_selection.copy(),
                                                                              y_train_selection.copy())
        self.logger.info(f"Final number of selected features: {len(final_selected_features)}")

        self.logger.info("Feature selection process completed.")
        return final_selected_features

def main():
    print("Starting the test scenario for FeatureSelection...")

    file_path_data_folder = pathlib.Path(__file__).parents[3].resolve() / 'data'

    df: pd.DataFrame = pd.read_pickle(file_path_data_folder / 'intermediate/engineered_df.pkl')

    features_dict: dict[str, Any]
    with open(file_path_data_folder / 'intermediate/features_dict.json', 'r') as f:
        features_dict = json.load(f)

    typed_features_dict: dict[str, list[str]] = {}
    if isinstance(features_dict, dict):
        for key, value in features_dict.items():
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                typed_features_dict[key] = value
            elif key == 'target' and isinstance(value, list):
                typed_features_dict[key] = [str(item) for item in value]

    if 'target' not in typed_features_dict:
        print("Error: 'target' not found or has an incorrect format in features_dict.json")
    else:
        selected_features: list[str] = FeatureSelector().select_features(
            df_processed=df,
            target_names=typed_features_dict['target'],
            feature_lists_dict=typed_features_dict
        )

        print(f'{len(selected_features)} features were selected during processing')

        with open(file_path_data_folder / 'intermediate/selected_features.json', 'w') as f_out:
            json.dump({'selected_features': selected_features}, f_out)

if __name__ == '__main__':
    main()