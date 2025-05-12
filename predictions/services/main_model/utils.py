import logging
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def get_logger(name: str,
               log_file_name: str = 'log.log',
               level: int = logging.INFO) -> logging.Logger:
    """
    Creates and configures a logger that writes to both a file and the console.
    """
    # Create the directory for logs if it does not exist
    log_dir = pathlib.Path(__file__).parents[1].resolve() / 'logs'
    log_file_path = log_dir / log_file_name

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Handler for writing logs to a file
    try:
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
    except Exception as e:
        sys.stderr.write(f"Error creating file handler for log {log_file_name}: {e}\n")
        file_handler = None

    # Handler for outputting logs to the console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if file_handler:
        logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False  # Avoid duplicate logs if the root logger is set up
    return logger


def check_datetime_index_continuity(df: pd.DataFrame,
                                    expected_freq: str = 'D',
                                    verbose: bool = True) -> Tuple[bool, Optional[pd.DatetimeIndex]]:
    """
    Checks the continuity of a DataFrame's DatetimeIndex.

    It verifies if all dates within the observed range (min to max date in the index)
    are present, according to the expected frequency.
    """
    if verbose:
        print("\n--- Checking DatetimeIndex Continuity ---")

    if not isinstance(df.index, pd.DatetimeIndex):
        if verbose:
            print("WARNING: The DataFrame index is not a DatetimeIndex.")
        return False, None

    if df.empty:
        if verbose:
            print("INFO: The DataFrame is empty. Continuity check is not needed (assumed continuous).")
        return True, None

    # Ensure the index is sorted for correct min(), max(), and difference() operations.
    df_sorted_index = df.index.sort_values()

    actual_min_date = df_sorted_index.min()
    actual_max_date = df_sorted_index.max()

    if verbose:
        print(f"Actual date range in the DataFrame: from {actual_min_date} to {actual_max_date}")
        print(f"Actual duration of the date range: {actual_max_date - actual_min_date}")
        print(f"Number of unique dates in the index: {len(df_sorted_index.unique())}")

    # Create the ideal, continuous date range based on actual min/max and expected frequency.
    ideal_full_range = pd.date_range(start=actual_min_date, end=actual_max_date, freq=expected_freq)

    if verbose:
        print(f"Expected number of dates in this range (frequency '{expected_freq}'): {len(ideal_full_range)}")

    # Find dates present in the ideal range but missing from the actual index.
    missing_dates = ideal_full_range.difference(df_sorted_index)

    if not missing_dates.empty:
        if verbose:
            print(f"MISSING DATES FOUND! Number: {len(missing_dates)}")
        return False, missing_dates
    else:
        if verbose:
            print("INFO: The DatetimeIndex is continuous within its range.")
        return True, None

def check_features(df: pd.DataFrame, features_dict: Dict[str, List[str]], verbose: bool = False) -> Optional[List[str]]:
    """
    Checks if all DataFrame columns are included in the features dictionary.
    """
    features_counter = 0
    print(' --- Features check ---')
    for k, v in features_dict.items():
        print(f"{k}: {len(v)}")
        features_counter += len(v)
    print(' --------------------- ')
    print(f'Total number of features in the dictionary: {features_counter}')
    print(f'Total number of columns in the DataFrame: {len(df.columns)}')

    if features_counter < len(df.columns):
        temp_all_features = sum([list(v) for _, v in features_dict.items()], [])
        not_included_features = list(filter(lambda x: x not in temp_all_features, df.columns.to_list()))
        not_included_count = len(df.columns) - features_counter
        if not_included_count <= 10 or verbose:
            print(f'Not included in the dictionary: {not_included_features}')
        else:
            print(f'Not included in the dictionary: {not_included_count}')
        return not_included_features
    else:
        return None

def select_features_multicollinearity_graph_target_corr(
    X: pd.DataFrame,
    y: pd.DataFrame, # DataFrame for one or more target variables
    predictor_correlation_threshold: float = 0.8,
    min_target_correlation_for_group_representative: float = 0.01 # minimum correlation with target to be considered
) -> list[str]:
    """
    Selects features by building a graph of highly correlated predictor features and
    then choosing one feature from each connected component (group of highly
    correlated features). The choice is based on which feature within the component
    has the highest absolute correlation with any of the target variables.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")
    if not isinstance(y, pd.DataFrame):
        raise ValueError("y must be a pandas DataFrame.")
    if not 0 <= predictor_correlation_threshold <= 1:
        raise ValueError("predictor_correlation_threshold must be between 0 and 1.")
    if not 0 <= min_target_correlation_for_group_representative <= 1:
        raise ValueError("min_target_correlation_for_group_representative must be between 0 and 1.")


    # Ensure X contains only numeric columns for correlation
    X_numeric = X.select_dtypes(include=np.number)
    if X_numeric.shape[1] < X.shape[1]:
        print("Попередження: Нечислові колонки були виключені з X перед розрахунком кореляції.")

    # Ensure y contains only numeric columns
    y_numeric = y.select_dtypes(include=np.number)
    if y_numeric.shape[1] < y.shape[1]:
        print("Попередження: Нечислові колонки були виключені з y перед розрахунком кореляції.")
    if y_numeric.empty:
        raise ValueError("Цільовий DataFrame y не містить числових колонок.")


    # 1. Calculate correlation matrix of predictor features
    predictor_corr_matrix = X_numeric.corr().abs()
    features = X_numeric.columns.tolist()

    # 2. Build graph of highly correlated features
    graph = nx.Graph()
    graph.add_nodes_from(features)

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feature1 = features[i]
            feature2 = features[j]
            if predictor_corr_matrix.loc[feature1, feature2] >= predictor_correlation_threshold:
                graph.add_edge(feature1, feature2)

    # 3. Find connected components (groups of highly inter-correlated features)
    connected_components = list(nx.connected_components(graph))

    selected_features_final = []

    # 4. For each component, select the feature most correlated with the target(s)
    for component in connected_components:
        component_features = list(component)
        if not component_features:
            continue

        if len(component_features) == 1:
            # This feature is not highly correlated with any other feature, so keep it
            selected_features_final.append(component_features[0])
            continue

        # For components with multiple highly correlated features,
        # choose the one with the highest absolute correlation with any target variable.
        best_feature_in_component = None
        max_abs_corr_with_any_target = -1.0

        for feature_in_component in component_features:
            current_feature_max_corr_with_targets = 0.0
            # Calculate correlation with each target and find the maximum absolute correlation
            for target_col_name in y_numeric.columns:
                # Ensure Series are aligned and handle potential NaNs in correlation calculation
                temp_corr_series = pd.concat([X_numeric[feature_in_component], y_numeric[target_col_name]], axis=1).corr()
                # Access correlation value, could be NaN if one series is all NaN or constant
                correlation_value = temp_corr_series.iloc[0, 1]
                if pd.notna(correlation_value):
                    current_feature_max_corr_with_targets = max(current_feature_max_corr_with_targets, abs(correlation_value))

            if current_feature_max_corr_with_targets >= min_target_correlation_for_group_representative:
                if current_feature_max_corr_with_targets > max_abs_corr_with_any_target:
                    max_abs_corr_with_any_target = current_feature_max_corr_with_targets
                    best_feature_in_component = feature_in_component
                elif current_feature_max_corr_with_targets == max_abs_corr_with_any_target:
                    # Tie-breaking: prefer shorter name, then alphabetically for consistency
                    if best_feature_in_component is None or \
                       len(feature_in_component) < len(best_feature_in_component) or \
                       (len(feature_in_component) == len(best_feature_in_component) and feature_in_component < best_feature_in_component):
                        best_feature_in_component = feature_in_component

        if best_feature_in_component: # If a suitable representative was found
            selected_features_final.append(best_feature_in_component)
        elif component_features: # Fallback if no feature met min_target_correlation (should be rare if min_target_correlation_for_group_representative is 0)
                                 # Or if all correlations were NaN. Pick the first one for stability.
            component_features.sort() # Sort for stable selection
            selected_features_final.append(component_features[0])


    return sorted(list(set(selected_features_final)))