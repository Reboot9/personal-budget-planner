DROP_SECONDARY_FEATURES: bool = True

LAGS: list[int] = [1, 2, 3, 7, 14]
WINDOWS: list[int] = [7, 14, 30, 60]
WINDOW_RATIO: int = 30
WINDOW_NON_TARGET: int = 7

# Константи для відбору ознак
N_TOP_CORRELATIONS: int = 5
ALPHA_SIGNIFICANCE: float = 0.05
MIN_NON_ZERO_FRACTION: float = 0.01
MIN_VARIANCE_NON_ZERO: float = 1e-4
IMPORTANCE_SCORE_THRESHOLD: float = 1e-3

# calendar data
CALENDAR_COLUMNS_TO_DROP_POST_ANALYSIS: list[str] = ['day_of_month', 'week_of_month']
PREDICTOR_CORRELATION_THRESHOLD: float = 0.9
MIN_TARGET_CORRELATION_FOR_GROUP_REPRESENTATIVE: float = 0.1

# RandomForest params
RF_N_ESTIMATORS: int = 150
RF_MAX_DEPTH: int = 20
RF_MIN_SAMPLES_LEAF: int = 5
RF_N_REPEATS_PERMUTATION: int = 5

# ml settings
TEST_SIZE_DAYS=90

WINDOW_SIZE_MAIN = 14
FORECAST_HORIZON_MAIN = 1
TRAIN_RATIO_MAIN = 0.7
VAL_RATIO_MAIN = 0.15
N_RECURSIVE_TEST_STEPS_MAIN = 7
FINAL_TRAIN_EPOCHS_MAIN = 10
TUNER_MAX_TRIALS_MAIN = 5
TUNER_EPOCHS_MAIN = 7
CHECKPOINT_FILEPATH_MAIN = './final_trained_model_oop.keras'

SKIPPED_PERIODS: int = 3
TEMPORAL_COLUMNS_CONFIG: dict[str, int] = {
    'month': 12,
    'week_of_month': 5,
    'day_of_week': 7
}

RAW_TEMPORAL_COLUMNS_TO_DROP: list[str] = ['year', 'month', 'week_of_month', 'day_of_week', 'month_period']