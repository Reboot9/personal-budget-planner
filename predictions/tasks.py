from celery import shared_task

from predictions.services.main_model.model_prediction import ModelTrainer


def convert_to_builtin_type(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(i) for i in obj]
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    return obj

# @shared_task
# def train_and_predict_task(user_id, transaction_ids, n_days):
#     from predictions.models import Transaction
#     from predictions.services.budget_prediction import BudgetPredictionModel
#
#     transactions = Transaction.objects.filter(id__in=transaction_ids)
#     model = BudgetPredictionModel(user_id=user_id)
#     model.train(transactions)
#     predictions = model.predict(n_days=n_days)
#
#     # Convert predictions to Python float to ensure JSON serializability
#     # Assuming predictions is a numpy array or list with numpy float32 types
#     return convert_to_builtin_type(predictions)


@shared_task
def train_and_predict_task(user_id, transaction_ids, n_days):
    import json
    from predictions.models import Transaction
    import pathlib
    from predictions.services.main_model.feature_engineering import FeatureEngineer
    from predictions.services.main_model.model_prediction import Predictor
    from personal_budget_planner.constants import CHECKPOINT_FILEPATH_MAIN
    from predictions.services.main_model.model_prediction import DataPreprocessor

    import pandas as pd

    file_path_data_folder = pathlib.Path(__file__).parents[3].resolve() / 'data/intermediate'
    window_size = 14
    forecast_horizon = 1
    n_recursive_test_steps = n_days
    train_ratio = 0.8
    val_ratio = 0.1

    transactions = Transaction.objects.filter(id__in=transaction_ids).order_by('date')
    df = pd.DataFrame.from_records(transactions.values())

    if df.empty:
        return {'error': 'No transactions found.'}

    df = df[['user_id', 'date', 'category', 'amount']]
    df['date'] = pd.to_datetime(df['date'])

    df_wide = df.pivot_table(index='date', columns='category', values='amount', aggfunc='sum').fillna(0)
    df_wide.reset_index(inplace=True)
    df_wide['user_id'] = user_id
    df_wide = df_wide.sort_values('date')

    full_range = pd.date_range(df_wide['date'].min(), df_wide['date'].max(), freq='D')
    df_wide = df_wide.set_index('date').reindex(full_range).fillna(0).rename_axis('date').reset_index()
    df_wide['user_id'] = user_id

    history_days = 60
    df_wide = df_wide.sort_values('date').iloc[-history_days:]

    fe = FeatureEngineer()
    target_cols = df_wide.columns.drop(['date', 'user_id']).tolist()
    column_categories = {'regular': target_cols}
    features_df, _ = fe.engineer_features(df_wide, column_categories=column_categories, target_cols=target_cols)

    features_df = features_df.sort_values('date').iloc[-14:]

    # keep only selected features
    intermediate_path = pathlib.Path(__file__).parents[3] / 'data' / 'intermediate'
    with open(intermediate_path / 'selected_features.json') as f:
        selected_features = json.load(f)
    features_df = features_df[selected_features]
    X_raw = features_df.to_numpy()
    y_raw = df_wide[target_cols].iloc[-len(features_df):].to_numpy()

    data_preprocessor = DataPreprocessor(
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        n_recursive_test_steps=n_recursive_test_steps
    )
    data_preprocessor.prepare_all_data(X_raw, y_raw, target_cols)


    required_len = window_size + (n_recursive_test_steps - 1) + forecast_horizon
    if len(X_raw) < required_len:
        return {'error': 'Not enough data for recursive prediction.'}
    model_checkpoint = file_path_data_folder / f'model/{CHECKPOINT_FILEPATH_MAIN}'
    tuner_directory = file_path_data_folder / f'keras_tuner'

    model_trainer = ModelTrainer(data_preprocessor=data_preprocessor,
                                 checkpoint_filepath=str(model_checkpoint.resolve()),
                                 tuner_directory_path=str(tuner_directory.resolve()))

    predictor = Predictor(model=model_trainer.trained_model, data_preprocessor=data_preprocessor)
    rec_input_start = len(X_raw) - required_len
    init_window = X_raw[rec_input_start: rec_input_start + window_size]

    n_exo = data_preprocessor.n_input_features - data_preprocessor.n_target_categories
    true_future_exo_raw = None
    if n_exo > 0:
        exo_start = rec_input_start + window_size
        exo_end = exo_start + (n_recursive_test_steps - 1)
        true_future_exo_raw = X_raw[exo_start:exo_end, data_preprocessor.n_target_categories:]


    recursive_preds = predictor.predict_recursively(
        init_window,
        num_recursive_steps=n_recursive_test_steps,
        true_future_exogenous_raw=true_future_exo_raw
    )

    reshaped = recursive_preds.reshape(n_recursive_test_steps, forecast_horizon, -1)
    json_predictions = {
        f'Day {i + 1}': dict(zip(target_cols, reshaped[i, 0]))
        for i in range(n_recursive_test_steps)
    }

    return convert_to_builtin_type(json_predictions)
