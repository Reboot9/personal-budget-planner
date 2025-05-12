from datetime import timedelta

from django.core.management import BaseCommand

from predictions.models import Transaction
from predictions.services.budget_prediction import BudgetPredictionModel
import pandas as pd
import numpy as np

from predictions.services.main_model.model_prediction import DataPreprocessor


class Command(BaseCommand):
    def handle(self, *args, **options):
        transactions = Transaction.objects.filter(user_id=2).order_by('date')
        df = pd.DataFrame(list(transactions.values('date', 'category_id', 'amount')))

        df['category_id'] = df['category_id'].astype('category').cat.codes
        df['date'] = pd.to_datetime(df['date'])

        df.set_index('date', inplace=True)
        target_cols = ['amount']
        df['amount'] = df['amount'].astype(float)

        X = df[['category_id']].values  # Features array
        y = df[target_cols].values  # Target array

        window_size = 30  # For example, 30 days for window
        forecast_horizon = 7  # For example, forecasting next 7 days
        train_ratio = 0.7
        val_ratio = 0.15
        n_recursive_test_steps = 7

        data_preprocessor = DataPreprocessor(
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            n_recursive_test_steps=n_recursive_test_steps
        )

        # Prepare the data
        data_preprocessor.prepare_all_data(X, y, target_cols)
