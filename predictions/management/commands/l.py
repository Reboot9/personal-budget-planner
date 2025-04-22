from django.core.management import BaseCommand

from predictions.models import Transaction
from predictions.services.budget_prediction import BudgetPredictionModel
import pandas as pd


class Command(BaseCommand):
    def handle(self, *args, **options):
        transactions = Transaction.objects.filter(user_id=2)
        model = BudgetPredictionModel(user_id=2)
        model.train(transactions)
        predictions = model.predict(n_days=5)
        for start, pred in enumerate(predictions, start=1):
            print(f"Day {start}: {pred}")

        # categories = [transaction.category.name if transaction.category else 'Other' for transaction in transactions]
        #
        # categories_series = pd.Series(categories)
        # category_counts = categories_series.value_counts()
        #
        # print(category_counts)
