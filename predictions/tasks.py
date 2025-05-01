from celery import shared_task

def convert_to_builtin_type(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(i) for i in obj]
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    return obj

@shared_task
def train_and_predict_task(user_id, transaction_ids, n_days):
    from predictions.models import Transaction
    from predictions.services.budget_prediction import BudgetPredictionModel

    transactions = Transaction.objects.filter(id__in=transaction_ids)
    model = BudgetPredictionModel(user_id=user_id)
    model.train(transactions)
    predictions = model.predict(n_days=n_days)
    predictions = convert_to_builtin_type(predictions)
    print(predictions)
    # Convert predictions to Python float to ensure JSON serializability
    # Assuming predictions is a numpy array or list with numpy float32 types
    return convert_to_builtin_type(predictions)
