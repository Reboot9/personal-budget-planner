from django.core.management import BaseCommand

from predictions.services.main_model.data_preparation import main as run_data_preparation_main
from predictions.services.main_model.feature_engineering import main as run_feature_engineering_main
from predictions.services.main_model.feature_selection import main as run_feature_selection_main
from predictions.services.main_model.data_splitter import main as run_data_splitter_main
from predictions.services.main_model.model_prediction import main as run_model_prediction_main


class Command(BaseCommand):
    def handle(self, *args, **options):
        run_data_preparation_main()
        run_feature_engineering_main()
        run_feature_selection_main()
        run_data_splitter_main()
        run_model_prediction_main()
        print("Done")
