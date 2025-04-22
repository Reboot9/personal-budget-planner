from django.urls import path
from .views import FileUploadView, FileListView, PredictionFormView, PredictionResultView

urlpatterns = [
    path('upload/', FileUploadView.as_view(), name='file-upload'),
    path('files/', FileListView.as_view(), name='file-list'),
    path('predict/', PredictionFormView.as_view(), name='prediction_form'),
    path('result/', PredictionResultView.as_view(), name='prediction_result'),
]
