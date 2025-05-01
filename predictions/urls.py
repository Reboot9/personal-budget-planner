from django.urls import path
from .views import FileUploadView, FileListView, PredictionFormView, PredictionResultView, prediction_task_status

urlpatterns = [
    path('task-status/<str:task_id>/', prediction_task_status, name='prediction_task_status'),
    path('upload/', FileUploadView.as_view(), name='file-upload'),
    path('files/', FileListView.as_view(), name='file-list'),
    path('predict/', PredictionFormView.as_view(), name='prediction_form'),
    path('result/', PredictionResultView.as_view(), name='prediction_result'),
]
