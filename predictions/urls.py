from django.urls import path
from predictions import views

urlpatterns = [
    path('task-status/<str:task_id>/', views.prediction_task_status, name='prediction_task_status'),
    path('upload/', views.FileUploadView.as_view(), name='file-upload'),
    path('files/', views.FileListView.as_view(), name='file-list'),
    path('predict/', views.PredictionFormView.as_view(), name='prediction_form'),
    path('result/<int:pk>/', views.PredictionResultView.as_view(), name='view_saved_prediction'),
    path('result/', views.PredictionResultView.as_view(), name='prediction_result'),

    path('save_prediction/', views.SavePredictionView.as_view(), name='save_prediction'),
    path('delete/<int:pk>/', views.DeletePredictionView.as_view(), name='delete_prediction'),

    path('transactions/add/', views.TransactionCreateView.as_view(), name='transaction_add'),
    path('transactions/delete/<int:pk>/', views.TransactionDeleteView.as_view(), name='transaction_delete'),

]
