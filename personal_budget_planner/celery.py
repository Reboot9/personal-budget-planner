import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'personal_budget_planner.settings.local')

app = Celery('personal_budget_planner')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
