import json
import os

import pandas as pd
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.sites import requests
from django.core.exceptions import ValidationError, PermissionDenied
from django.http import JsonResponse, HttpResponseRedirect, HttpResponseBadRequest
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse_lazy, reverse
from django.views import View
from django.views.generic import FormView, ListView, DeleteView
from django.views.generic import TemplateView

from .forms import UserFileUploadForm, PredictionForm
from .models import UserFileUpload, TransactionCategory, Transaction, UserPrediction, PredictionDay, PredictionCategory
from .services.budget_prediction import BudgetPredictionModel
from predictions.tasks import train_and_predict_task
from celery.result import AsyncResult
from django.http import JsonResponse
from personal_budget_planner.celery import app

def prediction_task_status(request, task_id):
    result = AsyncResult(task_id, app=app)
    if result.ready():
        prediction = result.result
        # Replace None or missing values with a default value
        cleaned_prediction = [
            {key: (value if value is not None else 0) for key, value in day.items()}
            for day in prediction
        ]
        cleaned_prediction = [
            {'Taxi': 4, 'Coffee': 16.475573, 'Learning': 16.209991, 'Market': 15.977781, 'Phone': 15.803815,
             'Restaurant': 16.35557, },
            {'Coffee': 16.475573, 'Learning': 24, 'Market': 15.977781, 'Phone': 15.803815,},
            {'Coffee': 16.475573, 'Learning': 20, 'Phone': 15.803815,
             'Restaurant': 16.35557, 'Taxi': 16.80649},
            {'Coffee': 16.475573, 'Learning': 16.209991, 'Market': 15.977781, 'Phone': 15.803815,
             'Restaurant': 16.35557, 'Taxi': 16.80649}
        ]
        request.session['prediction_data'] = cleaned_prediction
        return JsonResponse({'status': 'done', 'result': cleaned_prediction})
    return JsonResponse({'status': 'pending'})


class HomeView(LoginRequiredMixin, TemplateView):
    template_name = "home.html"
    login_url = "login"
    redirect_field_name = "next"


class FileUploadView(LoginRequiredMixin, FormView):
    template_name = 'file_upload.html'
    form_class = UserFileUploadForm
    success_url = reverse_lazy('file-list')

    def form_valid(self, form):
        file_instance = form.save(commit=False)
        file_instance.user = self.request.user
        file_path = file_instance.file.path
        file_extension = file_path.split('.')[-1].lower()

        if file_extension not in ['csv', 'xls', 'xlsx']:
            form.add_error('file', 'Unsupported file type. Please upload a CSV or XLSX file.')
            return self.form_invalid(form)

        file_instance.save()
        status = self.process_uploaded_file(file_instance)
        if status == 0:
            form.add_error('file', 'There was an error processing the file.')
            return self.form_invalid(form)

        return super().form_valid(form)

    def process_uploaded_file(self, file_instance):
        file_path = file_instance.file.path
        file_extension = file_path.split('.')[-1].lower()

        if file_extension == 'csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file_path)
        else:
            return 0

        default_category = TransactionCategory.objects.filter(name='Other').first()

        if not default_category:
            raise ValidationError("'Other' category does not exist in the database.")
        required_columns = ['category', 'date', 'amount']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValidationError(f"Missing required columns: {', '.join(missing_columns)}")
        for index, row in df.iterrows():
            category_name = row.get('category', 'Other')
            category = TransactionCategory.objects.filter(name=category_name).first()
            if not category:
                category = default_category

            try:
                transaction_date = pd.to_datetime(row['date'], errors='raise').date()
            except Exception:
                raise ValidationError(f"Invalid date format at row {index + 1}.")
            try:
                amount = float(row['amount'])
            except ValueError:
                raise ValidationError(f"Invalid amount format at row {index + 1}.")

            transaction = Transaction(
                user=file_instance.user,
                date=transaction_date,
                category=category,
                amount=amount
            )
            transaction.save()

        return 1


class FileListView(LoginRequiredMixin, ListView):
    model = UserFileUpload
    template_name = 'file_list.html'
    context_object_name = 'files'

    def get_queryset(self):
        return UserFileUpload.objects.filter(user=self.request.user).order_by('-uploaded_at')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        for file in context['files']:
            file.filename = os.path.basename(file.file.name)
        return context


class PredictionFormView(LoginRequiredMixin, FormView):
    template_name = 'prediction_form.html'
    form_class = PredictionForm

    def get(self, request, *args, **kwargs):
        form = self.form_class()
        saved_predictions = UserPrediction.objects.filter(user=request.user) \
            .prefetch_related('days') \
            .order_by('-created_at')
        return self.render_to_response({
            'form': form,
            'saved_predictions': saved_predictions
        })

    def form_valid(self, form):
        categories = form.cleaned_data['category']
        date_from = form.cleaned_data.get('date_from')
        date_to = form.cleaned_data.get('date_to')

        transactions = Transaction.objects.filter(category__in=categories, user=self.request.user)

        if date_from:
            transactions = transactions.filter(date__gte=date_from)
        if date_to:
            transactions = transactions.filter(date__lte=date_to)

        transaction_ids = list(transactions.values_list('id', flat=True))
        task = train_and_predict_task.delay(self.request.user.id, transaction_ids, 7)

        # Store task ID in session for later checking in JS
        self.request.session['task_id'] = task.id

        return redirect('prediction_form')

        # return render(self.request, 'prediction_result.html', {'prediction': predictions, 'form': form})

    def form_invalid(self, form):
        saved_predictions = UserPrediction.objects.filter(user=self.request.user).prefetch_related('days').order_by('-created_at')
        return self.render_to_response({
            'form': form,
            'saved_predictions': saved_predictions
        })


class PredictionResultView(TemplateView):
    template_name = 'prediction_result.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        prediction_id = self.kwargs.get('pk')
        context['readonly'] = bool(self.kwargs.get('pk'))

        if prediction_id:
            prediction = get_object_or_404(UserPrediction, pk=prediction_id, user=self.request.user)
            days_data = []
            for day in prediction.days.order_by('order'):
                day_dict = {cat.category.name: float(cat.amount) for cat in day.categories.all()}
                days_data.append(day_dict)
            context['prediction'] = json.dumps(days_data)
        else:
            days_data = self.request.session.get('prediction_data', [])

        context['prediction'] = json.dumps(days_data)
        context['prediction_length'] = len(days_data)

        return context

class SavePredictionView(LoginRequiredMixin, View):
    def post(self, request):
        data_raw = request.POST.get('data', '')
        if not data_raw:
            return HttpResponseBadRequest("No prediction data provided.")

        try:
            data = json.loads(data_raw)
        except json.JSONDecodeError:
            return HttpResponseBadRequest("Invalid JSON in prediction data.")

        name = request.POST.get('name', '').strip()

        user_prediction = UserPrediction.objects.create(user=request.user, name=name)
        days = [PredictionDay(prediction=user_prediction, order=i + 1) for i in range(len(data))]
        PredictionDay.objects.bulk_create(days)

        days = list(user_prediction.days.all())
        categories = []
        for i, day_data in enumerate(data):
            for category_name, amount in day_data.items():
                category = TransactionCategory.objects.filter(name=category_name).first()
                if category:
                    categories.append(PredictionCategory(
                        prediction_day=days[i],
                        category=category,
                        amount=amount
                    ))
        PredictionCategory.objects.bulk_create(categories)

        # Delete prediction data from session after saving
        request.session.pop('prediction_data', None)

        return redirect('home')

class DeletePredictionView(LoginRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        prediction = get_object_or_404(UserPrediction, pk=kwargs['pk'], user_id=self.request.user.id)

        if prediction.user != request.user:
            raise PermissionDenied("You do not have permission to delete this prediction.")

        prediction.delete()
        return redirect('prediction_form')