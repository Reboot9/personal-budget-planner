import os

import pandas as pd
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.sites import requests
from django.core.exceptions import ValidationError
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic import FormView, ListView
from django.views.generic import TemplateView

from .forms import UserFileUploadForm, PredictionForm
from .models import UserFileUpload, TransactionCategory, Transaction
from .services.budget_prediction import BudgetPredictionModel


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

    def form_valid(self, form):
        categories = form.cleaned_data['category']
        date_from = form.cleaned_data.get('date_from')
        date_to = form.cleaned_data.get('date_to')

        transactions = Transaction.objects.filter(category__in=categories, user=self.request.user)

        if date_from:
            transactions = transactions.filter(date__gte=date_from)
        if date_to:
            transactions = transactions.filter(date__lte=date_to)

        model = BudgetPredictionModel(user_id=self.request.user.id)
        # try:
        print("pred")
        model.train(transactions)
        predictions = model.predict(n_days=7)

        return render(self.request, 'prediction_result.html', {'prediction': predictions, 'form': form})
        # except Exception as e:
        #     print(e)
        #     form.add_error(None, f"Error while making prediction {str(e)}")
        #     return self.form_invalid(form)

    def form_invalid(self, form):
        return self.render_to_response({'form': form})

class PredictionResultView(TemplateView):
    template_name = 'prediction_result.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context