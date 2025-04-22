import pickle

from django.contrib.auth import get_user_model
from django.core.files.storage import FileSystemStorage
from django.db import models

model_storage = FileSystemStorage(location='models/')
User = get_user_model()

class TransactionCategory(models.Model):
    name = models.CharField(max_length=255)

    class Meta:
        verbose_name = "Transaction Category"
        verbose_name_plural = "Transaction Categories"
        ordering = ['name']

    def __str__(self):
        return self.name

class Transaction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField()
    category = models.ForeignKey(TransactionCategory, on_delete=models.SET_NULL, null=True, blank=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.date} - {self.category.name if self.category else 'No Category'} - {self.amount}"

    class Meta:
        indexes = [
            models.Index(fields=['user', 'date']),
        ]

class PredictionAlgorithm(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True, max_length=4096)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class PredictionModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    model_file = models.FileField(upload_to='models/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Model for {self.user.username}"

    def save_model(self, model):
        with open(self.model_file.path, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self):
        with open(self.model_file.path, 'rb') as f:
            return pickle.load(f)


class UserFileUpload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='transaction_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"File uploaded by {self.user.email} on {self.uploaded_at}"