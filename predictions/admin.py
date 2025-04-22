from django.contrib import admin

from predictions.models import TransactionCategory, PredictionAlgorithm


@admin.register(TransactionCategory)
class TransactionCategoryAdmin(admin.ModelAdmin):
    list_display = ['id', 'name']
    search_fields = ['name',]
    ordering = ['name',]

@admin.register(PredictionAlgorithm)
class PredictionAlgorithmAdmin(admin.ModelAdmin):
    list_display = ['id', 'name']
    search_fields = ['name',]
    ordering = ['name',]