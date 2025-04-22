from django import forms
from .models import UserFileUpload, TransactionCategory


class UserFileUploadForm(forms.ModelForm):
    class Meta:
        model = UserFileUpload
        fields = ['file']


class PredictionForm(forms.Form):
    category = forms.ModelMultipleChoiceField(
        queryset=TransactionCategory.objects.all(),
        required=False,
        widget=forms.SelectMultiple(attrs={
            'class': 'form-control selectpicker',
            'data-live-search': 'true',
            'multiple': 'multiple'
        })
    )
    date_from = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    date_to = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.data.get('category'):
            self.fields['category'].initial = [category.id for category in TransactionCategory.objects.all()]