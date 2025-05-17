from django import forms
from .models import UserFileUpload, TransactionCategory, Transaction


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


class TransactionForm(forms.ModelForm):
    class Meta:
        model = Transaction
        fields = ['date', 'category', 'amount']
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.required = True
        self.initial['data_source'] = 2

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.data_source = 2
        if commit:
            instance.save()
        return instance

class TransactionFilterForm(forms.Form):
    date_from = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    date_to = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    category = forms.ModelChoiceField(queryset=TransactionCategory.objects.all(), required=False, empty_label="All categories")
    amount_min = forms.DecimalField(required=False, decimal_places=2, max_digits=10)
    amount_max = forms.DecimalField(required=False, decimal_places=2, max_digits=10)