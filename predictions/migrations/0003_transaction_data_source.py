# Generated by Django 5.1.7 on 2025-05-17 16:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictions', '0002_predictionday_alter_transactioncategory_options_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='transaction',
            name='data_source',
            field=models.IntegerField(default=1),
        ),
    ]
