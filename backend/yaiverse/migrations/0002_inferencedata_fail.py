# Generated by Django 4.1.3 on 2022-11-03 13:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('yaiverse', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='inferencedata',
            name='fail',
            field=models.BooleanField(default=False),
        ),
    ]