# Generated by Django 4.1.3 on 2022-11-03 12:19

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='InferenceData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('col', models.CharField(max_length=10, unique=True)),
                ('style', models.CharField(max_length=10)),
                ('user_code', models.CharField(max_length=10)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]