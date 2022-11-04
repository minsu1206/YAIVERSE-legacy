from django.db import models

# Create your models here.

class InferenceData(models.Model):
    col = models.CharField(max_length=10, unique=True)
    style = models.CharField(max_length=10)
    user_code = models.CharField(max_length=10)
    timestamp = models.DateTimeField(auto_now_add=True)
    fail = models.BooleanField(default=False)

    def __str__(self):
        return self.col