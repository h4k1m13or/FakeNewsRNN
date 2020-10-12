from django.db import models

# Create your models here.
class dataset(models.Model):
    author = models.CharField(max_length=50)
    title = models.CharField(max_length=500)
    text = models.CharField(max_length=20000)
    label = models.BooleanField()