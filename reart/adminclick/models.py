from django.db import models

# Create your models here.
class MediumOfWaste(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    rate=models.DecimalField(max_digits=10,decimal_places=2,default=0.00)

    def __str__(self):
        return self.name

# class Category(models.Model):
#     name=models.CharField(max_length=255, unique=True)

#     def __str__(self):
#         return self.name
    