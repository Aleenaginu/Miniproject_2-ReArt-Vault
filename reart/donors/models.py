from django.db import models
from accounts.models import *
from adminclick.models import *

# Create your models here.
class Donation(models.Model):
    donor = models.ForeignKey(Donors, on_delete=models.CASCADE)
    medium_of_waste = models.ForeignKey(MediumOfWaste, on_delete=models.CASCADE)
    quantity = models.DecimalField(max_digits=10, decimal_places=2)
    location = models.CharField(max_length=255)
    date_donated = models.DateTimeField(auto_now_add=True)
    image=models.ImageField(upload_to='picture/donor',null=True)
    STATUS_CHOICES = [
        ('pending','Pending'),
        ('accepted','Accepted'),
        ('rejected','Rejected'),
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')

    def __str__(self):
        return f'{self.donor.user.username} - {self.medium_of_waste.name}'