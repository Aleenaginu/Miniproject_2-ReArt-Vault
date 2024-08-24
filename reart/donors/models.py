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
    STATUS_CHOICES = [
        ('pending','Pending'),
        ('accepted','Accepted'),
        ('rejected','Rejected'),
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')

    def __str__(self):
        return f'{self.donor.user.username} - {self.medium_of_waste.name}'
    
class DonationImage(models.Model):
    donation = models.ForeignKey(Donation, on_delete=models.CASCADE, related_name='images')
    image=models.ImageField(upload_to='picture/donor')

    def __str__(self):
        return f'Image for {self.donation}'
    

class InterestRequest(models.Model):
    STATUS_CHOICES = [
        ('pending','Pending'),
        ('accepted','Accepted'),
        ('rejected','Rejected'),
    ]
    
    artist=models.ForeignKey(Artist, on_delete=models.CASCADE)
    donor=models.ForeignKey(Donors, on_delete=models.CASCADE)
    donation=models.ForeignKey(Donation, on_delete=models.CASCADE)
    status=models.CharField(max_length=10,choices=STATUS_CHOICES, default='pending')
    expressed_at=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.artist.user.username} - {self.donation}'
    
class DonorNotification(models.Model):
    donor=models.ForeignKey(Donors, on_delete=models.CASCADE)
    message=models.TextField()
    created_at=models.DateTimeField(auto_now_add=True)
    is_read=models.BooleanField(default=False)
    interest_request=models.ForeignKey(InterestRequest, on_delete=models.CASCADE,null=True, blank=True)

    def _str_(self):
        return f'Notification for {self.donor.user.username}'
