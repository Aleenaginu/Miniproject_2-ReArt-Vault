from django.db import models
from django.contrib.auth.models import User
from donors.models import *
from donors.models import InterestRequest
# Create your models here.
class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    donation=models.ForeignKey('donors.Donation', on_delete=models.CASCADE, null=True,blank=True)

    def __str__(self):
        return f'Notification for {self.user.username}'
    
class Interest(models.Model):
    artist=models.ForeignKey(Artist,on_delete=models.CASCADE)
    donation=models.ForeignKey(Donation, on_delete=models.CASCADE)
    expressed_at=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f' {self.artist.user.username} - {self.donation}'

class InterestNotification(models.Model):
    artist = models.ForeignKey(Artist, on_delete=models.CASCADE)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    interest_request=models.ForeignKey(InterestRequest, on_delete=models.CASCADE, null=True,blank=True)

    def __str__(self):
        return f'Notification for {self.artist.user.username}'


class Payment(models.Model):
    artist=models.ForeignKey(Artist, on_delete=models.CASCADE)
    amount=models.DecimalField(max_digits=10, decimal_places=2)
    payment_id=models.CharField(max_length=100,blank=True,null=True)
    order_id=models.CharField(max_length=100,blank=True,null=True)
    status=models.CharField(max_length=20,default='pending')
    created_at=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Payment {self.id} - {self.artist.user.username}'

