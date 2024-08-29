from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Customers(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone = models.IntegerField(default=9999999999)
    profile_pic = models.ImageField(upload_to='picture/customer', null =True, default='picture/customer/hi.jpg')
    
    
    def __str__(self):
        return self.user.username
    
