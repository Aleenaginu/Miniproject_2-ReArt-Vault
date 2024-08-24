from django.db import models
from django.contrib.auth.models import User
from adminclick.models import *
# Create your models here.
class Donors(models.Model):
    user = models.OneToOneField(User, on_delete = models.CASCADE)
    phone = models.IntegerField(default=99999999)
    profile_pic = models.ImageField(upload_to= 'picture/donor', null=True,default='picture/donor/hi.jpg')

    def __str__(self):
        return self.user.username
    
class Artist(models.Model):
    user = models.OneToOneField(User, on_delete = models.CASCADE)
    phone = models.IntegerField(default=99999999)
    profile_pic = models.ImageField(upload_to= 'picture/artist', null=True,default='picture/artist/hi.jpg')
    is_approved = models.BooleanField(default=False)
    mediums = models.ManyToManyField(MediumOfWaste, blank=True) 
    certificate=models.FileField(upload_to='certificates/',null= True, blank=True)

    def __str__(self):
        return self.user.username
    
    
class Adminclick(models.Model):
    user = models.OneToOneField(User, on_delete = models.CASCADE)
    phone = models.IntegerField(default=99999999)
    profile_pic = models.ImageField(upload_to= 'picture/admin/admin.jpg', null=True)

    def __str__(self):
        return self.user.username