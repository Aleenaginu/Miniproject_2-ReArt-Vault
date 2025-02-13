from django.db import models
from django.contrib.auth.models import User
from adminclick.models import *
from django.core.validators import RegexValidator
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
    
class ArtistAddress(models.Model):
    artist = models.OneToOneField(Artist, on_delete=models.CASCADE, related_name='address_details')
    address = models.TextField()
    pincode = models.CharField(max_length=6, validators=[
        RegexValidator(
            regex='^[0-9]{6}$',
            message='Pincode must be 6 digits',
            code='invalid_pincode'
        )
    ])
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.artist.user.username}'s Address"
    
class Adminclick(models.Model):
    user = models.OneToOneField(User, on_delete = models.CASCADE)
    phone = models.IntegerField(default=99999999)
    profile_pic = models.ImageField(upload_to= 'picture/admin/admin.jpg', null=True)

    def __str__(self):
        return self.user.username