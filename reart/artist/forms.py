from django import forms 
from accounts.models import Artist
from django.contrib.auth.models import User
class UserUpdateFormArtist(forms.ModelForm):
    class Meta:
        model=User
        fields=['email']
class ProfileUpdateFormArtist(forms.ModelForm):
    class Meta:
        model=Artist
        fields=['phone', 'profile_pic']
