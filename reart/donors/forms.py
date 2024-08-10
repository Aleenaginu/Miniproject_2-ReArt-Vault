from django import forms 
from accounts.models import Donors
from django.contrib.auth.models import User
from . models import *
class UserUpdateForm(forms.ModelForm):
    class Meta:
        model=User
        fields=['email']
class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model=Donors
        fields=['phone', 'profile_pic']

class DonationForm(forms.ModelForm):
    class Meta:
        model = Donation
        fields = ['medium_of_waste','quantity','location','image']

