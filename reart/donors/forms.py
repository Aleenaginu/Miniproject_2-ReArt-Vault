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
    images=forms.FileField(widget=forms.ClearableFileInput(),required=False)
    class Meta:
        model = Donation
        fields = ['medium_of_waste','quantity','location','images']

    def save(self,commit=True):
        donation=super().save(commit=False)
        if commit:
            donation.save()
        if self.cleaned_data['images']:
            for image in self.cleaned_data['images']:
                DonationImage.objects.create(donation=donation,image=image)
        return donation

