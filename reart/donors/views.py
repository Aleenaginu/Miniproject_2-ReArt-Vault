from django.shortcuts import render,redirect
from . forms import *

from adminclick.models import *
from django.http import HttpResponseForbidden
from django.contrib import messages
from .models import *
from django.contrib.auth.decorators import login_required

from .models import Donation

# Create your views here.
@login_required
def donor_dashboard(request):
       if request.user.is_authenticated and request.user.donors:
         donor= request.user.donors
         return render(request,'Donors/dashboard.html',{'donor':donor})
       
def donor_update(request):
    if request.method=='POST':
        user_form = UserUpdateForm(request.POST,instance=request.user)
        profile_form=ProfileUpdateForm(request.POST, request.FILES,instance=request.user.donors)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            return redirect ('donor_dashboard')
    else:
        if request.user.is_authenticated and request.user.donors:
         donor= request.user.donors
        user_form = UserUpdateForm(instance=request.user)
        profile_form=ProfileUpdateForm(instance=request.user.donors)
    context = {
     'user_form': user_form,
     'profile_form':profile_form,
     'donor':donor,
     
    }
    return render(request,'Donors/updateprofile.html', context)

def donate_waste(request):
    
    if not hasattr(request.user, 'donors'):
        return HttpResponseForbidden("You are not allowed to access this page.")
    
    donor=request.user.donors
    context={
        'mediums': MediumOfWaste.objects.all(),
        'donor':donor
        
    }

    if request.method == 'POST':
        medium_id = request.POST.get('medium_of_waste')
        quantity = request.POST.get('quantity')
        location = request.POST.get('location')
        image = request.FILES.get('image')

        try:
            medium_of_waste = MediumOfWaste.objects.get(id=medium_id)
        except MediumOfWaste.DoesNotExist:
            messages.error(request, 'Invalid medium of waste selected.')
            return render(request, 'Donors/donate_waste.html', {'mediums': MediumOfWaste.objects.all()})

        Donation.objects.create(
            donor=request.user.donors,
            medium_of_waste=medium_of_waste,
            quantity=quantity,
            location=location,
            image=image,
            status='pending'
        )

        messages.success(request, 'Donation recorded successfully.')
        return redirect('donor_dashboard')  

    return render(request, 'Donors/donate_waste.html', context)



def view_donations(request):
     donations = Donation.objects.filter(donor=request.user.donors)
     donor=request.user.donors
     context={
         'donations': donations , 
         'donor':donor
     }
     return render(request, 'Donors/manage_donation.html',context )

def view_rates(request):
  
     donor=request.user.donors
     mediums=MediumOfWaste.objects.all()
     context={
         'mediums': mediums , 
         'donor':donor
     }
     return render(request, 'Donors/view_rates.html',context )
  
    # donations = Donation.objects.select_related('donor', 'medium_of_waste').all()
    # return render(request, 'Donors/manage_donation.html', {'donations': donations})
