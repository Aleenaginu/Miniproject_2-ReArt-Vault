from django.shortcuts import render, redirect, get_object_or_404
from django.core.mail import send_mail
from django.http import HttpResponseForbidden
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import never_cache
from .forms import DonationForm, UserUpdateForm, ProfileUpdateForm
from accounts.models import Donors
from .models import Donation, DonorNotification, DonationImage
from artist.models import MediumOfWaste, InterestRequest

@login_required
@never_cache
def donor_dashboard(request):
    try:
        # Check if user has a donor profile
        if not hasattr(request.user, 'donors'):
            # Try to create a donor profile if the user has a customer profile
            if hasattr(request.user, 'customers'):
                try:
                    # Get phone from customer profile if available
                    phone = request.user.customers.phone if hasattr(request.user.customers, 'phone') else 99999999
                    
                    # Create donor profile
                    donor = Donors.objects.create(
                        user=request.user,
                        phone=phone
                    )
                    messages.success(request, "Donor profile created successfully.")
                except Exception as e:
                    print(f"Error creating donor profile: {str(e)}")
                    messages.error(request, "Could not create donor profile. Please contact support.")
                    return redirect('userlogin')
            else:
                messages.error(request, "You don't have a donor profile. Please register as a donor first.")
                return redirect('donor_register')
        
        # Now the user should have a donor profile
        donor = request.user.donors
        total_donations = Donation.objects.filter(donor=donor).count()
        approved_donations = Donation.objects.filter(donor=donor, status ='accepted').count()

        context={
            'donor':donor,
            'total_donations':total_donations,
            'approved_donations':approved_donations
            }
        return render(request, 'Donors/dashboard.html', context)
    except Exception as e:
        # Log the error for debugging
        print(f"Error in donor_dashboard: {str(e)}")
        messages.error(request, "An error occurred while accessing your donor dashboard. Please try again.")
        return redirect('userlogin')

@login_required
@never_cache
def donor_update(request):
    if request.method == 'POST':
        user_form = UserUpdateForm(request.POST, instance=request.user)
        profile_form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.donors)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            return redirect('donor_dashboard')
    else:
        if request.user.is_authenticated and request.user.donors:
            donor = request.user.donors
        user_form = UserUpdateForm(instance=request.user)
        profile_form = ProfileUpdateForm(instance=request.user.donors)
    context = {
        'user_form': user_form,
        'profile_form': profile_form,
        'donor': donor,
    }
    return render(request, 'Donors/updateprofile.html', context)

@login_required
@never_cache
def donate_waste(request):
    if not hasattr(request.user, 'donors'):
        return HttpResponseForbidden("You are not allowed to access this page.")
    
    donor = request.user.donors
    total_donations = Donation.objects.filter(donor=donor).count()
    approved_donations = Donation.objects.filter(donor=donor, status ='accepted').count()
    context = {
        'mediums': MediumOfWaste.objects.all(),
        'donor': donor,
        'total_donations':total_donations,
        'approved_donations':approved_donations
    }

    if request.method == 'POST':
        medium_id = request.POST.get('medium_of_waste')
        quantity = request.POST.get('quantity')
        location = request.POST.get('location')
        images = request.FILES.getlist('images')  

        try:
            medium_of_waste = MediumOfWaste.objects.get(id=medium_id)
        except MediumOfWaste.DoesNotExist:
            messages.error(request, 'Invalid medium of waste selected.')
            return render(request, 'Donors/donate_waste.html', {'mediums': MediumOfWaste.objects.all()})

        donation = Donation.objects.create(
            donor=request.user.donors,
            medium_of_waste=medium_of_waste,
            quantity=quantity,
            location=location,
            status='pending'
        )

     
        for image in images:
            DonationImage.objects.create(donation=donation, image=image)
   
        messages.success(request, 'Donation recorded successfully.')
        return redirect('donor_dashboard')

    return render(request, 'Donors/donate_waste.html',context)

@login_required
@never_cache
def view_donations(request):
    donations = Donation.objects.filter(donor=request.user.donors)
    donor = request.user.donors
    total_donations = Donation.objects.filter(donor=donor).count()
    approved_donations = Donation.objects.filter(donor=donor, status ='accepted').count()
    context = {
        'donations': donations,
        'donor':donor,
        'total_donations':total_donations,
        'approved_donations':approved_donations
        
    }
    return render(request, 'Donors/manage_donation.html', context)

@login_required
@never_cache
def edit_donation(request, donation_id):
    donation=get_object_or_404(Donation, id=donation_id, donor=request.user.donors)

    if request.method == 'POST':
        form=DonationForm(request.POST, request.FILES, instance=donation)
        if form.is_valid():
            form.save()
            messages.success(request, "Donation updated successfully!")
            return redirect('donor_dashboard')
        
    else:
        form = DonationForm(instance=donation)

    return render(request, 'Donors/edit_donation.html',{'form':form})

def delete_donation(request,donation_id):
    #  donor = get_object_or_404(Donors, user=request.user)
     donation = get_object_or_404(Donation, id=donation_id, donor=request.user.donors)
     if request.method == 'POST':
        donation.delete()
        messages.success(request, "Donation deleted successfully!")
        return redirect('view_donations')

     return redirect('view_donations')
    


@login_required
@never_cache
def view_rates(request):
    donor = request.user.donors
    total_donations = Donation.objects.filter(donor=donor).count()
    approved_donations = Donation.objects.filter(donor=donor, status ='accepted').count()
    mediums = MediumOfWaste.objects.all()
    context = {
        'mediums': mediums,
        'donor':donor,
        'total_donations':total_donations,
        'approved_donations':approved_donations
    }
    return render(request, 'Donors/view_rates.html', context)

@login_required
@never_cache
def donor_notifications(request):
    donor = get_object_or_404(Donors, user=request.user)
    notifications = DonorNotification.objects.filter(donor=donor).order_by('-created_at')
    notifications.update(is_read=True)
    total_donations = Donation.objects.filter(donor=donor).count()
    approved_donations = Donation.objects.filter(donor=donor, status ='accepted').count()
    context = {
        
        'donor':donor,
        'total_donations':total_donations,
        'approved_donations':approved_donations,
        'notifications': notifications
    }
    return render(request, 'Donors/donor_notifications.html', context)

@login_required
@never_cache

def handle_interest_request(request, notification_id):
    notification = get_object_or_404(DonorNotification, id=notification_id)
    interest_request = get_object_or_404(InterestRequest, id=notification.interest_request.id)

    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'accept':
            interest_request.status = 'accepted'
            interest_request.save()
            send_mail(
                'Interest Accepted',
                f'Your interest in the donation of {interest_request.donation.medium_of_waste.name} has been accepted',
                'reartvault@gmail.com',
                [interest_request.artist.user.email],
                fail_silently=False,
            )
            messages.success(request, 'The interest request has been accepted and the artist has been notified')
        elif action == 'reject':
            # Update status to rejected instead of deleting
            interest_request.status = 'rejected'
            interest_request.save()
            send_mail(
                'Interest Rejected',
                f'Your interest in the donation of {interest_request.donation.medium_of_waste.name} has been rejected',
                'reartvault@gmail.com',
                [interest_request.artist.user.email],
                fail_silently=False,
            )
            messages.success(request, 'The interest request has been rejected and the artist has been notified')

        notification.is_read = True
        notification.save()
    return redirect('donor_notifications')


