from django.http import JsonResponse
from django.shortcuts import render,redirect,get_object_or_404
from . forms import *
from .models import *
from django.contrib.auth.decorators import login_required
from adminclick.models import *
from donors.models import *
from django.core.mail import send_mail
from django.contrib import messages
from django.views.decorators.cache import never_cache

# Create your views here.
@login_required
@never_cache

def artist_dashboard(request):
       if request.user.is_authenticated and request.user.artist:
        artist= request.user.artist
        unread_count = Notification.objects.filter(user=request.user, is_read=False).count()
        context={
            'artist':artist,
            'unread_count':unread_count
        }
        return render(request,'artist/dashboard.html',context)
def profile_update(request):
    if request.method=='POST':
        user_form_artist = UserUpdateFormArtist(request.POST,instance=request.user)
        profile_form_artist=ProfileUpdateFormArtist(request.POST, request.FILES,instance=request.user.artist)

        if user_form_artist.is_valid() and profile_form_artist.is_valid():
            user_form_artist.save()
            profile_form_artist.save()
            return redirect ('artist_dashboard')
    else:
        if request.user.is_authenticated and request.user.artist:
         artist= request.user.artist
        user_form_artist = UserUpdateFormArtist(instance=request.user)
        profile_form_artist=ProfileUpdateFormArtist(instance=request.user.artist)
    context = {
     'user_form_artist': user_form_artist,
     'profile_form_artist':profile_form_artist,
      'artist':artist,
     
    }
    return render(request,'artist/updateprofile.html', context)

def pending_approval(request):
    return render(request, 'artist/pending_approval.html',{'user':request.user})


def upload_certificate(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        certificate = request.FILES.get('certificate')

        if not username:
            return JsonResponse({'success': False, 'error': 'Username not provided'})

        try:
            artist = Artist.objects.get(user__username=username)
        except Artist.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Artist not found'})

        # Check if the artist is already approved
        if artist.is_approved:
            return JsonResponse({'success': False, 'error': 'Account is already approved'})

        # Check if the certificate is provided in the request
        if certificate:
            # Check if the artist already has a certificate uploaded
            if artist.certificate:
                return JsonResponse({'success': False, 'error': 'Certificate already uploaded. Verification is under processing.'})
            else:
                # If no certificate is uploaded yet, save the new certificate
                artist.certificate = certificate
                artist.save()
                return JsonResponse({'success': True, 'message': 'Certificate uploaded successfully. Verification is under processing.'})
        else:
            return JsonResponse({'success': False, 'error': 'No file uploaded'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


def notifications(request):
    notifications = Notification.objects.filter(user=request.user)
    notifications.update(is_read=True)
    return render(request, 'artist/notifications.html', {'notifications': notifications})

@login_required
@never_cache

def view_ratesartist(request):
  
     artist=request.user.artist
     mediums=MediumOfWaste.objects.all()
     unread_count = Notification.objects.filter(user=request.user, is_read=False).count()
     context={
         'mediums': mediums , 
         'artist':artist,
        'unread_count':unread_count
     }
     return render(request, 'artist/view_rates.html',context )

@login_required
@never_cache

def express_interest(request,donation_id):
    donation=get_object_or_404(Donation, id=donation_id)
    artist = get_object_or_404(Artist, user=request.user)
    donor=donation.donor


    interest_request= InterestRequest.objects.create(
        artist=artist,
        donor=donor,
        donation=donation,
    )

    send_mail(
        'Interest in Donation',
        f'Artist {artist.user.username} has expressed interest in your donation:\n'
        f'Medium : {donation.medium_of_waste.name}\n'
        f'Quantity : {donation.quantity}\n'
        f'Location : {donation.location}\n'
        f'Please visit your dashboard and furthur proceed with accept or reject',
        'reartvault@gmail.com',
        [donor.user.email],
        fail_silently=False,
    )

    DonorNotification.objects.create(
        donor=donor,
        message=f'Artist{artist.user.username} has expressed interest in your donotation',
        interest_request=interest_request
    )
    return redirect('notifications')

@login_required
@never_cache
def artist_interest_status(request):
    artist=request.user.artist
    interests=InterestRequest.objects.filter(artist=artist).select_related('donation','donor')
    return render(request,'artist/interest_status.html',{'interests':interests})

def delete_notification(request, notification_id):
    notification=get_object_or_404(Notification,id=notification_id,user=request.user)
    notification.delete()
    return render('notifications')