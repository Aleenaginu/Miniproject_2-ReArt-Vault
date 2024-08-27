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
from django.http import HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt 
from django.conf import settings


# Create your views here.
@login_required
@never_cache

def artist_dashboard(request):
       if request.user.is_authenticated and request.user.artist:
        artist= request.user.artist
        expressed_interest_count = Interest.objects.filter(artist=artist).count()
        accepted_interest_count = InterestRequest.objects.filter(
            artist=artist,
            status='accepted'
        ).count()
        unread_count = Notification.objects.filter(user=request.user, is_read=False).count()
        context={
            'artist':artist,
            'unread_count':unread_count,
            'expressed_interest_count':expressed_interest_count,
            'accepted_interest_count':accepted_interest_count
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
    artist= request.user.artist
    expressed_interest_count = Interest.objects.filter(artist=artist).count()
    accepted_interest_count = InterestRequest.objects.filter(
        artist=artist,
        status='accepted'
        ).count()
    unread_count = Notification.objects.filter(user=request.user, is_read=False).count()
    context={
            'artist':artist,
            'unread_count':unread_count,
            'expressed_interest_count':expressed_interest_count,
            'accepted_interest_count':accepted_interest_count,
            'notifications': notifications
    }
    return render(request, 'artist/notifications.html', context)

@login_required
@never_cache

def view_ratesartist(request):
  
     artist=request.user.artist
     mediums=MediumOfWaste.objects.all()
     expressed_interest_count = Interest.objects.filter(artist=artist).count()
     accepted_interest_count = InterestRequest.objects.filter(
            artist=artist,
            status='accepted'
        ).count()
     unread_count = Notification.objects.filter(user=request.user, is_read=False).count()
     context={
         'mediums': mediums , 
         'artist':artist,
        'unread_count':unread_count,
        'expressed_interest_count':expressed_interest_count,
        'accepted_interest_count':accepted_interest_count
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
    expressed_interest_count = Interest.objects.filter(artist=artist).count()
    accepted_interest_count = InterestRequest.objects.filter(
            artist=artist,
            status='accepted'
        ).count()
    interests=InterestRequest.objects.filter(artist=artist).select_related('donation','donor')
    context={
        'interests':interests,
        'expressed_interest_count':expressed_interest_count,
        'accepted_interest_count':accepted_interest_count,
        'artist':artist,
        'razorpay_api_key':settings.RAZORPAY_API_KEY,
        }
    return render(request,'artist/interest_status.html',context)

def delete_notification(request, notification_id):
    notification=get_object_or_404(Notification,id=notification_id,user=request.user)
    notification.delete()
    return render('notifications')

@login_required
@never_cache
def add_mediums(request):
    if not request.user.artist.is_approved:
        return HttpResponseForbidden("You are not allowed to access this page.")

    artist=request.user.artist
    expressed_interest_count = Interest.objects.filter(artist=artist).count()
    accepted_interest_count = InterestRequest.objects.filter(
            artist=artist,
            status='accepted'
        ).count()
    if request.method == 'POST':
        mediums = request.POST.getlist('mediums')
        custom_medium_name = request.POST.get('custom_medium')
        
        if custom_medium_name:
            custom_medium, created = MediumOfWaste.objects.get_or_create(name=custom_medium_name)
            artist.mediums.add(custom_medium)
        
        for medium_id in mediums:
            medium = MediumOfWaste.objects.get(id=medium_id)
            artist.mediums.add(medium)

        messages.success(request, 'Mediums added successfully.')
        return redirect('artist_dashboard')

    context = {
        'mediums': MediumOfWaste.objects.all(),
        'expressed_interest_count':expressed_interest_count,
        'accepted_interest_count':accepted_interest_count,
        'artist':artist,

    }
    return render(request, 'artist/add_mediums.html',context)

def create_payment(request, interest_id):
    interest_request = get_object_or_404(InterestRequest, id=interest_id)
    medium_of_waste = interest_request.donation.medium_of_waste
    amount = medium_of_waste.rate * interest_request.donation.quantity  

    client = razorpay.Client(auth=(settings.RAZORPAY_API_KEY, settings.RAZORPAY_API_SECRET_KEY))
    order_data = {
        'amount': int(amount * 100),  # Amount in paise
        'currency': 'INR',
        'payment_capture': '1'
    }
    order = client.order.create(order_data)

    payment = Payment.objects.create(
        artist=interest_request.artist,
        amount=amount,
        order_id=order['id']
    )

    context = {
        'order_id': order['id'],
        'razorpay_key': settings.RAZORPAY_API_KEY,
        'amount': amount,
        'interest_request': interest_request,
        'payment': payment
    }

    return render(request, 'artist/payment_page.html', context)


def payment_success(request):
    return render(request, 'artist/payment_success.html')

def payment_failed(request):
    return render(request, 'artist/payment_failed.html')

def verify_payment(request, payment_id):
    payment = get_object_or_404(Payment, id=payment_id)
    client = razorpay.Client(auth=(settings.RAZORPAY_API_KEY, settings.RAZORPAY_API_SECRET_KEY))

    if request.method == "POST":
        try:
            params_dict = {
                'razorpay_order_id': request.POST.get('razorpay_order_id'),
                'razorpay_payment_id': request.POST.get('razorpay_payment_id'),
                'razorpay_signature': request.POST.get('razorpay_signature')
            }

            client.utility.verify_payment_signature(params_dict)
            payment.payment_id = params_dict['razorpay_payment_id']
            payment.status = 'completed'
            payment.save()

            # Handle post-payment processing
            return redirect('payment_success')

        except razorpay.errors.SignatureVerificationError:
            payment.status = 'failed'
            payment.save()
            return redirect('payment_failed')

    return render(request, 'artist/payment_failed.html')


@csrf_exempt
def payment_callback(request):
    if request.method == "POST":
        client = razorpay.Client(auth=(settings.RAZORPAY_API_KEY, settings.RAZORPAY_API_SECRET_KEY))
        
        try:
            params_dict = {
                'razorpay_payment_id': request.POST['razorpay_payment_id'],
                'razorpay_order_id': request.POST['razorpay_order_id'],
                'razorpay_signature': request.POST['razorpay_signature']
            }

            client.utility.verify_payment_signature(params_dict)

            interest_id = request.POST['interest_id']
            interest = InterestRequest.objects.get(id=interest_id)
            interest.payment_status = 'paid'
            interest.save()

            return redirect('payment_success')

        except razorpay.errors.SignatureVerificationError:
            return redirect('payment_failed')
    
    return HttpResponseBadRequest()