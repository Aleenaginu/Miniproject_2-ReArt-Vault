from django.http import JsonResponse
from django.shortcuts import render,redirect
from . forms import *
from .models import *
from adminclick.models import *

# Create your views here.
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
        username=request.POST.get('username')
        certificate=request.FILES.get('certificate')

        print(f"Received username:{username} ")
        print(f"Received certificate:{certificate} ")

        if not username:
            return JsonResponse({'success': False, 'error':'Username not provided'})
        try:
            artist = Artist.objects.get(user_username=username)
        except Artist.DoesNotExist:
            print(f"Artist with username {username} not found.")
            return JsonResponse({'success': False, 'error':'Artist not found'})
        
        if not artist.is_approved:
            if certificate:
                artist.certificate = certificate
                artist.save()
                return redirect('login')
            else:
                return JsonResponse({'success': False, 'error':'No file uploaded'})
        else:
            return JsonResponse({'success':False, 'error':'Account is already approved'})
    return JsonResponse({'success':False, 'error': 'Invalid request method'})

def notifications(request):
    notifications = Notification.objects.filter(user=request.user)
    notifications.update(is_read=True)
    return render(request, 'artist/notifications.html', {'notifications': notifications})


def view_ratesartist(request):
  
     artist=request.user.artist
     mediums=MediumOfWaste.objects.all()
     context={
         'mediums': mediums , 
         'artist':artist
     }
     return render(request, 'artist/view_rates.html',context )