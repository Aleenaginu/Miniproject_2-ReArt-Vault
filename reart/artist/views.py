from django.shortcuts import render,redirect
from . forms import *
from .models import *

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

def notifications(request):
    notifications = Notification.objects.filter(user=request.user)
    notifications.update(is_read=True)
    return render(request, 'artist/notifications.html', {'notifications': notifications})