from django.shortcuts import render,redirect, get_object_or_404

from .models import MediumOfWaste
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from accounts.models import *
from donors.models import *
from django.core.mail import send_mail
from artist.models import Notification


# Create your views here.
def UserLoginadmin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        
        if user is not None and hasattr(user, 'adminclick'):
            login(request, user)
            messages.success(request, 'Login successful')
            return redirect('admin_dashboard')
        else:
            messages.error(request, 'Invalid credentials')
            return render(request, 'admin/adminlogin.html')
    return render(request, 'admin/adminlogin.html')



def admin_dashboard(request):
       if request.user.is_authenticated and request.user.adminclick:
        adminclick= request.user.adminclick
        return render(request,'admin/dashboard.html',{'adminclick':adminclick})
       

def approve_artists(request):
    adminclick= request.user.adminclick
    pending_artists = Artist.objects.filter(is_approved=False)
    artists = Artist.objects.all()
    context={
        'pending_artists': pending_artists,
        'artists': artists,
        'adminclick':adminclick

    }

    return render(request, 'admin/approve_artists.html', context)

@login_required
def approve_artist(request, artist_id):

    artist = get_object_or_404(Artist, id=artist_id)
    artist.is_approved = True
    artist.save()

    return redirect('approve_artists')

@login_required
def reject_artist(request, artist_id):

    artist = get_object_or_404(Artist, id=artist_id)
    artist.delete()

    return redirect('approve_artists')

def artist_details(request, artist_id):
    adminclick= request.user.adminclick
    artist=get_object_or_404(Artist, id=artist_id)
    context={
        'artist': artist,
        'adminclick': adminclick
    }
    return render(request, 'admin/artist_details.html', context)


@login_required
def add_medium_of_waste(request):
    adminclick= request.user.adminclick
    context={
        'adminclick':adminclick,
    }

    if request.method == 'POST':
        
        name = request.POST.get('name')
        description = request.POST.get('description')
        rate = request.POST.get('rate')
        
        
        if MediumOfWaste.objects.filter(name=name).exists():
            messages.info(request,"Already Exist")
            return render(request, 'admin/add_medium_of_waste.html')

        
        MediumOfWaste.objects.create(name=name, description=description, rate=rate)
        messages.success(request,"Medium of waste added successfully")
        return redirect('admin_dashboard')  

    return render(request, 'admin/add_medium_of_waste.html', context)

def set_rates(request):
    adminclick=request.user.adminclick
    if request.method=='POST':
        for medium_id, rate in request.POST.items():
            if medium_id.startswith('rate_'):
                try:
                    medium_id=int(medium_id.split('_')[1])
                    medium=MediumOfWaste.objects.get(id=medium_id)
                    medium.rate=float(rate)
                    medium.save()
                except (ValueError, MediumOfWaste.DoesNotExist):
                    continue
        messages.success(request,'Rates set successfully')
        return redirect('set_rates')
    mediums=MediumOfWaste.objects.all()
    context={
        'mediums': mediums,
        'adminclick': adminclick
    }
    return render(request,'admin/set_rates.html',context)

def donation_listview(request):
    adminclick= request.user.adminclick
    donations = Donation.objects.select_related('donor', 'medium_of_waste').all()
    context={
    'donations': donations,
    'adminclick':adminclick

    }
    return render(request, 'admin/donation_list.html', context)

def donation_detail(request, pk):
    donation = get_object_or_404(Donation, pk=pk)

    if request.method == 'POST':
        status = request.POST.get('status')
        if status == 'rejected':
            donation.delete()
            return redirect('donation_listview')
        elif status in dict(Donation.STATUS_CHOICES):
            donation.status = status
            donation.save()
            
            if status == 'accepted':
                # Notify relevant artists
                artists = Artist.objects.filter(medium=donation.medium_of_waste, is_approved=True)
                for artist in artists:
                    # Send email
                    send_mail(
                        'Donation Accepted',
                        f'A new waste donation in your medium has been accepted:\n'
                        f'Medium: {donation.medium_of_waste.name}\n'
                        f'Quantity: {donation.quantity}\n'
                        f'Location: {donation.location}\n',
                        'your-email@example.com',
                        [artist.user.email],
                        fail_silently=False,
                    )
                    # Optional: Create in-app notification
                    Notification.objects.create(
                        user=artist.user,
                        message=f'New waste donation in your medium: {donation.medium_of_waste.name}. Quantity: {donation.quantity}, Location: {donation.location}.'
                    )

            return redirect('donation_listview')

    return render(request, 'admin/donation_detail.html', {'donation': donation})

# def artist_list(request):
#     artists = Artist.objects.all()
#     return render(request, 'admin/artist_list.html', {'artists': artists})