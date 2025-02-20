from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.models import User
import requests
from accounts.models import Donors,Artist

import os
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from django.conf import settings
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import EmailMessage
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth.tokens import default_token_generator
from django.urls import reverse

from adminclick.models import *

from .forms import CustomPasswordResetForm
from django.contrib.auth.forms import SetPasswordForm
from django.db import transaction
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import never_cache

import logging
import json
import cv2

logger = logging.getLogger(__name__)

def donorRegister(request):
    request.session['user_role']='donor'
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        face_data = request.POST.get('face_data')  # Get face data from form
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        
        if password == confirm_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username Already Taken')
            elif User.objects.filter(email=email).exists():
                messages.error(request, 'Email Already Taken')
            elif Donors.objects.filter(phone=phone).exists():
                messages.error(request, 'Phone already registered')
            else:
                with transaction.atomic():
                    try:
                        # Create user and donor
                        user = User.objects.create_user(username=username, email=email, password=password)
                        Donors.objects.create(user=user, phone=phone)
                        
                        # Store face encoding if provided
                        if face_data:
                            from face_auth.views import base64_to_image, get_face_encoding
                            
                            # Convert base64 to image
                            image = base64_to_image(face_data)
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            
                            # Detect face
                            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                            face_cascade = cv2.CascadeClassifier(cascade_path)
                            faces = face_cascade.detectMultiScale(
                                gray,
                                scaleFactor=1.1,
                                minNeighbors=6,
                                minSize=(100, 100),
                                maxSize=(500, 500)
                            )
                            
                            if len(faces) == 1:
                                # Get face encoding
                                face_encoding = get_face_encoding(image, faces[0])
                                
                                # Store in FaceEncoding model
                                from face_auth.models import FaceEncoding
                                face_record = FaceEncoding.objects.create(
                                    user=user,
                                    encoding=json.dumps(face_encoding.tolist())
                                )
                                logger.info(f"Face encoding stored for user: {username}")
                            else:
                                logger.warning(f"Face detection failed during registration for user: {username}")
                                
                        messages.success(request, 'Registered successfully')
                        return redirect('userlogin')
                    except Exception as e:
                        logger.error(f"Error during registration: {str(e)}")
                        messages.error(request, 'Registration failed. Please try again.')
                        # Rollback will happen automatically due to transaction.atomic()
        else:
            messages.error(request, 'Passwords do not match')
        return redirect('donor_register')
    return render(request, 'Donors/Register.html')


def Userprofile(request):
    if request.user.is_authenticated and request.user.donors:
     donor= request.user.donors
     return render(request, 'Donors/userprofile.html',{'donor':donor})
     


def UserLogout(request):
    logout(request)
    request.session.flush()
    messages.success(request, 'Logged out successfully')
    return redirect('userlogin')

@never_cache
def UserLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        logger.debug(f"Login attempt for username: {username}")
        
        # Check if the user exists
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            logger.warning(f"User does not exist: {username}")
            messages.error(request, 'Invalid credentials')
            return render(request, 'login.html')
        
        # Attempt to authenticate
        user = authenticate(username=username, password=password)
        
        if user is not None:
            logger.debug(f"User authenticated: {username}")
            
            if hasattr(user, 'donors'):
                login(request, user)
                messages.success(request, 'Login successful')
                return redirect('donor_dashboard')
            elif hasattr(user, 'artist'):
                if user.artist.is_approved:
                    login(request, user)
                    messages.success(request, 'Login successful')
                    return redirect('artist_dashboard')
                elif user.artist.certificate:
                    messages.warning(request, 'You uploaded your certificate, but the verification is still under processing.')
                    return redirect('userlogin')

                else:
                    messages.error(request, 'Your account is pending approval. Please upload your certificate')
                    return redirect('pending_approval')
            else:
                logger.warning(f"User has no donor or artist profile: {username}")
                messages.error(request, 'Invalid user type')
        else:
            logger.warning(f"Authentication failed for username: {username}")
            messages.error(request, 'Invalid credentials')
        
    return render(request, 'login.html')


def sign_in(request):
    return render(request, 'Donors/Register.html', {
        'google_client_id': settings.GOOGLE_OAUTH_CLIENT_ID
    })


@csrf_exempt
def auth_receiver(request):
    token = request.POST['credential']
    try:
        user_data = id_token.verify_oauth2_token(token, google_requests.Request(), settings.GOOGLE_OAUTH_CLIENT_ID)
    except ValueError:
        return HttpResponse(status=403)
    request.session['user_data'] = user_data
    return redirect('sign_in')


def sign_out(request):
    del request.session['user_data']
    messages.success(request, 'Signed out successfully')
    return redirect('sign_in')


def google_auth_callback(request):
    code = request.GET.get('code')
    if not code:
        return JsonResponse({'error': 'No code provided'}, status=400)
    
    token_url = 'https://oauth2.googleapis.com/token'
    token_data = {
        'code': code,
        'client_id': settings.GOOGLE_OAUTH_CLIENT_ID,
        'client_secret': settings.GOOGLE_OAUTH_CLIENT_SECRET,
        'redirect_uri': 'http://localhost:8000/auth-receiver/',
        'grant_type': 'authorization_code'
    }
    token_response = requests.post(token_url, data=token_data)
    token_json = token_response.json()
    
    if 'error' in token_json:
        return JsonResponse({'error': token_json['error']}, status=400)
    
    access_token = token_json['access_token']
    user_info_url = 'https://www.googleapis.com/oauth2/v2/userinfo'
    user_info_response = requests.get(user_info_url, headers={'Authorization': f'Bearer {access_token}'})
    user_info = user_info_response.json()

    # Implement user authentication logic here
    try:
        user = User.objects.get(email=user_info['email'])
    except User.DoesNotExist:
        user = User.objects.create_user(username=user_info['email'], email=user_info['email'])
        user.save()
    login(request, user)
    messages.success(request, 'Login successful')
    return redirect('home')


def custom_password_reset(request):
    if request.method == 'POST':
        form = CustomPasswordResetForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            try:
                user = User.objects.get(email=email)
                current_site = get_current_site(request)
                mail_subject = 'Reset your password'
                uidb64 = urlsafe_base64_encode(force_bytes(user.pk))
                token = default_token_generator.make_token(user)
                reset_link = reverse('custom_password_reset_confirm', kwargs={'uidb64': uidb64, 'token': token})
                reset_url = f"http://{current_site.domain}{reset_link}"
                message = render_to_string('registration/password_reset_email.html', {
                    'user': user,
                    'domain': current_site.domain,
                    'uidb64': uidb64,
                    'token': token,
                    'reset_url': reset_url
                })
                email_message = EmailMessage(mail_subject, message, 'webmaster@localhost', [email])
                email_message.content_subtype = 'html'
                email_message.send()
                messages.success(request, 'A link to reset your password has been sent to your email.')
                return redirect('custom_password_reset_done')
            except User.DoesNotExist:
                messages.error(request, 'No user is associated with this email address.')
    else:
        form = CustomPasswordResetForm()
    return render(request, 'registration/password_reset_form.html', {'form': form})


def custom_password_reset_confirm(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    
    if user is not None and default_token_generator.check_token(user, token):
        if request.method == 'POST':
            form = SetPasswordForm(user, request.POST)
            if form.is_valid():
                form.save()
                messages.success(request, 'Your password has been set. You can now log in with the new password.')
                return redirect('userlogin')
        else:
            form = SetPasswordForm(user)
        return render(request, 'registration/password_reset_confirm.html', {'form': form})
    else:
        messages.error(request, 'The reset link is invalid or has expired.')
        return redirect('custom_password_reset')


def custom_password_reset_done(request):
    return render(request, 'registration/password_reset_done.html')


# def artistRegister(request):
#     if request.method == 'POST':
#         username = request.POST.get('username')
#         email = request.POST.get('email')
#         phone = request.POST.get('phone')
#         profile_pic = request.FILES.get('profile_pic')
#         password = request.POST.get('password')
#         confirm_password = request.POST.get('confirm_password')
#         medium_id = request.POST.get('medium')
        
#         if password == confirm_password:
#             if User.objects.filter(username=username).exists():
#                 messages.error(request, 'Username Already Taken')
#             elif User.objects.filter(email=email).exists():
#                 messages.error(request, 'Email Already Taken')
#             elif Artist.objects.filter(phone=phone).exists():
#                 messages.error(request, 'Phone already registered')
#             else:
                        
#                 try:
#                     medium = MediumOfWaste.objects.get(id=medium_id)
#                 except MediumOfWaste.DoesNotExist:
#                     messages.error(request, 'Invalid medium selected')
#                     return render(request, 'artist_register.html', {'mediums': MediumOfWaste.objects.all()})
                
#                 with transaction.atomic():
#                     user = User.objects.create_user(username=username, email=email, password=password)
#                     Artist.objects.create(user=user, phone=phone, profile_pic=profile_pic,medium=medium)
#                 messages.success(request, 'Registered successfully')
#                 return redirect('userlogin')
#         else:
#             messages.error(request, 'Passwords do not match')
#         return redirect('artist_register')
#     mediums = MediumOfWaste.objects.all()
#     return render(request, 'artist/Register.html', {'mediums': mediums})

def artistRegister(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        profile_pic = request.FILES.get('profile_pic')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        if password == confirm_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username Already Taken')
            elif User.objects.filter(email=email).exists():
                messages.error(request, 'Email Already Taken')
            elif Artist.objects.filter(phone=phone).exists():
                messages.error(request, 'Phone already registered')
            else:
                with transaction.atomic():
                    user = User.objects.create_user(username=username, email=email, password=password)
                    artist = Artist.objects.create(user=user, phone=phone, profile_pic=profile_pic)

                    artist.save()
                    
                messages.success(request, 'Registered successfully. You can add mediums to your profile after approval.')
                return redirect('userlogin')
        else:
            messages.error(request, 'Passwords do not match')
        return redirect('artist_register')

    return render(request, 'artist/Register.html')
                                                    
# def UserLoginArtist(request):
#     if request.method == 'POST':
#         username = request.POST.get('username')
#         password = request.POST.get('password')
#         user = authenticate(username=username, password=password)
        
#         if user is not None and hasattr(user, 'artist'):
#             login(request, user)
#             messages.success(request, 'Login successful')
#             return redirect('artist_dashboard')
#         else:
#             messages.error(request, 'Invalid credentials')
#             return render(request, 'login.html')
#     return render(request, 'login.html')

def UserprofileArtist(request):
    if request.user.is_authenticated and request.user.artist:
     artist= request.user.artist
     return render(request, 'artist/userprofile.html',{'artist':artist})

@login_required(login_url='userlogin')
@never_cache
def login_redirect(request):
    user = request.user
    user_role = request.session.pop('user_role', None)
    if user is not None and hasattr(user, 'donors'):
        messages.success(request, 'Login successful')
        return redirect('donor_dashboard')
    elif user is not None and hasattr(user, 'artist'):
        messages.success(request, 'Login successful')
        return redirect('artist_dashboard')
    else:
        messages.error(request, 'Invalid credentials')
        return render(request, 'login.html')