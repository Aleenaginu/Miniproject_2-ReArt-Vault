from django.contrib import admin
from django.urls import path,include
from . import views
from .views import google_auth_callback
urlpatterns = [
path('donor_register/',views.donorRegister,name='donor_register'),
 path('login/',views.UserLogin,name='userlogin'),
 path('userprofile/',views.Userprofile,name='userprofile'),
 path('logout/',views.UserLogout,name='userlogout'),
 path('', views.sign_in, name='sign_in'),
 path('sign-out', views.sign_out, name='sign_out'),
 path('auth-receiver', views.auth_receiver, name='auth_receiver'),
 path('google_auth_callback',views.google_auth_callback, name='google_auth_callback'),
 path('custom_password_reset/', views.custom_password_reset, name='custom_password_reset'),
 path('custom_password_reset_done/', views.custom_password_reset_done, name='custom_password_reset_done'),
 path('custom_password_reset_confirm/<uidb64>/<token>/', views.custom_password_reset_confirm, name='custom_password_reset_confirm'),
 path('artist_register/',views.artistRegister,name='artist_register'),
  path('userprofileartist/',views.UserprofileArtist,name='userprofileartist'),
  path('login-redirect/', views.login_redirect, name='login_redirect'),

#  path('loginartist/',views.UserLoginArtist,name='userloginartist'),

  
]