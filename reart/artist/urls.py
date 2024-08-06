from django.contrib import admin
from django.urls import path,include
from . import views
urlpatterns = [
    path('artist_dashboard',views.artist_dashboard,name='artist_dashboard'),
    path('profile/updateartist/',views.profile_update,name='profile_update'),
    path('notifications/',views.notifications,name='notifications'),
    path('pending-approval',views.pending_approval,name='pending_approval'),
    path('upload-certificate',views.upload_certificate,name='upload_certificate'),
     path('view-ratesartist', views.view_ratesartist, name='view_ratesartist'),
]
