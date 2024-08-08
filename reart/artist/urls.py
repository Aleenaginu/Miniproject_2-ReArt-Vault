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
    path('express-interest/<int:donation_id>/', views.express_interest, name='express_interest'),
    path('interest-status/', views.artist_interest_status, name='artist_interest_status'),
    path('delete-notification/<int:notification_id>/', views.delete_notification, name='delete_notification'),
]
