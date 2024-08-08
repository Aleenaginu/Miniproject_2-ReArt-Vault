from django.contrib import admin
from django.urls import path,include
from . import views
urlpatterns = [
    path('donor_dashboard',views.donor_dashboard,name='donor_dashboard'),
    path('donor/update/',views.donor_update,name='donor_update'),
   path('donate/', views.donate_waste, name='donate_waste'),
   path('view-donations/', views.view_donations, name='view_donations'),
    path('view-rates', views.view_rates, name='view_rates'),
    path('donor_notifications/', views.donor_notifications, name='donor_notifications'),
    path('handle-interest-request/<int:notification_id>/', views.handle_interest_request, name='handle_interest_request'),
]