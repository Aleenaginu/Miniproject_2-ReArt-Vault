from django.contrib import admin
from django.urls import path,include
from . import views
urlpatterns = [
     path('userloginadmin/',views.UserLoginadmin,name='userloginadmin'),
        path('admin_dashboard',views.admin_dashboard,name='admin_dashboard'),
        path('add-medium/', views.add_medium_of_waste, name='add_medium_of_waste'),

    path('approve-artists/', views.approve_artists, name='approve_artists'),
    path('approve-artist/<int:artist_id>/',views. approve_artist, name='approve_artist'),
    path('reject-artist/<int:artist_id>/', views.reject_artist, name='reject_artist'),
    
    path('donation_listview/', views.donation_listview, name='donation_listview'),
        path('donations/<int:pk>/', views.donation_detail, name='donation_detail'),
         path('artist-details/<int:artist_id>/',views. artist_details, name='artist_details'),
         path('set_rates', views.set_rates, name='set_rates'),
        # path('artist_list/', views.artist_list, name='artist_list'),

]