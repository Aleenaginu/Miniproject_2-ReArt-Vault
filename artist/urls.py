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
    path('artist/add-mediums/', views.add_mediums, name='add_mediums'),
    path('create-payment/<int:interest_id>/', views.create_payment, name='create_payment'),
    path('verify-payment/<int:payment_id>/', views.verify_payment, name='verify_payment'),
    path('payment-success/', views.payment_success, name='payment_success'),
    path('payment-failed/', views.payment_failed, name='payment_failed'),
    path('payment-callback/', views.payment_callback, name='payment_callback'),
    path('receipt/<int:interest_id>/', views.view_receipt, name='view_receipt'),
    path('buyed-items/', views.buyed_items, name='buyed_items'),
    path('buyed-items/<int:product_id>/', views.buyed_items, name='buyed_items_with_id'),
    path('get-recommendations/', views.get_recommendations_view, name='get_recommendations'),
    path('get-donation-details/<int:donation_id>/', views.get_donation_details, name='get_donation_details'),


    #shop
    path('shopdash/',views.shopdash,name='shopdash'),
    path('artist_shop/',views.artist_shop,name='artist_shop'),
    path('add_product/',views.add_product,name='add_product'),
    path('edit_product/<int:product_id>/',views.edit_product,name='edit_product'),

    path('order_notifications/',views.order_notifications, name='order_notifications'), 
    path('order/<int:order_id>/update_order_status/', views.update_order_status, name='update_order_status'),
    path('update-artist-address/', views.update_artist_address, name='update_artist_address'),
]
