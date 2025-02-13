from django.urls import path
from . import views

urlpatterns = [
    path('homed/', views.homed, name='homed'),
    path('logind/', views.logind, name='logind'),
    path('registerd/', views.registerd, name='registerd'),
    path('logoutd/', views.logoutd, name='logoutd'),
    path('manage-delivery-partners/', views.manage_delivery_partners, name='manage_delivery_partners'),
    path('approve-delivery-partner/<int:partner_id>/', views.approve_delivery_partner, name='approve_delivery_partner'),
    path('reject-delivery-partner/<int:partner_id>/', views.reject_delivery_partner, name='reject_delivery_partner'),
    path('delivery-partner-details/<int:partner_id>/', views.delivery_partner_details, name='delivery_partner_details'),
    # Delivery process URLs
    path('available-orders/', views.available_orders, name='available_orders'),
    path('order/<int:order_id>/lock/', views.lock_order, name='lock_order'),
    path('order/<int:order_id>/accept/', views.accept_order, name='accept_order'),
    path('order/<int:order_id>/release/', views.release_order, name='release_order'),
    path('order/<int:order_id>/update-status/', views.update_delivery_status, name='update_delivery_status'),
    path('my-deliveries/', views.my_deliveries, name='my_deliveries'),
]