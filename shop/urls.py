from django.urls import path
from . import views
from . import tracking_ws
from .views import submit_review


urlpatterns = [
    path('', views.shop_index, name='shop_index'),
    path('add_cart/<int:product_id>/', views.add_cart, name='add_cart'),
    path('cart/', views.cart, name='cart'),
    path('checkout/', views.checkout, name='checkout'),
    path('customer_register', views.customerRegister, name='customer_register'),
    path('customerlogin', views.customerLogin, name='customerlogin'),
    path('customerlogout', views.customerLogout, name='customerlogout'),
    path('customerprofile',views.customerprofile,name='customerprofile'),
    path('order-history/', views.order_history, name='order_history'),
    path('order-summary/<int:order_id>/', views.order_summary, name='order_summary'),
    path('order/<int:order_id>/items/', views.view_order_items, name='view_order_items'),
    path('order/<int:order_id>/update-status/', views.update_order_status, name='update_order_status'),
    path('order/<int:order_id>/status/', views.get_order_status, name='order_status'),
    path('payment-success/<int:order_id>/', views.payment_success, name='payment_success'),
    path('razorpay-callback/', views.razorpay_callback, name='razorpay_callback'),



    

    path('create-razorpay-order/', views.create_razorpay_order, name='create_razorpay_order'),
    path('view-your-orders/', views.view_your_orders, name='view_your_orders'),
    path('user-profile/', views.user_profile, name='user_profile'),
    path('add-address/', views.add_address, name='add_address'),
    path('edit-address/<int:address_id>/', views.edit_address, name='edit_address'),
    path('delete-address/<int:address_id>/', views.delete_address, name='delete_address'),
   
    # Wishlist URLs
    path('wishlist/', views.view_wishlist, name='wishlist'),
    path('wishlist/add/<int:product_id>/', views.add_to_wishlist, name='add_to_wishlist'),
    path('wishlist/remove/<int:product_id>/', views.remove_from_wishlist, name='remove_from_wishlist'),
    path('wishlist/add_to_cart/<int:product_id>/', views.add_to_cart_from_wishlist, name='add_to_cart_from_wishlist'),
    path('track-orders/',views.track_order_status,name='track_order_status'),
    path('search/', views.search_products, name='search_products'),
    path('face_auth/verify/', views.verify_face, name='verify_face'),
    path('face_auth/register/', views.register_face, name='register_face'),
    path('validate-aadhaar/', views.validate_aadhaar, name='validate_aadhaar'),
    path('<slug:category_slug>/', views.shop_index, name='product_by_category'),
    path('<slug:category_slug>/<slug:product_slug>/', views.product_detail, name='product_detail'),
    path('ws/track/<int:order_id>/', tracking_ws.tracking_handler.handle_request, name='order_tracking_ws'),
    path('submit_review/', views.submit_review, name='submit_review'),
]
