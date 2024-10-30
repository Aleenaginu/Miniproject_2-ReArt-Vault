from django.urls import path
from . import views

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


    path('<slug:category_slug>/', views.shop_index, name='product_by_category'),
    path('<slug:category_slug>/<slug:product_slug>/', views.product_detail, name='product_detail'),
]
